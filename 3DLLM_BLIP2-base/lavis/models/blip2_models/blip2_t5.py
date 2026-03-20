import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.

    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        location_tokens = [f"<loc{i}>" for i in range(32768)]
        self.t5_tokenizer.add_special_tokens(
            {"additional_special_tokens": location_tokens}
        )

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for _, param in self.t5_model.named_parameters():
            param.requires_grad = False

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.register_buffer(
            "pos_embedding",
            pos_model(x).squeeze(0),
            persistent=False,
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt if prompt is not None else ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def _maybe_autocast(self, device):
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _augment_pc_embeds_with_position(self, pc_embeds, pc):
        device = pc_embeds.device
        dtype = pc_embeds.dtype

        pc = pc.long()
        pos_embedding = self.pos_embedding.to(device=device, dtype=dtype)

        all_pcs = torch.zeros_like(pc_embeds)
        for j in range(pc.shape[0]):
            pcs = []
            for i in range(3):
                pc_i = pc[j][:, i]
                pcs.append(pos_embedding[pc_i])
            pcs = torch.cat(pcs, dim=-1)
            all_pcs[j][:, :1407] = pcs

        return pc_embeds + 0.01 * all_pcs

    def _encode_pc(self, samples, add_pos=True):
        device = samples["pc_feat"].device

        with self._maybe_autocast(device):
            pc_embeds = samples["pc_feat"]

            if add_pos and "pc" in samples:
                pc_embeds = self._augment_pc_embeds_with_position(
                    pc_embeds, samples["pc"]
                )

            image_atts = torch.ones(
                pc_embeds.size()[:-1], dtype=torch.long, device=device
            )
            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1).to(
                device=device, dtype=pc_embeds.dtype
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=device)

        return inputs_t5, atts_t5, device

    def forward(self, samples):
        inputs_t5, atts_t5, device = self._encode_pc(samples, add_pos=True)

        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input,
            padding="longest",
            truncation=True,
            max_length=400,
            return_tensors="pt",
        ).to(device)

        output_tokens = self.t5_tokenizer(
            samples["answer"],
            padding="longest",
            truncation=True,
            max_length=300,
            return_tensors="pt",
        ).to(device)

        batch_input_tokens_input_ids = []
        batch_input_tokens_atts = []
        batch_atts_t5 = []
        batch_inputs_t5 = []

        for b, n in enumerate(samples["n_answers"]):
            batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
            batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
            batch_atts_t5 += [atts_t5[b]] * n
            batch_inputs_t5 += [inputs_t5[b]] * n

        batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
        batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
        batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
        batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

        encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id,
            -100,
        )

        with self._maybe_autocast(device):
            inputs_embeds = self.t5_model.encoder.embed_tokens(
                batch_input_tokens_input_ids
            )
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        inputs_t5, atts_t5, device = self._encode_pc(
            samples, add_pos=("pc" in samples)
        )

        if "prompt" in samples:
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * inputs_t5.size(0)
        else:
            assert len(prompt) == inputs_t5.size(0), (
                "The number of prompts must be equal to the batch size."
            )

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self._maybe_autocast(device):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        inputs_t5, atts_t5, device = self._encode_pc(samples, add_pos=True)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        prompt = self.prompt if not prompt else prompt

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        num_beams = 1

        with self._maybe_autocast(device):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )

        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=False
        )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)
            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)
        return model
