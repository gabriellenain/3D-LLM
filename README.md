# 3D-LLM Colab Benchmark

## Repository overview

This repository is a personal fork and adaptation of the original **3D-LLM** project, whose goal is to inject **3D representations into a large language model** for tasks such as object- and scene-level question answering, captioning, and grounded 3D understanding.

At a high level, the repo contains three main layers:

- **Core inference / model code** in `3DLLM_BLIP2-base/`  
  This is the main implementation of the released BLIP2–FlanT5-based 3D-LLM pipeline.

- **3D-language data and feature generation utilities** in `3DLanguage_data/` and `three_steps_3d_feature/`  
  These folders correspond to the original data-generation logic used to build 3D features from rendered multi-view observations.

- **Notebook-based experimentation**
  The present notebook is a lightweight, reproducible entry point designed to make the released model easier to test in practice.

## What this notebook does

This notebook is a **minimal, self-contained Colab benchmark** for the released 3D-LLM checkpoint.

Its purpose is **not** to reproduce the full original training pipeline.  
Instead, it provides a practical way to:

1. run the released model in a modern Colab environment,
2. make single-GPU inference tractable,
3. and probe what the model actually depends on at inference time.

In particular, the notebook focuses on whether the model is most sensitive to:

- the semantic point features,
- the alignment between features and 3D points,
- or the exact wording of the prompt.

## Notebook workflow

The notebook follows the pipeline below:

1. **Environment setup**
   - checks that Colab is running with a GPU,
   - clones this GitHub repository,
   - installs the dependencies needed for 3D-LLM inference.

2. **Checkpoint and data setup**
   - downloads the released pretrained checkpoint,
   - downloads a small Objaverse subset with precomputed point features,
   - normalizes the extracted directory layout for inference.

3. **Compatibility patches**
   - applies small code fixes so the original 3D-LLM / LAVIS stack runs cleanly in a modern Colab environment.

4. **Model loading**
   - loads the BLIP2–FlanT5-based 3D-LLM checkpoint,
   - prepares the text processor and inference utilities,
   - keeps the setup inference-oriented rather than training-oriented.

5. **Sanity-check inference**
   - runs the model on one example object,
   - verifies that the full 3D-to-text pipeline is working end to end.

6. **Robustness probes**
   - evaluates how stable the answers are under controlled feature perturbations,
   - evaluates how stable the answers are under semantically similar prompt rephrasings.

7. **Result aggregation**
   - summarizes the results with simple agreement metrics,
   - exports CSV files for later analysis, tables, and figures.

## Experimental framing

The benchmark is intentionally narrow and controlled.

We work with a fixed subset of Objaverse objects, each represented by:

- a point cloud $P \in \mathbb{R}^{n \times 3}$,
- aligned point features $F \in \mathbb{R}^{n \times d}$,

with a capped point budget per object to keep inference practical on a single GPU.

For each object, the notebook compares the baseline generation to generations obtained under two families of probes:

### 1. Feature robustness
The point coordinates are kept fixed while the input features are modified.

Typical variants include:
- `baseline`
- `gaussian_005`
- `gaussian_020`
- `shuffle_feat`
- `zero_feat`

These perturbations test whether the model depends on:
- precise low-level feature values,
- feature-to-point alignment,
- or simply the presence of global semantic content.

### 2. Prompt robustness
The 3D input is kept fixed while the user query is paraphrased.

This tests whether the model behaves like a stable 3D understanding system, or whether its outputs change significantly under small linguistic rewordings.

## Evaluation metrics

For each object, perturbation, and prompt, the generated answer is compared to the corresponding baseline answer.

The notebook reports simple consistency metrics such as:

- **Exact Match**
- **Sequence Similarity**
- **Jaccard similarity**

These metrics should be interpreted as **agreement-with-baseline** measures, not as absolute task accuracy.

## How to interpret this notebook

This notebook is best understood as an **inference-time probing benchmark**.

It does **not** claim to:
- retrain 3D-LLM,
- reproduce the original paper’s large-scale results,
- or fully isolate geometric reasoning.

Instead, it is meant to answer a more practical question:

> Once the released checkpoint is made runnable on a single GPU, what kinds of information does it appear to rely on most?

That makes this notebook useful for:
- reproducible sanity checks,
- lightweight scientific probing,
- qualitative inspection,
- and report-ready experiments on robustness and failure modes.

## Scope and limitations

This notebook only studies the **released inference stack** in a controlled object-level regime.

Therefore, conclusions from these experiments should be read carefully:

- they are **inference-only**,
- they depend on **precomputed features** rather than end-to-end training,
- and they probe **local robustness properties**, not full 3D reasoning ability.

A stable answer under mild perturbation does not prove deep geometric understanding; likewise, prompt sensitivity does not imply total failure.  
The notebook is designed as a compact empirical probe, not as a definitive benchmark of 3D intelligence.
