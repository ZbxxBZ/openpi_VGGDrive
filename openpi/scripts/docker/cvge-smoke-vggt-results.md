# CVGE VGGT two-stage smoke benchmark

Date: 2026-09-20

## Setup

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (GPU 1)
- PyTorch: 2.7.1+cu128
- Geometry backbone: original VGGT, image size 224
- Policy: pi0.5, BF16, action horizon 32
- Batch size: 1
- Gradient checkpointing: enabled
- Input: synthetic observation with all three Robotwin cameras valid
- Steps: 3 for each stage
- Transition: Stage 2 reused the Stage 1-updated model and created a fresh AdamW optimizer
- Stage 2 policy: full pi0.5 backbone plus CVGE/projector; VGGT remained frozen

This benchmark measures model compute and optimizer memory. It does not include real dataset decoding, data loading,
augmentation, checkpoint writing, or distributed training.

## Stage 1: adapter_only

- Trainable parameters: 89,791,488
- Initial CUDA allocated memory: 10.527 GiB
- Peak CUDA allocated memory: 11.274 GiB
- Peak CUDA reserved memory: 11.430 GiB
- Average step time: 0.7627 s (1.311 steps/s)

| Step | Loss | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | Peak allocated (GiB) | Peak reserved (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4.145076 | 0.5693 | 0.3592 | 0.0758 | 1.0043 | 11.224 | 11.277 |
| 1 | 4.056664 | 0.1815 | 0.3146 | 0.0325 | 0.5286 | 11.274 | 11.430 |
| 2 | 3.940319 | 0.1898 | 0.5336 | 0.0319 | 0.7553 | 11.274 | 11.430 |

Stage 1 validation passed: the terminal CVGE projection had a finite nonzero gradient on every step; all 18 CVGE
layers had finite nonzero inner gradients after the zero-initialized first step; VGGT, PaliGemma, and the action expert
remained frozen.

## Stage 2: full

- Trainable parameters: 3,706,544,008
- Initial CUDA allocated memory: 10.543 GiB
- Peak CUDA allocated memory: 36.369 GiB
- Peak CUDA reserved memory: 36.646 GiB
- Average step time: 0.8954 s (1.117 steps/s)

| Step | Loss | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | Peak allocated (GiB) | Peak reserved (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.817866 | 0.2077 | 0.5803 | 0.1908 | 0.9788 | 36.368 | 36.568 |
| 1 | 2.712236 | 0.2141 | 0.5254 | 0.1180 | 0.8574 | 36.369 | 36.645 |
| 2 | 2.179147 | 0.2118 | 0.5190 | 0.1193 | 0.8500 | 36.368 | 36.646 |

Stage 2 validation passed: PaliGemma, the action expert, and all 18 CVGE layers had finite nonzero gradients; VGGT
remained frozen.

## Stage 2 batch-size comparison

These direct Full Stage 2 smoke runs used three valid cameras and three optimizer steps. The average includes the first
step's CUDA and AdamW warm-up; the final-step value is a better indication of steady-state throughput.

| Batch | Peak allocated (GiB) | Peak reserved (GiB) | Average step (s) | Final step (s) | Final-step samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.369 | 36.646 | 0.8954 | 0.8500 | 1.176 |
| 2 | 36.381 | 36.609 | 1.2797 | 0.9288 | 2.153 |
| 3 | 36.418 | 36.588 | 1.1843 | 0.8837 | 3.395 |
| 16 | 36.409 | 37.939 | 2.1392 | 1.8580 | 8.611 |

Batch 16 provides the best throughput of the tested values and uses only about 1.35 GiB more reserved CUDA memory than
batch 3. Peak allocated tensor memory remains essentially unchanged because full-model parameters, gradients, and
AdamW states dominate this gradient-checkpointed configuration.
These synthetic runs do not establish the largest safe batch size for real training because prompt lengths, decoded
images, augmentations, and allocator fragmentation depend on the real dataset.
