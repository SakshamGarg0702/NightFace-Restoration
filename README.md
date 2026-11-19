# NightFace-Restoration: Degradation-Robust Low-Light Face Identification via Text-Conditioned Biometric Restoration

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Base Paper](https://img.shields.io/badge/Base_Paper-IEEE_TIM_2024-green.svg)](https://ieeexplore.ieee.org/)
[![Restoration](https://img.shields.io/badge/Restoration-ECCV_2024-purple.svg)](https://arxiv.org/abs/2401.16468)

Official implementation of the **Two-Stage Language-Guided Biometric Restoration Framework** addressing extreme Poisson-Gaussian sensor noise and motion blur amplification in low-light face recognition.

---

## 📌 Executive Summary

Under extreme low-light conditions, face recognition pipelines suffer catastrophic failures. The baseline paper **Low-FaceNet** (*IEEE Transactions on Instrumentation and Measurement*, Vol. 73, March 2024) addresses underexposure through Retinex dynamic range enhancement:

```math
R(x) = \frac{S(x)}{I(x)}
```

However, in real-world camera sensor acquisition, underexposure is coupled with high-ISO Poisson shot noise, Gaussian read noise, and motion blur. Because Low-FaceNet brightens scenes by dividing pixel intensities by small predicted illumination values ($I(x) \ll 1$), it scales read noise variance quadratically by $1/I(x)^2$, amplifying noise by up to $10^4\times$ in deep shadows. On the degraded **CASIA-Test** benchmark, this leads to:
- Severe landmark jitter in **RetinaFace**, causing frequent face detection dropouts (`[!] No face detected`).
- Feature vector distortion in **FaceNet**, dropping top-1 closed-set identification accuracy to **12.6%**.

### 💡 Our Proposed Solution
We propose a **Two-Stage Decoupled Restoration-Identification Architecture**:
1. **Illumination Front-End:** Low-FaceNet expands global dynamic range and restores scene illumination.
2. **Text-Conditioned Restoration Back-End:** **InstructIR** (*ECCV 2024*), an instruction-conditioned NAFNet backbone, removes sensor noise and motion blur guided by natural language prompt: `"Deblur and denoise the image."`
3. **Biometric Identity Fine-Tuning:** InstructIR is fine-tuned using face-centric $320 \times 320$ crops with a compound loss combining $L_1$ pixel reconstruction and frozen MobileNet-FaceNet Cosine Identity loss:

```math
\mathcal{L}_{\text{Total}} = \mathcal{L}_{L1} + 1.0 \cdot \left(1 - \text{CosineSimilarity}\left(\text{FaceNet}(\hat{I}), \text{FaceNet}(I_{\text{clean}})\right)\right)
```

---

## 📊 Experimental Results (CASIA-Test Benchmark)

Evaluated under closed-set 1:N probe-to-gallery identification matching (500 probe images across 100 identities):

| Stage | Pipeline Configuration | Face Detection Rate (RetinaFace) | Top-1 Identity Accuracy (FaceNet) | $\Delta$ vs Baseline |
| :---: | :--- | :---: | :---: | :---: |
| **Stage 1** | **Low-FaceNet Alone** (*IEEE TIM 2024 Baseline*) | **68.0%** (160 faces missed due to noise) | **12.6%** (63 / 500 identities matched) | Baseline |
| **Stage 2** | **Low-FaceNet + Zero-Shot InstructIR** (*ECCV 2024*) | **100.0%** (500 / 500 faces successfully found) | **62.6%** (313 / 500 identities matched) | **+50.0%** |
| **Stage 3** | **Low-FaceNet + Biometrically Fine-Tuned InstructIR** (*Ours*) | **100.0%** (500 / 500 faces successfully found) | **70.2%** (351 / 500 identities matched) | **+57.6%** |

> **Understanding the Two Metrics:**
> - **Face Detection Rate (Step 1 - RetinaFace):** Did the detector locate a human face and its 5 landmarks in the image? Low-FaceNet's Retinex noise amplification caused **32% detection dropouts** (`[!] No face detected`). Our pipeline restores structural clarity, achieving a complete **100% detection rate** (zero dropouts).
> - **Top-1 Identity Accuracy (Step 2 - FaceNet):** Once the face is detected, was the correct person identified from the 100-identity gallery? Fine-tuning with biometric identity loss boosts recognition from **12.6% to 70.2%**.

---

## 🏗️ System Architecture

```text
[Dark Noisy Blurry Input (640x480)] ---> [Low-FaceNet Front-End] ---> [Brightened, High Noise Image]
                                                                                |
[Prompt: "Deblur and denoise the image."] ---> [MiniLM + LMHead]               |
                                                      |                         v
                                                      +-------------> [InstructIR Back-End]
                                                                                |
                                                                    [Restored Face (640x480)]
                                                                                |
                                                                                v
                                                                    [RetinaFace Alignment]
                                                                                |
                                                                                v
                                                                    [FaceNet 128-D Embedding]
                                                                                |
                                                                                v
                                                                    [1:N Closed-Set Matching]
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sakshamgarg0702/NightFace-Restoration.git
cd NightFace-Restoration
```

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

### 3. Pretrained Weights
- **InstructIR Weights:** Downloaded automatically via `huggingface_hub` from [marcosv/InstructIR](https://huggingface.co/marcosv/InstructIR) at runtime.
- **Low-FaceNet & FaceNet Weights:** Place `Low-FaceNet.pth`, `facenet_mobilenet.pth`, and `Retinaface_mobilenet0.25.pth` into the `checkpoint/` directory.

---

## 📁 Dataset Organization

Place the datasets in the `dataset/` directory according to the following structure:

```text
dataset/
├── CASIA-Test/
│   ├── normal/                            # 100 clean gallery portrait folders
│   ├── underexposed/                      # 100 underexposed folders
│   └── underexposed_noisy_blurry/         # 100 DEGRADED folders (dark + noise + blur)
│       ├── 001/
│       └── ...
│
└── LaPa-Face/                             # (Cross-Dataset Training Pairs)
    ├── normal/                            # Clean ground truth images
    └── degraded/                          # Synthetically degraded training images
```

> **Dataset Download Bundle:**  
> Download the pre-packaged dataset bundle and checkpoints directly from Kaggle:  
> 🔗 [Kaggle Dataset: Low-FaceNet Bundle](https://www.kaggle.com/datasets/sakshamgarg0702/lowfacenet-bundle)

---

## 💻 Running the Scripts

### Synthetic Degradation Generator
To synthetically generate the low-light, Poisson-Gaussian noise, and motion blur degradation on any clean image folder:
```bash
python scripts/degrade_dataset.py --input_dir dataset/LaPa-Face/normal --output_dir dataset/LaPa-Face/degraded
```

### Stage 1: Reproduce Baseline (Low-FaceNet Alone)
Runs Low-FaceNet on degraded CASIA-Test and calculates accuracy:
```bash
python scripts/run_stage1_baseline.py
# Expected Accuracy: ~12.6% (with [!] No face detected warnings)
```

### Stage 2: Two-Stage Zero-Shot (Low-FaceNet + InstructIR)
Restores Low-FaceNet outputs using off-the-shelf InstructIR with prompt `"Deblur and denoise the image."`:
```bash
python scripts/run_stage2_zeroshot.py
# Expected Accuracy: ~62.6% (100% face detection)
```

### Stage 3: Biometric Identity Fine-Tuning
Fine-tunes InstructIR using Face-Centric 320x320 crops and compound FaceNet Identity Loss:
```bash
python scripts/run_stage3_finetune.py --epochs 15 --batch_size 4
# Expected Accuracy: ~70.2%
```

---

## 📚 References

- **Base Paper:** Fan et al., *"Low-FaceNet: Face Recognition-Driven Low-Light Image Enhancement,"* **IEEE Transactions on Instrumentation and Measurement (TIM)**, Vol. 73, pp. 1-13, 2024.
- **Restoration Model:** Conde et al., *"High-Quality Image Restoration Following Human Instructions (InstructIR),"* **European Conference on Computer Vision (ECCV)**, 2024.
- **FaceNet:** Schroff et al., *"FaceNet: A Unified Embedding for Face Recognition and Clustering,"* **CVPR**, 2015.
- **RetinaFace:** Deng et al., *"RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild,"* **CVPR**, 2020.

---

## 📜 License
This project is released under the [MIT License](LICENSE).
