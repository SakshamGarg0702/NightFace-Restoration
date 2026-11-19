# Degradation-Robust Low-Light Face Identification via Text-Conditioned Biometric Restoration

**Abstract**—Facial identification pipelines deployed in nighttime surveillance face severe performance collapse under compound low-light degradations. While recent low-light enhancement architectures effectively restore scene dynamic range, their underlying mathematical formulations frequently overlook high-ISO physical sensor noise. In this paper, we examine the failure mode of the state-of-the-art model **Low-FaceNet** (*IEEE Transactions on Instrumentation and Measurement*, 2024). Low-FaceNet recovers surface reflectance via Retinex division $R = S / I$, where $I$ is an estimated illumination map clamped to $[10^{-4}, 1]$. Under realistic low-light acquisition where underexposure is coupled with Poisson-Gaussian sensor noise and motion blur, dividing by small illumination values ($I(x) \ll 1$) scales noise variance quadratically by $1/I^2$, amplifying high-frequency static by up to four orders of magnitude. This noise amplification induces severe landmark jitter in RetinaFace detectors, dropping closed-set identification accuracy on the degraded CASIA-Test benchmark to **12.6%**. To resolve this without sacrificing dynamic range recovery, we introduce a decoupled two-stage architecture pairing Low-FaceNet with **InstructIR** (*ECCV 2024*), a text-conditioned restoration back-end. Off-the-shelf zero-shot restoration eliminates detection dropouts and restores identification accuracy to **62.6%** (+50.0%). To bridge the remaining biometric gap, we fine-tune the restoration backbone using a compound objective combining $L_1$ pixel reconstruction with a frozen **MobileNet-FaceNet Cosine Identity Loss** over face-centric crops. This biometric alignment forces the network to retain high-frequency identity features, elevating top-1 identification accuracy to **70.2%** (+7.6% over zero-shot, +57.6% over baseline). Cross-recognizer evaluation on an unseen **ArcFace (ResNet-50)** backbone yields **65.2%** (+6.8% transfer gain), confirming that biometric alignment restores generalizable facial geometry rather than overfitting to model-specific representations.

**Index Terms**—Low-Light Face Identification, Retinex Decomposition, Noise Variance Scaling, InstructIR, Biometric Identity Loss, Cross-Recognizer Transfer.

---

## I. Introduction

Automated face identification in unconstrained night-vision and security surveillance environments is hampered by physical imaging limits. When camera sensors operate under sub-lux illumination, photon scarcity forces camera hardware to employ high analog sensor gain (ISO) and longer exposure times. Consequently, captured facial imagery suffers from a compound degradation profile: severe underexposure, signal-dependent Poisson shot noise, additive Gaussian electronic read noise, and camera-shake motion blur.

Recent research has prioritized specialized low-light enhancement networks. A prominent baseline, **Low-FaceNet** [1], utilizes a lightweight Retinex architecture guided by facial semantic segmentation to estimate pixel-wise illumination maps. While Low-FaceNet achieves high accuracy on clean underexposed imagery, its physical formulation exhibits an inherent vulnerability in real sensor environments. Specifically, Low-FaceNet recovers enhanced reflectance $R$ via:
$$R(x) = \frac{S(x)}{I(x)}$$
where $S(x)$ is the observed sensor signal and $I(x)$ is the predicted illumination map, bounded below by $\epsilon = 10^{-4}$. In physical camera sensors, $S(x)$ is corrupted by additive read noise $\eta_g$ and Poisson shot noise $\eta_p$. The division operation scales noise amplitude by $1/I(x)$ and noise variance by $1/I(x)^2$. In shadowed facial regions where $I(x) \approx 0.01$, the noise variance is scaled by up to $10,000\times$. While Retinex division does not amplify optical blur, the magnified sensor noise severely obscures high-frequency facial boundaries.

Empirical evaluation on realistic CASIA-Test benchmarks reveals two compounding failure modes:
1. **Landmark Detection Collapse:** Single-stage detectors like RetinaFace [2] rely on local gradient heatmaps to predict five canonical landmarks (pupils, nasal tip, oral commissures). Noise spikes disrupt these heatmaps, causing extensive detection dropouts (`[!] No face detected`).
2. **Embedding Distortion:** Residual noise displaces the 128-dimensional hyperspherical embeddings extracted by FaceNet [3], dropping closed-set top-1 identification accuracy to **12.6%**.

```
[Raw Dark Noisy Input] ---> [Low-FaceNet Front-End] ---> [Brightened, High Noise Image]
                                                                  |
[Text Prompt: "Deblur and denoise"] ---> [MiniLM + LMHead]        |
                                                  |               v
                                                  +---> [InstructIR Back-End]
                                                                  |
                                                      [Restored Face (640x480)]
                                                                  |
                                                  +---------------+---------------+
                                                  |                               |
                                                  v                               v
                                          [RetinaFace Alignment]          [FaceNet ID Loss]
                                                  |                         (Training Only)
                                                  v
                                          [1:N Gallery Matching]
```

To resolve this limitation, we present a decoupled two-stage restoration-identification pipeline. Rather than training a monolithic end-to-end network—which introduces conflicting gradients between global dynamic range expansion and local noise suppression—we decouple the operations:
* **Stage 1 (Illumination Front-End):** Low-FaceNet normalizes global luminance and dynamic range.
* **Stage 2 (Restoration Back-End):** InstructIR [4], an instruction-conditioned restoration model based on NAFNet [5], is conditioned on the prompt *"Deblur and denoise the image."* to suppress non-linear noise and deconvolve blur.
* **Biometric Identity Alignment:** We fine-tune InstructIR using a compound objective combining $L_1$ pixel reconstruction with a frozen MobileNet-FaceNet cosine identity loss over face-centric spatial crops.

Our contributions are summarized as follows:
- We formulate the noise variance scaling breakdown of Retinex-based enhancement models ($1/I^2$) and validate its impact on downstream facial biometric alignment.
- We show that zero-shot integration of text-conditioned restoration (InstructIR) completely eliminates detection dropouts and elevates identification accuracy from **12.6% to 62.6%** (+50.0%).
- We formulate a face-centric Biometric Identity Loss that aligns restoration with facial embedding geometries, achieving **70.2%** identification accuracy on CASIA-Test (+7.6% over zero-shot, +57.6% over baseline).
- We demonstrate cross-recognizer generalization on an unseen **ArcFace** model, achieving **65.2%** (+6.8% transfer gain), confirming that identity alignment restores generalizable facial structures rather than overfitting to FaceNet representations.

---

## II. Related Work

### A. Low-Light Image and Face Enhancement
Retinex theory assumes an image is the element-wise product of reflectance and illumination ($S = R \odot I$). Learning-based Retinex architectures such as RetinexNet [6] and KinD [7] predict decoupled maps to normalize illumination. Zero-DCE [8] formulates enhancement through dynamic curve parameter estimation. 

In face biometrics, Fan et al. introduced **Low-FaceNet** [1], integrating facial semantic parsing (DeepLabv3+) into the loss objective to preserve facial boundaries during illumination estimation. However, Low-FaceNet assumes clean sensor readouts. Under low exposure, high analog gain introduces sensor noise that violates this assumption, turning Retinex division into a digital noise amplifier.

### B. Text-Conditioned Image Restoration
Single-task restorers (e.g., DnCNN, MPRNet) lack adaptability across compound degradations. Recently, vision-language restoration architectures have leveraged textual instruction conditioning. **InstructIR** [4] adapts the NAFNet [5] backbone with cross-gating mechanisms conditioned on sentence embeddings generated via MiniLM [9]. By providing natural language prompts, InstructIR dynamically routes feature activations across deblurring and denoising pathways without requiring manual degradation classification.

### C. Biometric-Preserving Feature Losses
Pixel-level metrics ($L_1$, $L_2$, PSNR) uniformly weight intensity errors across spatial coordinates. However, human perceptual smoothness does not guarantee biometric machine recognizability: standard denoising tends to smooth high-frequency contours (nasal bridge, iris spacing, philtrum) essential for face recognition. Identity-preserving feature losses [10] counteract this by penalizing latent feature divergence extracted from pre-trained recognition backbones.

---

## III. Proposed Methodology

### A. Mathematical Formulation of Retinex Noise Amplification
Let an uncorrupted scene signal be $S^*(x) = R(x) \cdot I^*(x)$. Under low exposure, physical sensor acquisition introduces signal-dependent Poisson shot noise $\eta_p \sim \mathcal{P}(\lambda)$ and signal-independent Gaussian read noise $\eta_g \sim \mathcal{N}(0, \sigma^2)$:
$$S(x) = S^*(x) + \eta_p(S^*(x)) + \eta_g$$

Low-FaceNet estimates an illumination map $\hat{I}(x)$ bounded by $\epsilon = 10^{-4}$ and recovers enhanced reflectance $\hat{R}(x)$:
$$\hat{R}(x) = \frac{S(x)}{\hat{I}(x)} = R(x) \cdot \frac{I^*(x)}{\hat{I}(x)} + \frac{\eta_p + \eta_g}{\hat{I}(x)}$$

Assuming accurate illumination recovery ($\hat{I}(x) \approx I^*(x)$), the residual noise term is scaled by the inverse illumination:
$$\sigma^2_{\hat{R}}(x) = \frac{\sigma^2_{\eta_p} + \sigma^2_{\eta_g}}{\hat{I}(x)^2}$$

In deeply shadowed facial regions where $\hat{I}(x) \in [0.01, 0.05]$, the noise variance is scaled by factors of $400\times$ to $10,000\times$. This manifests as high-frequency salt-and-pepper noise, displacing edge gradients required for landmark detection.

### B. Text-Conditioned Restoration Back-End (InstructIR)
Rather than forcing a single network to perform simultaneous illumination expansion and noise suppression, our framework deploys **InstructIR** as a dedicated back-end. 

InstructIR utilizes a NAFNet architecture composed of encoder, middle, and decoder blocks. Each block replaces heavy non-linear activations with Simple Gate (SG) and Simplified Channel Attention (SCA):
$$\text{SG}(X_1, X_2) = X_1 \odot X_2$$
$$\text{SCA}(X) = X \odot \mathcal{W}_p\left(\text{GlobalAvgPool}(X) + \mathbf{W}_e \mathbf{e}\right)$$

where $\mathbf{e} \in \mathbb{R}^{256}$ represents the text condition embedding vector. The prompt:
$$\mathcal{T} = \text{"Deblur and denoise the image."}$$
is mapped into linguistic latent space via MiniLM and projected through a linear head (`LMHead`):
$$\mathbf{e} = \text{LMHead}(\text{MiniLM}(\mathcal{T}))$$

This vector $\mathbf{e}$ modulates channel attention within each NAFNet block, scaling feature maps that emphasize low-pass smoothing and deconvolutional deblurring filters.

### C. Biometric Identity Fine-Tuning Objective
Off-the-shelf InstructIR is pre-trained on general perceptual datasets using $L_1$ pixel reconstruction, which can over-smooth discriminative facial boundaries.

To preserve biometric feature representations, we formulate a compound objective function. Let $\hat{I}$ represent the restored facial patch and $I_{\text{clean}}$ represent the clean ground truth.

#### 1. Pixel Reconstruction Loss
$$\mathcal{L}_{L1} = \frac{1}{HW} \sum_{i=1}^H \sum_{j=1}^W \left| \hat{I}(i, j) - I_{\text{clean}}(i, j) \right|$$

#### 2. Biometric Cosine Identity Loss
Patches are resized to $160 \times 160$ and normalized to $[-1, 1]$ before being passed through a frozen MobileNet-FaceNet feature extractor $\Phi_{\text{FN}}(\cdot)$:
$$\mathbf{f}_{\text{pred}} = \Phi_{\text{FN}}\left(\hat{I}_{160}\right), \quad \mathbf{f}_{\text{gt}} = \Phi_{\text{FN}}\left(I_{\text{clean}, 160}\right)$$
$$\mathcal{L}_{\text{ID}} = 1 - \frac{\mathbf{f}_{\text{pred}} \cdot \mathbf{f}_{\text{gt}}}{\|\mathbf{f}_{\text{pred}}\|_2 \|\mathbf{f}_{\text{gt}}\|_2}$$

#### 3. Total Optimization Objective
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{L1} + \lambda_{\text{ID}} \mathcal{L}_{\text{ID}}$$
We set $\lambda_{\text{ID}} = 1.0$, balancing low-level pixel fidelity with high-level identity alignment.

### D. Face-Centric Spatial Cropping Strategy
Random $256 \times 256$ spatial crops extracted from native $640 \times 480$ images caused the identity loss to plateau at 65.6%, as approximately 40% of crops landed on background walls, clothing, or hair borders where identity gradients are uninformative.

To resolve this, we implemented **Face-Centric Sampling**: crops of size $320 \times 320$ are extracted centered around canonical facial coordinates with bounded spatial jitter ($\Delta x \in [-15, 15], \Delta y \in [-25, 5]$). This ensures training patches consistently capture the facial landmark triad (eyes, nose, mouth), concentrating gradient updates on facial biometrics.

---

## IV. Experimental Setup

### A. Dataset and Degradation Profile
Evaluations were conducted on the official **CASIA-Test** benchmark established in the Low-FaceNet paper [1]. The dataset contains 100 distinct identities, each comprising 5 probe images (500 total evaluation instances) captured at $640 \times 480$ resolution.

The degradation pipeline simulates real-world sub-lux imaging:
1. **Severe Underexposure:** Gamma attenuation scaling luminance to $< 10\%$.
2. **Poisson-Gaussian Noise:** Signal-dependent shot noise combined with Gaussian read noise ($\sigma = 25$).
3. **Motion Blur:** Linear motion convolution simulating camera shake during slow shutter exposures.

### B. Evaluation Protocol
Identification is evaluated in a **closed-set 1:N probe-to-gallery identification paradigm** via `eval.py`:
1. **Gallery Construction:** 100 clean, frontal portraits (`normal/`) are encoded into 128-dimensional biometric vectors using RetinaFace alignment and MobileNet-FaceNet.
2. **Probe Evaluation:** Degraded probe images are passed through the pipeline. RetinaFace detects the face, extracts five landmarks, and crops an aligned $160 \times 160$ region.
3. **Metric:** Top-1 Identification Accuracy. If RetinaFace fails to detect a face in a probe image, it is scored as an identification failure ($0.0$).

### C. Implementation Details
All models were executed on an NVIDIA Tesla T4 GPU (16 GB VRAM) in PyTorch 2.0. InstructIR fine-tuning was performed using AdamW ($\text{lr} = 1.5 \times 10^{-4}$, weight decay $10^{-4}$) with a Cosine Annealing learning rate schedule ($\eta_{\text{min}} = 10^{-5}$) across 15 epochs with batch size 4. Total training time was approximately 3.8 minutes. All test restorations were processed strictly at native $640 \times 480$ resolution to prevent aspect-ratio distortion.

---

## V. Experimental Results & Analysis

### A. Master Benchmark & Pipeline Ablation

TABLE I presents the quantitative comparison across all pipeline permutations, isolating the contribution of each stage.

```
==================================================================================================================
TABLE I: Top-1 Identification Accuracy and Detection Rates on Degraded CASIA-Test (100 Identities, 500 Images)
==================================================================================================================
Method / Configuration        Illumination      Restoration     Biometric Loss   Detection Rate    Top-1 (FaceNet)
------------------------------------------------------------------------------------------------------------------
Raw Degraded Input            None              None            None                  16.8%              3.2%
InstructIR Alone              None              InstructIR      None                  56.4%             27.6%
Low-FaceNet Alone (Base [1])  Low-FaceNet       None            None                  68.0%             12.6%
Two-Stage Zero-Shot           Low-FaceNet       InstructIR      None                 100.0%             62.6%
Two-Stage Fine-Tuned (Ours)   Low-FaceNet       InstructIR      L1 + L_ID            100.0%             70.2%
------------------------------------------------------------------------------------------------------------------
With Inference-Time TTA       Low-FaceNet       InstructIR      L1 + L_Sobel + L_ID  100.0%             76.8%
==================================================================================================================
```

### B. Detailed Analysis of Failure Modes and Gains

#### 1. Why Raw Input and InstructIR Alone Fail
Raw degraded images yield an accuracy of only **3.2%**, with an **83.2% detection failure rate**. The signal is deeply buried in darkness, preventing RetinaFace from extracting edge gradients. When InstructIR is applied alone without Low-FaceNet, accuracy reaches only **27.6%**. InstructIR removes noise effectively, but because it was trained on standard-exposure datasets, it cannot perform extreme dynamic range expansion. The faces remain dark silhouettes. This confirms that **Low-FaceNet is indispensable as an illumination front-end**.

#### 2. The Base Paper Breakdown (Low-FaceNet Alone)
Low-FaceNet elevates luminance, increasing the detection rate from 16.8% to 68.0%. However, its top-1 identification accuracy remains crippled at **12.6%**. Because Retinex division scales noise variance by $1/I^2$, the face is engulfed in high-frequency static. RetinaFace experiences frequent detection dropouts (32% failure rate), and for detected faces, landmark coordinates jitter significantly, distorting FaceNet feature projections.

#### 3. Zero-Shot Restoration (+50.0% Jump)
Passing Low-FaceNet outputs through pretrained InstructIR completely resolves the detection bottleneck: **100% of faces are successfully detected (0% dropout)**. Top-1 identification surges from **12.6% to 62.6%**. This massive gain demonstrates that eliminating sensor noise allows standard face recognition backbones to recover their baseline operational capacity.

#### 4. Biometric Fine-Tuning (+7.6% Jump)
While zero-shot InstructIR restores image cleanliness, it optimizes purely for visual smoothness. Incorporating our Face-Centric Biometric Identity Loss yields a further **+7.6% gain**, reaching **70.2%**. The identity loss penalizes the smoothing of critical facial landmarks, training the restoration network to reconstruct sharp ocular and nasal contours.

---

### C. Cross-Recognizer Generalization (Addressing Recognizer Overlap)

To verify that fine-tuning with FaceNet does not simply overfit the restoration network to that specific embedding space, we evaluated the exact same restored images using an unseen, architecturally distinct model—**ArcFace** (ResNet-50 backbone, Additive Angular Margin loss).

```
=========================================================================================
TABLE II: Cross-Recognizer Generalization on CASIA-Test (FaceNet vs. ArcFace)
=========================================================================================
Pipeline Stage                     Evaluated on FaceNet      Evaluated on ArcFace (Unseen)
-----------------------------------------------------------------------------------------
Low-FaceNet Alone (Base Paper)            12.6%                         11.2%
Two-Stage Zero-Shot                       62.6%                         58.4%
Two-Stage Fine-Tuned (Ours)               70.2%                         65.2%
-----------------------------------------------------------------------------------------
Net Improvement (Ours vs. Zero-Shot)     +7.6%                         +6.8%
=========================================================================================
```

As demonstrated in TABLE II:
- ArcFace accuracy increases from **11.2%** (baseline) to **58.4%** (zero-shot) and achieves **65.2%** under our fine-tuned model.
- The net improvement on the unseen ArcFace model (**+6.8%**) shows strong transfer relative to the in-domain FaceNet model (**+7.6%**), exhibiting an expected minor transfer penalty ($0.8\%$).
This confirms that our Biometric Identity Loss encourages the recovery of **generalizable facial structures** rather than overfitting to FaceNet representations.

---

### D. Detection vs. Identification Decomposition

To clarify whether accuracy gains stem from improved face detection or improved feature matching, TABLE III decouples detection performance from identification on detected faces.

```
=========================================================================================
TABLE III: Decoupled Detection Rate and Conditional Identification Accuracy
=========================================================================================
Method                    RetinaFace Detection Rate    Accuracy on Detected Faces Only*   End-to-End Accuracy
-----------------------------------------------------------------------------------------
Raw Degraded                       16.8%                           19.0%                         3.2%
InstructIR Alone                   56.4%                           48.9%                        27.6%
Low-FaceNet Alone (Base)           68.0%                           18.5%                        12.6%
Two-Stage Zero-Shot               100.0%                           62.6%                        62.6%
Two-Stage Fine-Tuned (Ours)       100.0%                           70.2%                        70.2%
=========================================================================================
*Note: Conditional accuracy for baselines exhibits survival bias, as it evaluates only the easier,
less-degraded subset of images that successfully passed face detection.
```

Under Low-FaceNet alone, even when RetinaFace successfully detects a face (68% of cases), accuracy on those detected faces is only **18.5%**. This demonstrates that noise amplification damages both stages: it causes detection failures and severely corrupts identity embeddings in detected faces. Our two-stage pipeline resolves both failure modes simultaneously.

---

### E. Ablation Studies

#### 1. Cropping Strategy Ablation
We compared standard random cropping against our face-centric sampling strategy across identical 10-epoch fine-tuning schedules:
- **Random $256 \times 256$ Cropping:** Accuracy plateaued at **65.6%**. Approximately 40% of crops contained background walls, shoulders, or partial foreheads without facial landmarks.
- **Face-Centric $320 \times 320$ Cropping:** Accuracy reached **70.2%**, as training patches consistently captured the canonical facial region.

#### 2. Loss Function Formulation
- **$L_1$ Only (Retrained on CASIA pairs):** 63.8% (+1.2% over zero-shot). Pixel reconstruction alone produces marginal improvements.
- **$L_{\text{ID}}$ Only (No pixel reconstruction):** 58.2% (Degrades overall image structure; network introduces high-frequency visual artifacts to satisfy the recognizer).
- **Compound Objective ($L_1 + 1.0 \cdot L_{\text{ID}}$):** **70.2%** (Optimal balance between visual realism and biometric preservation).

---

## VI. Limitations and Future Work

While our two-stage pipeline achieves strong gains on CASIA-Test, several limitations remain:
1. **Computational Latency:** Chaining Low-FaceNet (0.01s inference) with InstructIR (0.21s inference on T4) increases per-image processing time compared to single-stage models. Future work will investigate model distillation into a unified student network.
2. **Fixed Prompt Conditioning:** Our experiments utilize a fixed instruction prompt (*"Deblur and denoise the image."*). Expanding to dynamic, prompt-guided multi-degradation tuning remains an area for further exploration.
3. **Domain Adaptation Scope:** Fine-tuning was evaluated within CASIA camera sensor profiles. Future work will extend evaluation to in-the-wild low-light benchmarks such as DARK FACE [11].

---

## VII. Conclusion

This paper analyzes a critical failure mode in the state-of-the-art low-light face enhancement architecture **Low-FaceNet** (*IEEE TIM 2024*): its Retinex division scales noise variance by $1/I^2$, causing closed-set identification accuracy to collapse to **12.6%** under compound degradation. 

We addressed this vulnerability through a decoupled two-stage architecture integrating Low-FaceNet for dynamic range expansion with **InstructIR** for text-conditioned restoration. Zero-shot integration recovered accuracy to **62.6%** with zero detection dropouts. Fine-tuning InstructIR with a compound Face-Centric Biometric Identity Loss pushed top-1 accuracy to **70.2%**, with cross-recognizer validation on ArcFace reaching **65.2%**. Our findings confirm that in physical low-light biometric enhancement, illumination restoration must be coupled with biometric-preserving denoising to maintain identity fidelity.

---

## References

[1] Z. Fan, Y. Wu, X. Gao, and Y. Song, "Low-FaceNet: Face Recognition-Driven Low-Light Image Enhancement," *IEEE Transactions on Instrumentation and Measurement*, vol. 73, pp. 1-13, 2024.  
[2] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 5203-5212.  
[3] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 815-823.  
[4] M. V. Conde, G. Geigle, and R. Timofte, "High-Quality Image Restoration Following Human Instructions," in *European Conference on Computer Vision (ECCV)*, 2024.  
[5] L. Chen, X. Chu, X. Zhang, and J. Sun, "Simple Baselines for Image Restoration," in *European Conference on Computer Vision (ECCV)*, 2022, pp. 17-33.  
[6] C. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," in *British Machine Vision Conference (BMVC)*, 2018.  
[7] Y. Zhang, J. Zhang, and X. Guo, "Kindling the Darkness: A Practical Low-Light Image Enhancer," in *ACM International Conference on Multimedia (MM)*, 2019, pp. 317-325.  
[8] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, and R. Cong, "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 1780-1789.  
[9] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 5776-5788.  
[10] S. Sengupta, A. Kanazawa, C. D. Castillo, and D. W. Jacobs, "SfSNet: Learning Shape, Reflectance and Illuminance under Dual-Graph Guidance," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.  
[11] Y. Yang, G. Meng, et al., "Advancing the Boundaries of Low-Light Face Detection: Benchmark and Baseline," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2020.
