# Model Checkpoints

This directory holds the checkpoints required for evaluation and inference across all three pipeline stages.

Because large model weights exceed standard git repository recommendations, model weights are hosted externally and can be obtained via:

1. **Our Biometrically Fine-Tuned Checkpoint** (`instructir_biometric_finetuned.pth`, ~60.7 MB):
   - Checkpoint of the second-stage InstructIR model trained with compound $L_1$ reconstruction and FaceNet identity cosine loss on face-centric crops.
   - Achieves **70.2% closed-set identification accuracy** on degraded CASIA-Test.
   - 🔗 **Direct Download:** [GitHub Releases v1.0 — `instructir_biometric_finetuned.pth`](https://github.com/sakshamgarg0702/NightFace-Restoration/releases/download/v1.0/instructir_biometric_finetuned.pth)  
     *(Place the downloaded file directly inside this `checkpoint/` directory).*

2. **Low-FaceNet Baseline Weights** (`Low-FaceNet.pth`):
   - Front-end illumination recovery network weights from the base paper.
   - 🔗 **Hosted on Kaggle:** [sakshamgarg0702/lowfacenet-bundle](https://www.kaggle.com/datasets/sakshamgarg0702/lowfacenet-bundle).

3. **Biometric Face Recognition & Detection**:
   - `facenet_mobilenet.pth` (FaceNet 128-D embedding backbone)
   - `Retinaface_mobilenet0.25.pth` (RetinaFace detector)
   - 🔗 **Hosted on Kaggle:** Available in the [Kaggle bundle](https://www.kaggle.com/datasets/sakshamgarg0702/lowfacenet-bundle) under `checkpoint/`. Place in this folder or in `face_module/model_data/`.

4. **Pretrained InstructIR (Zero-Shot)**:
   - `im_instructir-7d.pt` and `lm_instructir-7d.pt` are downloaded automatically on first run via `huggingface_hub` from [marcosv/InstructIR](https://huggingface.co/marcosv/InstructIR).
