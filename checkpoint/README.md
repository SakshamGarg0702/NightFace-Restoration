# Model Checkpoints

This directory holds the checkpoints required for evaluation and inference across all three pipeline stages.

Because large model weights exceed GitHub file size recommendations, external weights can be obtained via the following sources:

1. **Low-FaceNet Weights** (`Low-FaceNet.pth`):
   - Front-end illumination recovery network weights from the base paper.
   - Hosted on Kaggle: [sakshamgarg0702/lowfacenet-bundle](https://www.kaggle.com/datasets/sakshamgarg0702/lowfacenet-bundle).

2. **Biometric Face Recognition & Detection**:
   - `facenet_mobilenet.pth` (FaceNet 128-D embedding backbone)
   - `Retinaface_mobilenet0.25.pth` (RetinaFace detector)
   - Hosted in the Kaggle bundle under `checkpoint/`. Place in this folder or in `face_module/model_data/`.

3. **Pretrained InstructIR (Zero-Shot)**:
   - `im_instructir-7d.pt` and `lm_instructir-7d.pt` are downloaded automatically on first run via `huggingface_hub` from `marcosv/InstructIR`.

4. **Our Fine-Tuned InstructIR Checkpoint** (`instructir_biometric_finetuned.pth`, ~60.7 MB):
   - The fine-tuned Stage-2 checkpoint trained with compound $L_1$ reconstruction and FaceNet identity cosine loss on face-centric crops.
   - Achieves 70.2% closed-set identification accuracy on degraded CASIA-Test.
   - Placed in this directory for direct evaluation (`scripts/run_stage3_finetune.py`), or reproducible by retraining via the same script.
