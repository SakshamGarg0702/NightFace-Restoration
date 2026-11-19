# Pretrained Model Checkpoints

This directory holds the model checkpoints required to run the baseline and fine-tuning pipelines.

Because these weights exceed GitHub file size limits, please download them from our Kaggle bundle or Hugging Face:

1. **Low-FaceNet Checkpoints** (`Low-FaceNet.pth`, `best_casia_weight.pth`):
   - Available in the Kaggle bundle: [sakshamgarg0702/lowfacenet-bundle](https://www.kaggle.com/datasets/sakshamgarg0702/lowfacenet-bundle)

2. **FaceNet and RetinaFace Weights**:
   - `facenet_mobilenet.pth`
   - `Retinaface_mobilenet0.25.pth`
   - Available in the Kaggle bundle under `checkpoint/`. Place them in this directory or in `face_module/model_data/`.

3. **InstructIR Weights**:
   - `im_instructir-7d.pt` and `lm_instructir-7d.pt` are downloaded automatically by the scripts via `huggingface_hub` from `marcosv/InstructIR`.
