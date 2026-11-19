import os
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models import instructir
from text.models import LanguageModel, LMHead
from utils import load_yaml

def clean_folder(input_dir, output_dir, prompt="Deblur and denoise the image."):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running InstructIR on device: {device}")

    # 1. Load config and model architecture
    cfg_path = os.path.join(os.path.dirname(__file__), "configs/eval5d.yml") if not os.path.exists("configs/eval5d.yml") else "configs/eval5d.yml"
    cfg = load_yaml(cfg_path)
    model = instructir(cfg)

    ckpt_path = os.path.join(os.path.dirname(__file__), "models/im_instructir-7d.pt") if not os.path.exists("models/im_instructir-7d.pt") else "models/im_instructir-7d.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model = model.to(device).eval()

    # 2. Load language prompt module
    lm_model = LanguageModel()
    lm_head = LMHead(embedding_dim=cfg.get("model_dim", 32))
    lm_ckpt = os.path.join(os.path.dirname(__file__), "models/lm_instructir-7d.pt") if not os.path.exists("models/lm_instructir-7d.pt") else "models/lm_instructir-7d.pt"
    lm_head.load_state_dict(torch.load(lm_ckpt, map_location='cpu'))
    lm_head = lm_head.to(device).eval()

    # 3. Create text instruction embedding
    text_feat = lm_model(prompt)
    prompt_emb = lm_head(text_feat.to(device))

    # 4. Handle nested identity folders or flat images
    identities = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    with torch.no_grad():
        if identities:
            for identity in tqdm(identities, desc="InstructIR Denoising & Deblurring"):
                in_id_dir = os.path.join(input_dir, identity)
                out_id_dir = os.path.join(output_dir, identity)
                os.makedirs(out_id_dir, exist_ok=True)

                for fname in os.listdir(in_id_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(in_id_dir, fname)
                        img = Image.open(img_path).convert("RGB")
                        x = TF.to_tensor(img).unsqueeze(0).to(device)
                        restored = model(x, prompt_emb)
                        clean_img = TF.to_pil_image(torch.clamp(restored.squeeze(0), 0.0, 1.0).cpu())
                        clean_img.save(os.path.join(out_id_dir, fname))
        else:
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for fname in tqdm(files, desc="InstructIR Processing"):
                img_path = os.path.join(input_dir, fname)
                img = Image.open(img_path).convert("RGB")
                x = TF.to_tensor(img).unsqueeze(0).to(device)
                restored = model(x, prompt_emb)
                clean_img = TF.to_pil_image(torch.clamp(restored.squeeze(0), 0.0, 1.0).cpu())
                clean_img.save(os.path.join(output_dir, fname))

    print(f"[✓] Restored images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="Deblur and denoise the image.")
    args = parser.parse_args()

    clean_folder(args.input_dir, args.output_dir, args.prompt)
