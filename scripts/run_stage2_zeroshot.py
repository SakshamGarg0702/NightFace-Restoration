import os
import sys
import yaml
import glob
import shutil
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath("InstructIR"))

from models.instructir import create_model
from text.models import LanguageModel, LMHead


def collect_degraded_images(root):
    pairs = []
    valid_ext = ('.jpg', '.png', '.jpeg', '.bmp')
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if os.path.isdir(pdir):
            for f in os.listdir(pdir):
                if f.lower().endswith(valid_ext):
                    pairs.append((pid, f))
    return pairs


def load_instructir(device):
    cfg_candidates = [
        "InstructIR/configs/eval7d.yml",
        "InstructIR/configs/eval5d.yml"
    ] + glob.glob("InstructIR/configs/*.yml")
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    model = create_model(
        input_channels=mc.get("in_ch", 3),
        width=mc.get("width", 32),
        enc_blks=mc.get("enc_blks", [2, 2, 4, 8]),
        middle_blk_num=mc.get("middle_blk_num", 12),
        dec_blks=mc.get("dec_blks", [2, 2, 2, 2]),
        txtdim=mc.get("textdim", 768),
    ).to(device)

    wt = hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt")
    ckpt = torch.load(wt, map_location=device)
    model.load_state_dict(ckpt.get("params", ckpt), strict=False)
    model.eval()

    return model, cfg


def build_prompt_embedding(cfg, device):
    lc = cfg["llm"]

    lm = LanguageModel(model=lc["model"]).to("cpu")
    head = LMHead(
        embedding_dim=lc["model_dim"],
        hidden_dim=lc["embd_dim"],
        num_classes=lc["nclasses"],
    ).to("cpu")

    tw = hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt")
    head.load_state_dict(torch.load(tw, map_location="cpu"), strict=False)
    lm.eval()
    head.eval()

    instruction = "Deblur and denoise the image."
    print(f"Conditioning text prompt: '{instruction}'")
    with torch.no_grad():
        raw = lm(instruction)
        emb, _ = head(raw)
    return emb.to(device).detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_instructir(device)
    prompt_emb = build_prompt_embedding(cfg, device)

    in_root = "dataset/CASIA-Test/SOTAs/LowFaceNet_Baseline"
    out_root = "dataset/CASIA-Test/SOTAs/LowFaceNet_InstructIR_ZeroShot"
    os.makedirs(out_root, exist_ok=True)

    images = collect_degraded_images(in_root)
    print(f"Processing {len(images)} images (zero-shot)...")

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for pid, fname in tqdm(images, desc="InstructIR restoration"):
            src = os.path.join(in_root, pid, fname)
            dst_dir = os.path.join(out_root, pid)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)

            img = Image.open(src).convert("RGB")
            inp = to_tensor(img).unsqueeze(0).to(device)

            out = model(inp, prompt_emb)
            out = torch.clamp(out, 0.0, 1.0)
            T.ToPILImage()(out.squeeze(0).cpu()).save(dst)

    print(f"Saved zero-shot restored images to: {out_root}")

    shutil.rmtree("encoding", ignore_errors=True)
    os.makedirs("encoding", exist_ok=True)

    print("\nRunning evaluation...")
    os.system("python eval.py --dataset_name=CASIA-Test --method_name=LowFaceNet_InstructIR_ZeroShot")


if __name__ == "__main__":
    main()
