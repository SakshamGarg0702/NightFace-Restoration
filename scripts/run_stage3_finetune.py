import os
import sys
import yaml
import glob
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.abspath("InstructIR"))

from models.instructir import create_model
from text.models import LanguageModel, LMHead
from face_module.nets.facenet import Facenet


class DegradedPairFaceCentricDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, crop_size=320):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.crop_size = crop_size
        self.to_tensor = T.ToTensor()

        valid_ext = ('.jpg', '.png', '.jpeg', '.bmp')
        
        self.pairs = []
        if not (os.path.isdir(degraded_dir) and os.path.isdir(clean_dir)):
            raise FileNotFoundError(
                f"Missing training directory: {degraded_dir} or {clean_dir}"
            )

        for item in sorted(os.listdir(degraded_dir)):
            d_item = os.path.join(degraded_dir, item)
            c_item = os.path.join(clean_dir, item)
            if os.path.isdir(d_item) and os.path.isdir(c_item):
                for fn in os.listdir(d_item):
                    if fn.lower().endswith(valid_ext) and os.path.exists(os.path.join(c_item, fn)):
                        self.pairs.append((os.path.join(d_item, fn), os.path.join(c_item, fn)))
            elif os.path.isfile(d_item) and item.lower().endswith(valid_ext):
                if os.path.exists(c_item):
                    self.pairs.append((d_item, c_item))

        print(f"Loaded {len(self.pairs)} training pairs from {degraded_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, cln_path = self.pairs[idx]
        deg_img = Image.open(deg_path).convert("RGB")
        cln_img = Image.open(cln_path).convert("RGB")

        # Face-centric crop with jitter
        w, h = deg_img.size
        cx = w // 2 + torch.randint(-15, 16, (1,)).item()
        cy = h // 2 - 10 + torch.randint(-15, 16, (1,)).item()

        left = max(0, min(w - self.crop_size, cx - self.crop_size // 2))
        top = max(0, min(h - self.crop_size, cy - self.crop_size // 2))

        deg_crop = deg_img.crop((left, top, left + self.crop_size, top + self.crop_size))
        cln_crop = cln_img.crop((left, top, left + self.crop_size, top + self.crop_size))

        return self.to_tensor(deg_crop), self.to_tensor(cln_crop)


def setup_instructir(device):
    cfg_candidates = [
        "InstructIR/configs/eval7d.yml",
        "InstructIR/configs/eval5d.yml"
    ] + glob.glob("InstructIR/configs/*.yml")
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    mc = cfg['model']
    model = create_model(
        input_channels=mc.get('in_ch', 3),
        width=mc.get('width', 32),
        enc_blks=mc.get('enc_blks', [2, 2, 4, 8]),
        middle_blk_num=mc.get('middle_blk_num', 12),
        dec_blks=mc.get('dec_blks', [2, 2, 2, 2]),
        txtdim=mc.get('textdim', 768)
    ).to(device)

    wt = hf_hub_download(repo_id="marcosv/InstructIR", filename="im_instructir-7d.pt")
    ckpt = torch.load(wt, map_location=device)
    model.load_state_dict(ckpt.get("params", ckpt), strict=False)
    return model, cfg


def get_prompt_embedding(cfg, device):
    lc = cfg['llm']
    lm = LanguageModel(model=lc['model']).to('cpu')
    head = LMHead(
        embedding_dim=lc['model_dim'],
        hidden_dim=lc['embd_dim'],
        num_classes=lc['nclasses']
    ).to('cpu')

    tw = hf_hub_download(repo_id="marcosv/InstructIR", filename="lm_instructir-7d.pt")
    head.load_state_dict(torch.load(tw, map_location='cpu'), strict=False)
    lm.eval()
    head.eval()

    instruction = "Deblur and denoise the image."
    print(f"Conditioning text prompt: '{instruction}'")
    with torch.no_grad():
        raw = lm(instruction)
        emb, _ = head(raw)
    return emb.to(device).detach()


def main():
    parser = argparse.ArgumentParser(description="Biometric fine-tuning of InstructIR")
    parser.add_argument("--train_degraded_dir", type=str, default="dataset/LaPa-Face/degraded")
    parser.add_argument("--train_clean_dir", type=str, default="dataset/LaPa-Face/normal")
    parser.add_argument("--test_degraded_input", type=str, default="dataset/CASIA-Test/SOTAs/LowFaceNet_Baseline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--id_weight", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    instructir, cfg = setup_instructir(device)
    prompt_emb = get_prompt_embedding(cfg, device)

    facenet = Facenet(backbone="mobilenet", mode="predict").to(device)
    fn_wt = "face_module/model_data/facenet_mobilenet.pth"
    if not os.path.exists(fn_wt):
        fn_wt = "checkpoint/facenet_mobilenet.pth"
    facenet.load_state_dict(torch.load(fn_wt, map_location=device), strict=False)
    facenet.eval()
    for p in facenet.parameters():
        p.requires_grad = False

    dataset = DegradedPairFaceCentricDataset(args.train_degraded_dir, args.train_clean_dir, crop_size=320)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    l1_fn = nn.L1Loss()
    optimizer = optim.AdamW(instructir.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\nStarting fine-tuning ({len(dataset)} pairs, {args.epochs} epochs)...")
    instructir.train()
    for ep in range(1, args.epochs + 1):
        ep_loss, ep_l1, ep_id = 0.0, 0.0, 0.0
        for deg, cln in loader:
            deg, cln = deg.to(device), cln.to(device)
            optimizer.zero_grad()

            restored = torch.clamp(instructir(deg, prompt_emb), 0.0, 1.0)
            pixel_loss = l1_fn(restored, cln)

            # FaceNet requires 160x160 input
            r160 = nn.functional.interpolate(restored, (160, 160), mode="bilinear", align_corners=False)
            c160 = nn.functional.interpolate(cln, (160, 160), mode="bilinear", align_corners=False)
            id_loss = 1.0 - nn.functional.cosine_similarity(facenet(r160), facenet(c160)).mean()

            loss = pixel_loss + args.id_weight * id_loss
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_l1 += pixel_loss.item()
            ep_id += id_loss.item()

        scheduler.step()
        n = len(loader)
        print(f"Epoch [{ep:02d}/{args.epochs}] Loss: {ep_loss/n:.4f} (L1: {ep_l1/n:.4f}, FaceNet-ID: {ep_id/n:.4f})")

    os.makedirs("checkpoint", exist_ok=True)
    torch.save(instructir.state_dict(), "checkpoint/instructir_biometric_finetuned.pth")
    print("\nSaved checkpoint to checkpoint/instructir_biometric_finetuned.pth")

    out_root = "dataset/CASIA-Test/SOTAs/Our_FineTuned_InstructIR"
    os.makedirs(out_root, exist_ok=True)
    instructir.eval()

    ok_ext = ('.jpg', '.png', '.jpeg', '.bmp')
    casia_tasks = []
    for pid in sorted(os.listdir(args.test_degraded_input)):
        pdir = os.path.join(args.test_degraded_input, pid)
        if os.path.isdir(pdir):
            odir = os.path.join(out_root, pid)
            os.makedirs(odir, exist_ok=True)
            for fn in os.listdir(pdir):
                if fn.lower().endswith(ok_ext):
                    casia_tasks.append((os.path.join(pdir, fn), os.path.join(odir, fn)))

    print(f"\nProcessing {len(casia_tasks)} test images...")
    to_tensor = T.ToTensor()
    with torch.no_grad():
        for src, dst in tqdm(casia_tasks, desc="Inference"):
            img = Image.open(src).convert("RGB")
            inp = to_tensor(img).unsqueeze(0).to(device)
            out = torch.clamp(instructir(inp, prompt_emb), 0.0, 1.0)
            T.ToPILImage()(out.squeeze(0).cpu()).save(dst)

    shutil.rmtree("encoding", ignore_errors=True)
    os.makedirs("encoding", exist_ok=True)

    print("\nRunning evaluation...")
    os.system("python eval.py --dataset_name=CASIA-Test --method_name=Our_FineTuned_InstructIR")


if __name__ == "__main__":
    main()
