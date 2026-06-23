"""
Seatbelt Classifier — Local Fine-Tuning Script
===============================================
Fine-tunes RISEF/yolov11s-seatbelt on the overhead windshield crop dataset.

Hardware target : RTX 3050 Ti 4 GB (VRAM)
Batch size      : 16  (fits in 4 GB with AMP enabled)
Training time   : ~20-40 min for 50 epochs on local GPU

Dataset expected at:
    data/seatbelt datasets/final_dataset/
        train/
            seatbelt/       (658 images)
            no_seatbelt/    (660 images)
        val/
            seatbelt/       (166 images)
            no_seatbelt/    (166 images)

Output weights saved to:
    models/weights/seatbelt_finetuned.pt

After training, update src/configs/violations.yaml:
    hf_model_cache: "models/weights/seatbelt_finetuned.pt"
    auto_approve_confidence: 0.90   # restore from 0.97
"""

import sys
import os
from pathlib import Path

# ── Resolve project root (works regardless of where you run this from) ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Step 1: Check CUDA / GPU availability ─────────────────────────────────
import torch

print("=" * 55)
print("  Seatbelt Classifier — Local Fine-Tuning")
print("=" * 55)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅  GPU : {gpu_name}")
    print(f"   VRAM: {vram_gb:.1f} GB")
    DEVICE = "0"   # use GPU 0
else:
    print("⚠️  No CUDA GPU found — training on CPU (will be very slow).")
    print("   Make sure PyTorch with CUDA is installed:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    DEVICE = "cpu"

# ── Step 2: Validate dataset paths ────────────────────────────────────────
DATASET_DIR = ROOT / "data" / "seatbelt datasets" / "final_dataset"

if not DATASET_DIR.exists():
    print(f"\n❌ Dataset not found at: {DATASET_DIR}")
    print("   Make sure the folder structure is:")
    print("   data/seatbelt datasets/final_dataset/train/seatbelt/")
    print("   data/seatbelt datasets/final_dataset/train/no_seatbelt/")
    sys.exit(1)

print(f"\n📂 Dataset : {DATASET_DIR}")

# Count images per split
for split in ["train", "val"]:
    for cls in ["seatbelt", "no_seatbelt"]:
        p = DATASET_DIR / split / cls
        n = len(list(p.glob("*"))) if p.exists() else 0
        print(f"   {split}/{cls}: {n} images")

# ── Step 3: Download base model weights from HuggingFace ─────────────────
print("\n📥 Downloading base model from HuggingFace...")
try:
    from huggingface_hub import hf_hub_download
    base_weights = hf_hub_download(
        repo_id="RISEF/yolov11s-seatbelt",
        filename="weights/best.pt",
        local_dir=str(ROOT / "models" / "weights" / "hf_cache"),
    )
    print(f"   Base weights: {base_weights}")
except Exception as e:
    # Fallback: use existing cached weights if download fails
    fallback = ROOT / "models" / "weights" / "seatbelt_yolov11s.pt"
    if fallback.exists():
        base_weights = str(fallback)
        print(f"   ⚠️  HF download failed, using cached: {base_weights}")
    else:
        print(f"❌ Cannot find base weights: {e}")
        sys.exit(1)

# ── Step 4: Fine-tune ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from ultralytics import YOLO

    print("\n🚀 Starting fine-tuning...\n")

    model = YOLO(base_weights)

    OUT_DIR = ROOT / "models" / "weights" / "finetune_runs"

    results = model.train(
        task      = "classify",
        data      = str(DATASET_DIR),   # points to folder containing train/ and val/
        epochs    = 50,                 # increase to 80 if accuracy is still climbing
        imgsz     = 224,                # match windshield crop size used in production
        batch     = 16,                 # safe for 4 GB VRAM with AMP; reduce to 8 if OOM
        device    = DEVICE,
        workers   = 4,                  # data loader threads (reduce to 2 if CPU is bottleneck)

        # ── Optimiser ──────────────────────────────────────────────
        optimizer = "AdamW",
        lr0       = 0.0005,             # low LR for fine-tuning (preserve pretrained weights)
        lrf       = 0.01,               # cosine decay: final LR = lr0 * lrf
        weight_decay = 0.0005,
        warmup_epochs = 3,
        patience  = 15,                 # early stopping if val top1 doesn't improve for 15 epochs

        # ── Augmentation ───────────────────────────────────────────
        augment   = True,
        hsv_h     = 0.015,              # colour jitter (handles different lighting conditions)
        hsv_s     = 0.4,
        hsv_v     = 0.4,
        fliplr    = 0.5,                # horizontal flip (ok for windshields)
        flipud    = 0.0,                # NO vertical flip (upside-down cars don't exist)
        degrees   = 10,                 # rotation ±10° (camera tilt variation)
        translate = 0.1,
        scale     = 0.3,
        perspective = 0.0005,           # perspective warp (simulates overhead camera angle)
        erasing   = 0.2,                # random erasing (handles partial occlusion)

        # ── Output ─────────────────────────────────────────────────
        project   = str(OUT_DIR),
        name      = "seatbelt_v1",
        exist_ok  = True,
        save      = True,
        plots     = True,               # saves confusion matrix + training curves as images
        verbose   = True,
    )

    # ── Step 5: Copy best weights to the production path ─────────────────────
    import shutil

    best_src  = OUT_DIR / "seatbelt_v1" / "weights" / "best.pt"
    best_dest = ROOT / "models" / "weights" / "seatbelt_finetuned.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dest)
        print(f"\n✅ Fine-tuning complete!")
        print(f"   Best model copied to: {best_dest}")
        print(f"\n   Validation metrics:")
        print(f"   Top-1 accuracy : {results.results_dict.get('metrics/accuracy_top1', 'N/A'):.4f}")
        print(f"   Top-5 accuracy : {results.results_dict.get('metrics/accuracy_top5', 'N/A'):.4f}")
        print("""
       ── Next Steps ──────────────────────────────────────────────
       1. Update src/configs/violations.yaml:

            seatbelt:
              hf_model_cache: "models/weights/seatbelt_finetuned.pt"
              auto_approve_confidence: 0.90   # restore from 0.97

       2. Run the test suite to verify nothing broke:
            pytest tests/ -q

       3. Re-deploy the Streamlit app and test on your traffic footage.
       ─────────────────────────────────────────────────────────────
    """)
    else:
        print(f"⚠️  Best weights not found at {best_src}. Check training logs above.")
