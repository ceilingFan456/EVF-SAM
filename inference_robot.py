import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
from torch.profiler import profile, ProfilerActivity


from pathlib import Path
from tqdm import tqdm    



def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--version", required=True)
    parser.add_argument("--vis_save_path", default="/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105_evf-sam", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    parser.add_argument("--image_path", type=str, default="/home/t-qimhuang/disk/robot_dataset/final_test_set/run5_test105")
    parser.add_argument("--prompt", type=str, default="zebra top left")
    
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    '''
    preprocess of Segment Anything Model, including scaling, normalization and padding.  
    preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
    input: ndarray
    output: torch.Tensor
    '''
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    # ---- simple total param count ----
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")

    return tokenizer, model

def main(args):
    args = parse_args(args)
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- IO clarified (BATCH MODE, same structure as earlier repo) ---
    # [MODIFIED] we always run batch over IN_ROOT/image/**/*.jpg
    IN_ROOT = Path(args.image_path)                                   # [ADDED]
    if not IN_ROOT.exists():                                          # [ADDED]
        print(f"Path not found: {IN_ROOT}")                           # [ADDED]
        exit()                                                        # [ADDED]

    os.makedirs(args.vis_save_path, exist_ok=True)                    # (kept)
    OUT_BASE = Path(args.vis_save_path)                               # [ADDED]
    PROMPT_TO_CODE = {"robot": "000", "gripper": "001", "robot arm": "002"}  # [ADDED]

    def _ensure_dir(p: Path):                                         # [ADDED]
        os.makedirs(p, exist_ok=True)                                 # [ADDED]

    # initialize model and tokenizer (unchanged)
    tokenizer, model = init_models(args)

    # [ADDED] gather all images under IN_ROOT/image/**/*.jpg
    all_imgs = sorted((IN_ROOT / "image").rglob("*.jpg"))             # [ADDED]
    if not all_imgs:                                                  # [ADDED]
        print(f"[WARN] No .jpg under {IN_ROOT/'image'}")              # [ADDED]
        exit()                                                        # [ADDED]

    # [ADDED] iterate with tqdm
    for img_path in tqdm(all_imgs, desc="Processing images", unit="img"):  # [ADDED]
        # [ADDED] derive case name as 1st folder after image/
        try:                                                          # [ADDED]
            parts = img_path.resolve().relative_to(IN_ROOT.resolve()).parts  # [ADDED]
        except Exception:                                             # [ADDED]
            print(f"[WARN] Skip (not under IN_ROOT): {img_path}")     # [ADDED]
            continue                                                  # [ADDED]
        if len(parts) < 2 or parts[0] != "image":                     # [ADDED]
            print(f"[WARN] Skip (unexpected path): {img_path}")       # [ADDED]
            continue                                                  # [ADDED]

        case_name = parts[1]                                          # [ADDED]
        frame_name = img_path.name                                    # [ADDED]

        # --- preprocess (unchanged) ---
        image_np = cv2.imread(str(img_path))                          # [MODIFIED] read per image path
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)          # (kept)
        original_size_list = [image_np.shape[:2]]                     # (kept)

        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)  # (kept)
        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)                        # (kept)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)                                     # (kept)

        # [ADDED] save ORIGINAL once per frame (no CODE)
        img_dir = OUT_BASE / "image" / case_name                       # [ADDED]
        _ensure_dir(img_dir)                                           # [ADDED]
        cv2.imwrite(str(img_dir / frame_name), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  # [ADDED]

        # --- evaluation loop per prompt (EVAL CALL UNCHANGED) ---
        for user_text, code in PROMPT_TO_CODE.items():                 # [ADDED]
            input_ids = tokenizer(user_text, return_tensors="pt")["input_ids"].to(device=model.device)  # (kept API)

            # infer (UNCHANGED)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
              record_shapes=True,
              with_flops=True) as prof:
                pred_mask = model.inference(
                    image_sam.unsqueeze(0),
                    image_beit.unsqueeze(0),
                    input_ids,
                    resize_list=[resize_shape],
                    original_size_list=original_size_list,
                    multimask_output=True,                                    # (kept)
                )                                                          # (unchanged call)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))   
            print(prof.key_averages().table(sort_by="flops"))
            total_flops = sum([item.flops for item in prof.key_averages()])
            print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")

            pred_mask = pred_mask.detach().cpu().numpy()[0]           # (kept)
            pred_mask = pred_mask > 0                                  # (kept)

            # --- save outputs in original structure ---
            # 1) binary mask -> OUT_BASE/mask/<CASE>/<CODE>/<FRAME>.jpg
            mask_dir = OUT_BASE / "mask" / case_name / code            # [ADDED]
            _ensure_dir(mask_dir)                                      # [ADDED]
            cv2.imwrite(str(mask_dir / frame_name), (pred_mask.astype(np.uint8) * 255))  # [ADDED]

            # 2) masked overlay (transparent red) -> OUT_BASE/masked/<CASE>/<CODE>/<FRAME>.jpg
            alpha = 0.5                                                # [ADDED]
            image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)   # [ADDED]
            overlay = image_np_bgr.copy()                              # [ADDED]
            overlay[pred_mask] = (0, 0, 255)                           # [ADDED] red in BGR
            vis_bgr = cv2.addWeighted(overlay, alpha, image_np_bgr, 1 - alpha, 0)  # [ADDED]

            masked_dir = OUT_BASE / "masked" / case_name / code        # [ADDED]
            _ensure_dir(masked_dir)                                    # [ADDED]
            cv2.imwrite(str(masked_dir / frame_name), vis_bgr)         # [ADDED]


if __name__ == "__main__":
    main(sys.argv[1:])