from __future__ import annotations

import argparse
from pathlib import Path


def try_generate_sdxl_turbo(prompt: str, negative_prompt: str, seed: int | None, steps: int, width: int, height: int, out_path: Path) -> bool:
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionXLPipeline  # type: ignore
    except Exception as e:
        print(f"Diffusers/Torch not available: {e}")
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(device)
        pipe.enable_attention_slicing()
        gen = torch.Generator(device=device)
        if seed is not None:
            gen = gen.manual_seed(int(seed))
        imgs = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=max(1, min(steps, 6)),
            guidance_scale=0.0,
            generator=gen,
        ).images
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(out_path)
        return True
    except Exception as e:
        print(f"SDXL Turbo generation failed: {e}")
        return False


def generate_fallback(path: Path, size: int = 1024) -> Path:
    from .gen_background import generate_bb10_ppm
    ppm_path = Path("data/backgrounds/bb10.ppm")
    generate_bb10_ppm(ppm_path, size=size)
    print(f"Wrote {ppm_path}")
    return ppm_path


def show_image(path: Path) -> None:
    try:
        import tkinter as tk
    except Exception as e:
        print(f"Tkinter not available to show image: {e}")
        return
    root = tk.Tk()
    root.title(f"Preview - {path.name}")
    try:
        img = tk.PhotoImage(file=str(path))
    except Exception as e:
        tk.Label(root, text=f"Failed to load image: {e}").pack()
        root.mainloop()
        return
    w, h = img.width(), img.height()
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    canvas.create_image(w//2, h//2, image=img)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate and preview a BB-10 background (SDXL Turbo or procedural fallback)")
    p.add_argument("--out", default=str(Path("data/backgrounds/bb10.png")))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--prompt", default=(
        "Top-down Beyblade BB-10 Attack Type stadium, three knockout pockets at 0 deg, 120 deg, 240 deg, "
        "smooth concave bowl, visible tornado ridge, orthographic view, high-contrast pocket cutouts, "
        "matte plastic, subtle scuffs, studio lighting, dark background"
    ))
    p.add_argument("--negative-prompt", dest="neg", default=(
        "perspective, camera angle, tilt, distortion, text, watermark, logo, people, hands, toys, lowres, blurry"
    ))
    p.add_argument("--show", action="store_true", help="Open a preview window after generation")

    args = p.parse_args(argv)
    try:
        w_str, h_str = args.size.lower().split("x")
        width = int(w_str); height = int(h_str)
    except Exception:
        width = height = 1024

    out_path = Path(args.out)
    wrote = out_path
    if try_generate_sdxl_turbo(args.prompt, args.neg, args.seed, args.steps, width, height, out_path):
        print(f"Wrote {out_path}")
    else:
        wrote = generate_fallback(out_path, size=max(width, height))
    if args.show:
        show_image(wrote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

