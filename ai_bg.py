from __future__ import annotations

import argparse
from pathlib import Path


def generate_sdxl_turbo(prompt: str, negative_prompt: str, seed: int | None, steps: int, width: int, height: int, out_path: Path) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(int(seed))

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=max(1, min(steps, 6)),  # Turbo works well at 1-4 steps
        guidance_scale=0.0,  # CFG off for Turbo speed/quality tradeoff
        generator=generator,
    ).images

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path)


def _show_image(path: Path) -> None:
    try:
        import tkinter as tk
    except Exception as e:
        print(f"Tkinter not available to show image: {e}")
        return
    root = tk.Tk()
    root.title(f"Preview — {path.name}")
    try:
        img = tk.PhotoImage(file=str(path))
    except Exception as e:
        tk.Label(root, text=f"Failed to load image: {e}").pack()
        root.mainloop()
        return
    w = img.width(); h = img.height()
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.pack()
    canvas.create_image(w//2, h//2, image=img)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a BB-10 background using SDXL Turbo")
    p.add_argument("--out", default=str(Path("data/backgrounds/bb10.png")))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--prompt", default=(
        "Top-down Beyblade BB-10 Attack Type stadium, three knockout pockets at 0°, 120°, 240°, "
        "smooth concave bowl, visible tornado ridge, orthographic view, high-contrast pocket cutouts, "
        "matte plastic, subtle scuffs, studio lighting, dark background"
    ))
    p.add_argument("--negative-prompt", dest="neg", default=(
        "perspective, camera angle, tilt, distortion, text, watermark, logo, people, hands, toys, lowres, blurry"
    ))

    args = p.parse_args(argv)
    p.add_argument("--show", action="store_true", help="Open a preview window after generation")

    try:
        w_str, h_str = args.size.lower().split("x")
        width = int(w_str); height = int(h_str)
    except Exception:
        width = height = 1024

    out_path = Path(args.out)
    wrote_path = out_path
    try:
        generate_sdxl_turbo(args.prompt, args.neg, args.seed, args.steps, width, height, out_path)
        print(f"Wrote {out_path}")
    except Exception as e:
        # Fallback: generate procedural BB-10 PPM so you still get a background
        print(f"SDXL Turbo generation failed ({e}). Falling back to procedural PPM background.")
        from .gen_background import generate_bb10_ppm
        wrote_path = Path("data/backgrounds/bb10.ppm")
        generate_bb10_ppm(wrote_path, size=max(width, height))
        print(f"Wrote {wrote_path}")

    if args.show:
        _show_image(wrote_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
