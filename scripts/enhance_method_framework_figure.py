from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "figure_method.png"
OUT_DIR = ROOT / "outputs" / "figures" / "paper_summary"
PNG_OUT = OUT_DIR / "method_framework_hq.png"
TIFF_OUT = OUT_DIR / "method_framework_hq.tiff"
PDF_OUT = OUT_DIR / "method_framework_hq.pdf"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.open(SOURCE).convert("RGB")
    # Preserve the original geometry and content. The operations below only
    # improve print readability: 4x Lanczos resampling, mild denoising, and
    # conservative sharpening without recoloring or relayout.
    scale = 4
    enlarged = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    enlarged = enlarged.filter(ImageFilter.MedianFilter(size=3))
    enlarged = ImageEnhance.Contrast(enlarged).enhance(1.04)
    enlarged = ImageEnhance.Sharpness(enlarged).enhance(1.65)
    enlarged = enlarged.filter(ImageFilter.UnsharpMask(radius=1.35, percent=115, threshold=2))

    # Keep the background white and avoid alpha-related transparency artifacts
    # when journals convert PDF/TIFF assets.
    enlarged = ImageOps.expand(enlarged, border=0, fill="white")

    dpi = (600, 600)
    enlarged.save(PNG_OUT, dpi=dpi, optimize=True)
    enlarged.save(TIFF_OUT, dpi=dpi, compression="tiff_lzw")
    enlarged.save(PDF_OUT, "PDF", resolution=600.0)

    print(f"source={SOURCE}")
    print(f"source_size={image.width}x{image.height}")
    print(f"output_size={enlarged.width}x{enlarged.height}")
    for path in (PNG_OUT, TIFF_OUT, PDF_OUT):
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
