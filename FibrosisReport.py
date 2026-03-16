# FibrosisReport.py

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

from Utils.paths import project_root_from_image, ensure_masks_from_annotations
from Utils.mask_reader import MaskRegionReader
from Utils.tile_selection import calculate_useful_tiles
from Utils.disk_cache import tile_cache_dirs, save_disk_tile_bundle
from Utils.zarr_helpers import count_positive_pixels_level0
from Utils.image_utils import get_mpp_um, save_binary_mask_tiff
from Utils.fibrosis_methods import fibrosis_red_green, fibrosis_stain_deconv


def process_one_image(
    image_path: str,
    method: str,
    report_mode: str,
    save_mask: bool,
    post_processing: bool,
    storage_mode: str,
    tile_size: int = 1024,
    overlap: int = 256,
    cortex_tile_threshold_pct: float = 25.0,
):
    img = Path(image_path)
    root = project_root_from_image(img)
    stem = img.stem

    fibrosis_masks_dir = root / "Fibrosis Masks" / method
    fibrosis_reports_dir = root / "Fibrosis Reports" / method
    fibrosis_masks_dir.mkdir(parents=True, exist_ok=True)
    fibrosis_reports_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = ensure_masks_from_annotations(str(img))
    cortex_mask_path = mask_paths["Cortex"]
    glom_mask_path = mask_paths["Glomeruli"]
    tub_mask_path = mask_paths["Tubules"]

    slide = open_slide(str(img))
    dz_img = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=False)
    level = dz_img.level_count - 1

    cortex_reader = MaskRegionReader(cortex_mask_path)
    glom_reader = MaskRegionReader(glom_mask_path)
    tub_reader = MaskRegionReader(tub_mask_path)

    useful_tiles = calculate_useful_tiles(
        image_path=str(img),
        tile_deepzoom_object=dz_img,
        useful_region_annotation_path=cortex_mask_path,
        level=level,
        percentage_threshold=cortex_tile_threshold_pct,
    )

    W, H = map(int, slide.dimensions)

    memmap_path = None
    disk_dirs = None

    if storage_mode == "DISK":
        memmap_path = fibrosis_masks_dir / f"{stem}__tmp_full_mask.dat"
        full_mask = np.memmap(str(memmap_path), dtype=np.uint8, mode="w+", shape=(H, W))
        full_mask[:] = 0
        disk_dirs = tile_cache_dirs(root, method, stem)
    else:
        full_mask = np.zeros((H, W), dtype=np.uint8)

    tile_rows = []

    for (col, row) in tqdm(useful_tiles, desc=f"[{stem}] tiles ({method})"):
        tile_rgb = np.array(dz_img.get_tile(level, (col, row)))  # RGB
        th, tw = tile_rgb.shape[0], tile_rgb.shape[1]

        loc, _, size = dz_img.get_tile_coordinates(level=level, address=(col, row))
        x0, y0 = loc
        w0, h0 = size

        cortex = cortex_reader.read_region_level0(x0, y0, w0, h0)
        tubules = tub_reader.read_region_level0(x0, y0, w0, h0)
        glomeruli = glom_reader.read_region_level0(x0, y0, w0, h0)

        if cortex.shape != (th, tw):
            cortex = cv2.resize(cortex, (tw, th), interpolation=cv2.INTER_NEAREST)
        if tubules.shape != (th, tw):
            tubules = cv2.resize(tubules, (tw, th), interpolation=cv2.INTER_NEAREST)
        if glomeruli.shape != (th, tw):
            glomeruli = cv2.resize(glomeruli, (tw, th), interpolation=cv2.INTER_NEAREST)

        if method == "red_green_filtering":
            fib = fibrosis_red_green(tile_rgb, cortex, tubules, glomeruli, structural_removal=True, post_processing=post_processing)
        elif method == "stain_decon":
            fib = fibrosis_stain_deconv(tile_rgb, cortex, tubules, glomeruli, structural_removal=True, post_processing=post_processing)
        else:
            raise ValueError("Method must be one of: stain_decon, red_green_filtering")

        min_x = int(max(x0, 0))
        min_y = int(max(y0, 0))
        max_x = int(min(x0 + w0, W))
        max_y = int(min(y0 + h0, H))

        out_h = max_y - min_y
        out_w = max_x - min_x

        fib_u8 = (fib > 0).astype(np.uint8) * 255
        fib_u8 = fib_u8[:out_h, :out_w]
        tile_crop = tile_rgb[:out_h, :out_w, :]

        region = full_mask[min_y:max_y, min_x:max_x]
        full_mask[min_y:max_y, min_x:max_x] = np.maximum(region, fib_u8)

        tile_img_path = ""
        tile_mask_path = ""
        tile_comp_path = ""

        if storage_mode == "DISK":
            tile_img_path, tile_mask_path, tile_comp_path = save_disk_tile_bundle(
                dirs=disk_dirs,
                stem=stem,
                col=col,
                row=row,
                x0=min_x,
                y0=min_y,
                w=out_w,
                h=out_h,
                tile_rgb_crop=tile_crop,
                fibrosis_mask_u8=fib_u8,
            )

        if report_mode == "full":
            tile_rows.append(
                {
                    "Image name": stem,
                    "method": method,
                    "tile_col": col,
                    "tile_row": row,
                    "x0": min_x,
                    "y0": min_y,
                    "x1": max_x,
                    "y1": max_y,
                    "Fibrosis % (tile)": float((fib_u8 > 0).mean() * 100.0),
                    "Cortex % (tile)": float((cortex[:out_h, :out_w] > 0).mean() * 100.0),
                    "tile_image_path": tile_img_path,
                    "tile_fibrosis_mask_path": tile_mask_path,
                    "tile_fibrosis_component_path": tile_comp_path,
                }
            )

    fibrosis_mask_path = str(fibrosis_masks_dir / f"{stem}.tiff")
    if save_mask:
        save_binary_mask_tiff(np.asarray(full_mask), fibrosis_mask_path)

    mpp = get_mpp_um(slide)
    if mpp is None:
        mpp_x = mpp_y = 1.0
        pixel_area_um2 = 1.0
        note = "mpp_missing_used_1.0"
    else:
        mpp_x, mpp_y = mpp
        pixel_area_um2 = mpp_x * mpp_y
        note = ""

    cortex_px = count_positive_pixels_level0(cortex_mask_path)
    tub_px = count_positive_pixels_level0(tub_mask_path)
    glom_px = count_positive_pixels_level0(glom_mask_path)
    fibrosis_px = int((np.asarray(full_mask) > 0).sum())

    fibrosis_pct = (fibrosis_px / cortex_px * 100.0) if cortex_px > 0 else 0.0

    fibrosis_area_um2 = fibrosis_px * pixel_area_um2
    cortex_area_um2 = cortex_px * pixel_area_um2
    tub_area_um2 = tub_px * pixel_area_um2
    glom_area_um2 = glom_px * pixel_area_um2
    interstitium_area_um2 = max(cortex_area_um2 - tub_area_um2 - glom_area_um2, 0.0)

    summary_row = {
        "Image name": stem,
        "method": method,
        f"Fibrosis Area {method} (µm²)": fibrosis_area_um2,
        f"Fibrosis Percentage {method}": fibrosis_pct,
        "Cortex Area (µm²)": cortex_area_um2,
        "Tubules Area (µm²)": tub_area_um2,
        "Glomeruli Area (µm²)": glom_area_um2,
        "Interstitium Area (µm²)": interstitium_area_um2,
        "Edema Area (µm²)": np.nan,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "note": note,
        "Fibrosis mask path": fibrosis_mask_path if save_mask else "",
        "Tile cache dir": str(disk_dirs["base"]) if disk_dirs is not None else "",
    }

    (fibrosis_reports_dir / f"{stem}.csv").write_text(pd.DataFrame([summary_row]).to_csv(index=False), encoding="utf-8")

    if report_mode == "full":
        (fibrosis_reports_dir / f"{stem}_tiles.csv").write_text(pd.DataFrame(tile_rows).to_csv(index=False), encoding="utf-8")

    cortex_reader.close()
    glom_reader.close()
    tub_reader.close()

    if memmap_path is not None:
        try:
            memmap_path.unlink(missing_ok=True)
        except Exception:
            pass

    slide.close()
    return summary_row, str(fibrosis_reports_dir)


def parse_args():
    p = argparse.ArgumentParser(prog="FibrosisReport")
    p.add_argument("--image_path", required=True)

    fd = p.add_mutually_exclusive_group(required=True)
    fd.add_argument("-f", action="store_true")
    fd.add_argument("-d", action="store_true")

    mem = p.add_mutually_exclusive_group(required=False)
    mem.add_argument("-RAM", action="store_true")
    mem.add_argument("-DISK", action="store_true")

    p.add_argument("--method", required=True, choices=["stain_decon", "red_green_filtering"])
    p.add_argument("--report", default="simple", choices=["simple", "full"])
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--post_processing", default="True", choices=["True", "False"])
    return p.parse_args()


def find_images(image_path: Path, is_file: bool):
    if is_file:
        return [image_path]

    if (image_path / "Images").exists():
        images_dir = image_path / "Images"
    else:
        images_dir = image_path

    exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(imgs)


def main():
    args = parse_args()

    storage_mode = "DISK" if args.DISK else "RAM"
    save_mask = not args.no_mask
    post_processing = True if args.post_processing == "True" else False

    image_path = Path(args.image_path)
    images = find_images(image_path, is_file=args.f)
    if not images:
        raise FileNotFoundError(f"No images found for: {image_path}")

    summary_rows = []
    reports_dir_last = None

    for img in images:
        summary_row, reports_dir = process_one_image(
            image_path=str(img),
            method=args.method,
            report_mode=args.report,
            save_mask=save_mask,
            post_processing=post_processing,
            storage_mode=storage_mode,
        )
        summary_rows.append(summary_row)
        reports_dir_last = reports_dir

    if reports_dir_last is not None:
        out_path = Path(reports_dir_last) / f"FibrosisReport_{args.report}.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"Saved combined report: {out_path}")


if __name__ == "__main__":
    main()