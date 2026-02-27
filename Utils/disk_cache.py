# Utils/disk_cache.py

from pathlib import Path
import numpy as np

from Utils.image_utils import rgb_to_bgr, save_png


def tile_cache_dirs(root: Path, method: str, stem: str) -> dict:
    base = root / "Fibrosis Tile Cache" / method / stem
    d = {
        "base": base,
        "tile_rgb": base / "tile_rgb",
        "fibrosis_mask": base / "fibrosis_mask",
        "fibrosis_component": base / "fibrosis_component",
    }
    for p in d.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return d


def save_disk_tile_bundle(
    dirs: dict,
    stem: str,
    col: int,
    row: int,
    x0: int,
    y0: int,
    w: int,
    h: int,
    tile_rgb_crop: np.ndarray,
    fibrosis_mask_u8: np.ndarray,
):
    name = f"{stem}__c{col}_r{row}_x{x0}_y{y0}_w{w}_h{h}.png"

    tile_path = dirs["tile_rgb"] / name
    mask_path = dirs["fibrosis_mask"] / name
    comp_path = dirs["fibrosis_component"] / name

    save_png(rgb_to_bgr(tile_rgb_crop), tile_path)
    save_png(fibrosis_mask_u8, mask_path)

    m = (fibrosis_mask_u8 > 0).astype(np.uint8)
    comp = tile_rgb_crop.copy()
    comp[m == 0] = 0
    save_png(rgb_to_bgr(comp), comp_path)

    return str(tile_path), str(mask_path), str(comp_path)