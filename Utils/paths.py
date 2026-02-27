# Utils/paths.py

from pathlib import Path

from Utils.geojson_mask import createMask


def project_root_from_image(image_path: Path) -> Path:
    if image_path.parent.name.lower() == "images":
        return image_path.parent.parent
    if image_path.name.lower() == "images":
        return image_path.parent
    return image_path.parent


def ensure_masks_from_annotations(image_path: str) -> dict:
    img = Path(image_path)
    root = project_root_from_image(img)
    stem = img.stem

    ann_root = root / "Annotations"
    mask_root = root / "Masks"

    mapping = {
        "Cortex": (ann_root / "Cortex" / f"{stem}.geojson", mask_root / "Cortex" / f"{stem}.tiff"),
        "Glomeruli": (ann_root / "Glomeruli" / f"{stem}.geojson", mask_root / "Glomeruli" / f"{stem}.tiff"),
        "Tubules": (ann_root / "Tubules" / f"{stem}.geojson", mask_root / "Tubules" / f"{stem}.tiff"),
    }

    out = {}
    for key, (geo, tif) in mapping.items():
        if not geo.exists():
            raise FileNotFoundError(f"Missing annotation file: {geo}")
        if not tif.exists():
            tif.parent.mkdir(parents=True, exist_ok=True)
            createMask(str(geo), str(tif.parent), str(img), save_path=str(tif))
        out[key] = str(tif)

    return out