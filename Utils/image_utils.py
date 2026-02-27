# Utils/image_utils.py

from pathlib import Path

import cv2
import numpy as np
import pyvips
import openslide
import skimage as ski


def process_mask(mask: np.ndarray, effective_tile_size: tuple, invert: bool = False):
    mask = np.asarray(mask)
    mask = mask / 255 if np.max(mask) == 255 else mask
    if invert:
        mask = 1 - mask
    mask = cv2.resize(mask, (effective_tile_size[1], effective_tile_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask = np.stack([mask] * 3, axis=-1)
    return mask


def clean_image_by_min_area(binary_image, min_area):
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros(binary_image.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(keep, [contour], -1, 255, thickness=cv2.FILLED)
    out = cv2.bitwise_and(binary_image.astype(np.uint8), keep)
    return (out > 0).astype(np.uint8)


def enhance_contrast(tile, saturated_pixels=0.35, normalize=False):
    img_flat = tile.flatten()
    if saturated_pixels > 0:
        low_perc = saturated_pixels / 2.0
        high_perc = 100 - (saturated_pixels / 2.0)
        vmin, vmax = np.percentile(img_flat, [low_perc, high_perc])
    else:
        vmin, vmax = img_flat.min(), img_flat.max()

    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    img_stretched = np.clip((tile - vmin) / denom, 0, 1)

    if normalize:
        if tile.dtype == np.uint8:
            img_stretched = (img_stretched * 255).astype(np.uint8)
        elif tile.dtype == np.uint16:
            img_stretched = (img_stretched * 65535).astype(np.uint16)
        elif tile.dtype in (np.float32, np.float64):
            img_stretched = img_stretched.astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype {tile.dtype}")
    else:
        img_stretched = (img_stretched * (vmax - vmin) + vmin).astype(tile.dtype)

    return img_stretched


def get_mpp_um(slide: openslide.OpenSlide):
    props = slide.properties
    try:
        mpp_x = float(props.get(openslide.PROPERTY_NAME_MPP_X))
        mpp_y = float(props.get(openslide.PROPERTY_NAME_MPP_Y))
        return mpp_x, mpp_y
    except Exception:
        return None


def save_binary_mask_tiff(mask_2d_uint8: np.ndarray, saving_path: str):
    mask_2d_uint8 = (mask_2d_uint8 > 0).astype(np.uint8) * 255
    img = pyvips.Image.new_from_array(mask_2d_uint8)
    h, w, c = img.height, img.width, img.bands

    xml_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
<Image ID="Image:0">
    <Pixels DimensionOrder="XYCZT"
            ID="Pixels:0"
            SizeC="{c}"
            SizeT="1"
            SizeX="{w}"
            SizeY="{h}"
            SizeZ="1"
            Type="uint8">
    </Pixels>
</Image>
</OME>"""

    img = img.copy()
    img.set_type(pyvips.GValue.gint_type, "page-height", h)
    img.set_type(pyvips.GValue.gstr_type, "image-description", xml_str)

    Path(saving_path).parent.mkdir(parents=True, exist_ok=True)
    img.tiffsave(
        saving_path,
        compression="lzw",
        tile=True,
        tile_width=512,
        tile_height=512,
        Q=100,
        pyramid=True,
        subifd=True,
    )
    return True


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb
    return img_rgb[:, :, ::-1]


def save_png(img: np.ndarray, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)