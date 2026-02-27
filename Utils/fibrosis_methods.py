# Utils/fibrosis_methods.py

import cv2
import numpy as np
import skimage as ski
import histomicstk

from Utils.image_utils import process_mask, clean_image_by_min_area, enhance_contrast


def fibrosis_red_green(
    tile: np.ndarray,
    tile_cortex: np.ndarray,
    tile_tubules: np.ndarray,
    tile_glomeruli: np.ndarray,
    structural_removal: bool = True,
    post_processing: bool = True,
):
    def process_tubules_mask():
        tubules_mask_color = enhance_contrast(cv2.subtract(tile_g, tile_b), 0.35).astype(np.float32)
        tubules_mask_color = (tubules_mask_color > ski.filters.threshold_otsu(tubules_mask_color)).astype(np.uint8)
        return cv2.bitwise_or(tile_tubules.astype(np.uint8), tubules_mask_color.astype(np.uint8))

    def create_fibrosis_mask():
        fibrosis_by_color = enhance_contrast(cv2.subtract(tile_r, tile_g), 0.7).astype(np.float32)
        fibrosis_by_color = (fibrosis_by_color > ski.filters.threshold_otsu(fibrosis_by_color)).astype(np.uint8)
        return process_mask(fibrosis_by_color, tile.shape, False)

    def process_resultant_tile(resultant_image):
        grey = cv2.cvtColor(resultant_image, cv2.COLOR_RGB2GRAY)
        th = ski.filters.threshold_otsu(grey)
        mask = (grey > th).astype(np.uint8)
        if post_processing:
            mask = clean_image_by_min_area(mask, 100)
        return mask

    tile_r, tile_g, tile_b = cv2.split(tile)
    tile_r, tile_g, tile_b = tile_r.astype(np.float32), tile_g.astype(np.float32), tile_b.astype(np.float32)

    tile_cortex = process_mask(tile_cortex, tile.shape, False)
    tile_glomeruli = process_mask(tile_glomeruli, tile.shape, True)
    tile_tubules = process_mask(process_tubules_mask(), tile.shape, True)

    tile_fibrosis = create_fibrosis_mask()

    if structural_removal:
        res = tile * tile_cortex * tile_glomeruli * tile_tubules * tile_fibrosis
    else:
        res = tile * tile_cortex * tile_fibrosis

    return process_resultant_tile(res)


def fibrosis_stain_deconv(
    tile: np.ndarray,
    tile_cortex: np.ndarray,
    tile_tubules: np.ndarray,
    tile_glomeruli: np.ndarray,
    structural_removal: bool = True,
    post_processing: bool = True,
):
    tile_cortex = process_mask(tile_cortex, tile.shape, False)
    tile_glomeruli = process_mask(tile_glomeruli, tile.shape, True)
    tile_tubules = process_mask(tile_tubules, tile.shape, True)

    stain_color_map = histomicstk.preprocessing.color_deconvolution.stain_color_map
    stains = ["hematoxylin", "eosin", "null"]
    W = np.array([stain_color_map[s] for s in stains]).T

    deconv = histomicstk.preprocessing.color_deconvolution.color_deconvolution(tile, W)
    _, binary = cv2.threshold(deconv.Stains[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    binary = process_mask(binary, tile.shape, True)

    if structural_removal:
        final = tile_cortex * tile_glomeruli * tile_tubules * binary
    else:
        final = tile_cortex * binary

    out = (final > 0).astype(np.uint8)[:, :, 0]
    if post_processing:
        out = clean_image_by_min_area(out, 100)
    return out