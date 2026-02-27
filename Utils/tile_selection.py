# Utils/tile_selection.py

from openslide.deepzoom import DeepZoomGenerator

from Utils.mask_reader import MaskRegionReader


def calculate_useful_tiles_by_coords(
    dz_img: DeepZoomGenerator,
    level: int,
    cortex_reader: MaskRegionReader,
    threshold_pct: float,
):
    total_cols, total_rows = dz_img.level_tiles[level]
    useful = []

    for col in range(total_cols):
        for row in range(total_rows):
            loc, _, size = dz_img.get_tile_coordinates(level=level, address=(col, row))
            x0, y0 = loc
            w0, h0 = size

            cortex_patch = cortex_reader.read_region_level0(x0, y0, w0, h0)
            pct = float((cortex_patch > 0).mean() * 100.0)

            if pct >= threshold_pct:
                useful.append((col, row))

    return useful