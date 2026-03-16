# Utils/tile_selection.py

import xml.etree.ElementTree as ET

import numpy as np
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from joblib import delayed
from tqdm_joblib import ParallelPbar


def return_tile(path: str, tile_size: int, overlap: int, level: int, address: tuple):
    temp_image_object = open_slide(path)
    temp_slide_object = DeepZoomGenerator(temp_image_object, tile_size, overlap)
    return np.array(temp_slide_object.get_tile(level, address))


def calculate_useful_tiles(
    image_path: str,
    tile_deepzoom_object: openslide.deepzoom.DeepZoomGenerator,
    useful_region_annotation_path: str = None,
    level: int = None,
    percentage_threshold: int = None,
):
    def process_tile(args):
        tile_size, overlap, level, address, percentage_threshold = args

        mask_tile = return_tile(useful_region_annotation_path, tile_size, overlap, level, address)
        mask_tile_percentage = np.sum(mask_tile) / (mask_tile.shape[0] * mask_tile.shape[1]) * 100

        return address if mask_tile_percentage > percentage_threshold else None

    total_col_tiles, total_row_tiles = tile_deepzoom_object.level_tiles[level]
    root = ET.fromstring(tile_deepzoom_object.get_dzi(format="png"))
    tile_size = int(root.attrib["TileSize"])
    overlap = int(root.attrib["Overlap"])

    args_list = [
        (tile_size, overlap, level, (col, row), percentage_threshold)
        for col in range(total_col_tiles)
        for row in range(total_row_tiles)
    ]

    with ParallelPbar("Calculating useful tiles...")(n_jobs=12, backend="loky") as parallel:
        results = parallel(delayed(process_tile)(args) for args in args_list)

    return [tile_address for tile_address in results if tile_address is not None]