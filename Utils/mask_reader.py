# Utils/mask_reader.py

import numpy as np
from openslide import open_slide

from Utils.zarr_helpers import _open_best_zarr_array


class MaskRegionReader:
    def __init__(self, mask_path: str):
        self.mask_path = mask_path
        self.mode = None
        self._arr = None
        self._slide = None

        try:
            self._arr = _open_best_zarr_array(mask_path)
            self.mode = "zarr"
            return
        except Exception:
            pass

        self._slide = open_slide(mask_path)
        self.mode = "openslide"

    def read_region_level0(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        if self.mode == "zarr":
            patch = np.asarray(self._arr[y : y + h, x : x + w])
            if patch.ndim == 3:
                patch = patch[:, :, 0]
            return patch.astype(np.uint8)

        patch = np.array(self._slide.read_region((x, y), 0, (w, h)))[:, :, 0]
        return patch.astype(np.uint8)

    def close(self):
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass