# Utils/zarr_helpers.py

import numpy as np
from openslide import open_slide


def _open_best_zarr_array(tiff_path: str):
    import tifffile
    import zarr

    store = tifffile.imread(tiff_path, aszarr=True)
    zobj = zarr.open(store, mode="r")

    arrays = []
    if hasattr(zobj, "shape"):
        arrays = [zobj]
    else:
        for _, a in zobj.arrays():
            arrays.append(a)

    if not arrays:
        raise RuntimeError("No arrays found in Zarr store")

    def score(a):
        shp = a.shape
        if len(shp) == 2:
            return shp[0] * shp[1]
        if len(shp) == 3:
            return shp[0] * shp[1]
        return 0

    return max(arrays, key=score)


def count_positive_pixels_level0(mask_path: str) -> int:
    # Count >0 pixels at level-0
    try:
        import dask.array as da

        arr = _open_best_zarr_array(mask_path)
        darr = da.from_zarr(arr)
        if darr.ndim == 3:
            darr = darr[:, :, 0]
        return int((darr > 0).sum().compute())
    except Exception:
        slide = open_slide(mask_path)
        w, h = slide.dimensions
        tile = 2048
        total = 0
        for y in range(0, h, tile):
            hh = min(tile, h - y)
            for x in range(0, w, tile):
                ww = min(tile, w - x)
                patch = np.array(slide.read_region((x, y), 0, (ww, hh)))[:, :, 0]
                total += int((patch > 0).sum())
        slide.close()
        return total