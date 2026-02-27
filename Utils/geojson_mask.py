# Utils/geojson_mask.py

import os
import json
from pathlib import Path

import cv2
import numpy as np
import pyvips
from openslide import open_slide


def createMask(geojson_path, output_path, image_name, save_path=None):
    def __add_image_metadata(pyvipsImage, pyvipsImage_path):
        img_tmp = pyvipsImage.copy()
        h = img_tmp.height
        w = img_tmp.width
        c = img_tmp.bands

        img_tmp = pyvipsImage.copy()
        img_tmp.set_type(pyvips.GValue.gint_type, "page-height", h)
        img_tmp.set_type(
            pyvips.GValue.gstr_type,
            "image-description",
            f"""<?xml version="1.0" encoding="UTF-8"?>
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
</OME>""",
        )

        img_tmp.tiffsave(
            pyvipsImage_path,
            compression="lzw",
            tile=True,
            tile_width=512,
            tile_height=512,
            Q=100,
            pyramid=True,
            subifd=True,
        )
        return True

    width, height = map(int, open_slide(image_name).dimensions)

    with open(geojson_path, "r") as f:
        data = json.load(f)

    features = data["features"]
    mask = np.zeros((height, width), dtype=np.uint8)

    for feature in features:
        geometry = feature["geometry"]
        geom_type = geometry["type"]
        polygons = geometry["coordinates"]

        if geom_type == "Polygon":
            polygons = [polygons]
        elif geom_type == "MultiPolygon":
            pass
        else:
            continue

        for polygon in polygons:
            if len(polygon) == 0:
                continue

            outer_contour = polygon[0]
            holes = polygon[1:]

            outer_x = [int(coord[0]) for coord in outer_contour]
            outer_y = [int(coord[1]) for coord in outer_contour]
            outer_arr = np.array(list(zip(outer_x, outer_y)), dtype=np.int32)
            cv2.drawContours(mask, [outer_arr], -1, 255, -1)

            for hole in holes:
                hole_x = [int(coord[0]) for coord in hole]
                hole_y = [int(coord[1]) for coord in hole]
                hole_arr = np.array(list(zip(hole_x, hole_y)), dtype=np.int32)
                cv2.drawContours(mask, [hole_arr], -1, 0, -1)

    vips_mask = pyvips.Image.new_from_array(mask)

    if save_path is None:
        if ("Glomerulli" in geojson_path) or ("Glomeruli" in geojson_path):
            mask_filename = os.path.join(output_path, f"{Path(geojson_path).stem}-Glome.tiff")
        else:
            mask_filename = os.path.join(output_path, f"{Path(geojson_path).stem}.tiff")
    else:
        mask_filename = save_path

    Path(mask_filename).parent.mkdir(parents=True, exist_ok=True)
    __add_image_metadata(pyvipsImage=vips_mask, pyvipsImage_path=mask_filename)
    return True