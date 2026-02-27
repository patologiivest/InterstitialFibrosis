# InterstitialFibrosis
Implementation of the methods described in "Image Analysis for Non-Neoplastic Kidney Disease: Utilizing Morphological Segmentation to Improve Quantification of Interstitial Fibrosis". Compute interstitial fibrosis from Sirius-red-stained whole-slide images (WSI) using QuPath-style GeoJSON annotations for cortex, tubules, and glomeruli. The tool generates fibrosis masks, summary reports, and (optionally) per-tile debug outputs.

![Semi-automated method for interstitial specific segmentation of renal fibrotic tissue](src/img/Figure1.png "Method")

## Expected folder structure

```text
ProjectRoot/
├── Images/
│   └── Image1.svs
├── Annotations/
│   ├── Cortex/
│   │   └── Image1.geojson
│   ├── Glomeruli/
│   │   └── Image1.geojson
│   └── Tubules/
│       └── Image1.geojson
├── FibrosisReport.py
└── Utils/
    ├── __init__.py
    ├── geojson_mask.py
    ├── zarr_helpers.py
    ├── mask_reader.py
    ├── image_utils.py
    ├── fibrosis_methods.py
    ├── paths.py
    ├── tile_selection.py
    └── disk_cache.py
```

## Usage

### Environment

```bash
conda env create -f Env/FibrosisReport.yml
```

### CLI

```bash
python FibrosisReport.py --image_path [Required] -f/-d -RAM/-DISK --method [Required] --report [Optional] --no-mask [Optional] --post_processing [Optional]
```

### Options

- `--image_path` (required)  
  Path to a single image file (`-f`) or a directory (`-d`).
- `-f` / `-d` (required)  
  Treat `image_path` as file or directory.
- `-RAM` / `-DISK` (optional)  
  `-RAM` (default): store only full mask in RAM.  
  `-DISK`: additionally saves per-tile debug outputs to disk.
- `--method` (required)  
  `stain_decon`  
  `red_green_filtering`
- `--report` (optional)  
  `simple` (default): per-image summary CSV.  
  `full`: adds tile-level CSV and stores tile output paths.
- `--no-mask` (optional)  
  Disable saving the final fibrosis mask TIFF.
- `--post_processing` (optional)  
  `True` (default): remove small components.  
  `False`: keep raw output.

---

## Examples

### 1) Run on a single file (RAM mode)
```bash
python FibrosisReport.py --image_path "ProjectRoot/Images/Image1.svs" -f -RAM --method stain_decon --report simple --post_processing True
```

### 2) Run on all images in a folder (full report)
```bash
python FibrosisReport.py --image_path "ProjectRoot" -d -RAM --method red_green_filtering --report full --post_processing True
```

### 3) DISK mode (save per-tile bundles)
```bash
python FibrosisReport.py --image_path "ProjectRoot/Images/Image1.svs" -f -DISK --method stain_decon --report full --post_processing True
```
---

# License

Shield: [![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
