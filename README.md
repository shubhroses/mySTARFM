# mySTARFM

A small NumPy version of STARFM, the Spatial and Temporal Adaptive Reflectance Fusion Model for blending fine- and coarse-resolution satellite images. The distance, filtering and weighting functions are adapted from [starfm4py](https://github.com/nmileva/starfm4py) by Nikolina Mileva and rewritten to work on one moving window at a time, without dask. The repository adds one experiment on top: run the expensive moving-window prediction only on pixels near edges in the fine image, and let every other pixel take the coarse value.

This is a prototype written in February and March 2023. It runs end to end on the simulated test images in `Images/`, which also come from starfm4py. Read [Known limitations](#known-limitations) before pointing it at real data, and [Origin and credits](#origin-and-credits) for what was taken from the upstream project.

## Background

STARFM (Gao et al., 2006) predicts a fine-resolution image, such as Landsat, for a date on which only a coarse-resolution image, such as MODIS, was acquired. It works from a fine/coarse pair taken on a base date and the coarse image from the prediction date. This code takes exactly one such pair.

| Name in the code | Meaning | Sample file |
| --- | --- | --- |
| `F0` | fine image, base date | `Images/sim_Landsat_t1.tif` |
| `C0` | coarse image, base date | `Images/sim_MODIS_t1.tif` |
| `C1` | coarse image, prediction date | `Images/sim_MODIS_t2.tif` |
| `F1` | predicted fine image, prediction date | truth for comparison: `Images/sim_Landsat_t2.tif` |

Each predicted pixel is a weighted sum of `F0 + (C1 - C0)` over the pixels in a moving window that are spectrally similar to the centre pixel. The weights favour neighbours where the fine and coarse images agree, where little changed between the two dates, and that sit close to the centre.

## How `prediction(F0, C0, C1)` works

The entry point is `prediction` in `src/starfm.py`. The three inputs are arrays of the same shape `(rows, cols, bands)`. The script and the notebooks load them with `cv2.imread`, which gives `uint8` arrays of shape `(150, 150, 3)` for the sample files.

1. **Edge mask.** `sobel_edge_detection` converts `F0` to grayscale, takes 3 x 3 Sobel gradients in x and y, rescales the gradient magnitude to the range 0 to 255, thresholds it with Otsu's method, and widens the result by convolving it with a 15 x 15 box of ones. The mask therefore covers every pixel within 7 rows and 7 columns of a detected edge pixel. It is computed once and used for every band.
2. **Default value.** The output starts as a copy of `C1`.
3. **Windowed prediction.** For every pixel inside the mask, and for each band, the code cuts a 31 x 31 window (`windowSize`) out of zero-padded copies of `F0`, `C0` and `C1` and computes:
   - a spectral distance from `F0 - C0` and a temporal distance from `C1 - C0`, each as `1 / (|difference| + 1)`, so despite the name a larger value means a closer match;
   - a spatial distance `1 / (1 + d / spatImp)`, where `d` is the distance in pixels from the window centre;
   - the set of similar pixels: those whose `F0` value lies within `2 * std / numberClass` of the centre pixel (`std` is taken over the non-zero `F0` values in the window) and whose absolute spectral difference is smaller than the centre pixel's plus the combined sensor uncertainty;
   - normalised weights: the product of the three distances over the similar pixels. If the centre pixel has zero spectral or temporal difference, it takes all of the weight.
4. **Output.** The pixel is set to the weighted sum of `F0 + (C1 - C0)` over the window.

Pixels outside the mask keep their `C1` value, so the window search is skipped for them. To run the search on every pixel instead, replace the mask with an array of ones. `prediction` contains a commented-out line that does this for a 150 x 150 image.

Two switches in `src/parameters.py` change the weighting: `logWeight = True` applies `log(distance + 1)` to the spectral and temporal distances before combining them, and `temp = True` adds a temporal test to the similar-pixel filter. Both are off by default.

## Results on the sample images

These figures come from running the code on the sample images in the locked Poetry environment.

- The edge mask selects 8,192 of the 22,500 pixels (36%), so the window search runs on roughly a third of the image. The count includes the 14 right-most columns in full (2,100 pixels), which are selected only because of the file-reading problem described under [Known limitations](#known-limitations).
- The masked prediction equals the all-pixels prediction stored in `results/output.tif` at all but 129 pixels.
- `compareImages.ipynb` scores the masked prediction against the true fine image at t2 and records RMSE 1.25, MAE 1.21, PSNR 46.2 dB and SSIM 0.988. The notebook loads the all-pixels prediction as `F1_control` but never scores it. Putting `F1_control` through the same RMSE cell gives 1.18; that figure is not recorded in the notebook.

All of this is measured on 8-bit values that only span 0 to 15, and the recorded MAE, PSNR and SSIM are distorted by the way they are computed, so the figures are not a benchmark of the method. [Known limitations](#known-limitations) explains each point.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/starfm.py` | The fusion code and the edge mask. Run as a script, it predicts t2 for the sample images |
| `src/parameters.py` | Tunables: `windowSize` (31), `spatImp` (150), `numberClass` (4), sensor uncertainties (0.03 each), `logWeight`, `temp`. `path` and `sizeSlices` are left over from starfm4py and are not used |
| `src/spectralDistance.ipynb` | The algorithm built up step by step on a 3 x 3 example with a 3 x 3 window. Writes `results/prediction.tif`, which is not committed |
| `src/edgeDetection.ipynb` | Development of the Sobel and Otsu edge mask, with the mask plotted for the sample image. The notebook's own copy of the function widens with a 5 x 5 box; `starfm.py` uses 15 x 15 |
| `src/convertArrayToImage.ipynb` | Runs `prediction` on the sample images, plots inputs and output, and writes the result to `results/output.tif` with rasterio. Re-running it replaces the committed file with a masked prediction |
| `src/compareImages.ipynb` | Scores the edge-masked prediction against the true fine image at t2 with RMSE, PSNR, MAE and SSIM |
| `src/dividePixels.ipynb` | Side experiment: a ring drawn in a 30 x 30 array and upsampled to 100 x 100 with nearest-neighbour interpolation |
| `Images/` | Simulated 150 x 150 fine ("Landsat") and coarse ("MODIS") rasters from the starfm4py test data, for dates t1, t2 and t4. The code uses t1 and t2. The t4 pair belongs to a different upstream test case and is not used |
| `results/output.tif` | The t2 prediction from the all-pixels version of the code, committed on 26 February 2023, before the edge mask was added. Running the current code with the mask replaced by ones reproduces it exactly. `compareImages.ipynb` loads it as `F1_control` |
| `pyproject.toml`, `poetry.lock` | Poetry project file and the lock file for the environment the notebooks were run in |

The notebooks are development notes rather than polished reports. Two of them contain a cell that stops with an error. The last cell of `convertArrayToImage.ipynb` prints the shape of what `cv2.imread` returned for `results/prediction.tif`, and that is `None`: the file holds 64-bit floats, which OpenCV cannot read, and it is absent from a fresh clone. The third cell of `dividePixels.ipynb` is an abandoned attempt to draw a diagonal line.

## Running it

The project uses [Poetry](https://python-poetry.org/). `pyproject.toml` asks for Python 3.10 or newer; the notebooks were run on Python 3.10.8.

```bash
git clone https://github.com/shubhroses/mySTARFM.git
cd mySTARFM
poetry install --no-root
poetry run python src/starfm.py
```

`--no-root` installs the dependencies only. The code is run from `src/` and is not set up as an installable package: without the flag, Poetry 2 installs the dependencies and then exits with an error because it finds no `mystarfm` package.

These commands were last checked in October 2026 with Poetry 2.5.1 and Python 3.10.18 on an Apple silicon Mac. The install completes from the lock file. Poetry warns that the pinned opencv-python 4.7.0.68 has since been yanked from PyPI in favour of 4.7.0.71, and installs it anyway.

Run the script from the repository root, because it loads `Images/sim_Landsat_t1.tif`, `Images/sim_MODIS_t1.tif` and `Images/sim_MODIS_t2.tif` by relative path. It prints the shapes of `F0` and `F1`, both `(150, 150, 3)`, and then shows the prediction in a matplotlib window. It does not write a file: the `saveImage(F1)` call at the end of the script is commented out. Uncommenting it writes the prediction to `results/output.tif`, replacing the committed reference file.

Without Poetry, `src/starfm.py` needs `numpy`, `scipy`, `opencv-python`, `rasterio` and `matplotlib`. In October 2026 the script also ran unchanged, and produced the same array, on current releases of those packages (Python 3.13, NumPy 2.5, SciPy 1.18, OpenCV 5.0, rasterio 1.5, Matplotlib 3.11). `compareImages.ipynb` needs `scikit-image` as well, and its SSIM cell depends on an old release such as the locked 0.19.3: it passes `multichannel=True`, which scikit-image 0.26 no longer supports, so the call raises a `ValueError` there. `channel_axis=2` is the replacement.

### Notebooks

Open the notebooks with `src/` as the working directory. They `import starfm` directly and read the sample images from `../Images/`. The Poetry environment includes `ipykernel` but no notebook server, so use an editor that can run that environment as a kernel (the notebooks were written in VS Code) or add JupyterLab yourself.

In October 2026 the code cells of all five notebooks were executed in order in the locked environment. Every stored printed or returned value was reproduced, including the four metrics in `compareImages.ipynb`, and so were the two errors described above. The stored figures were not compared.

### Calling it from Python

In a Python session or notebook started in `src/`:

```python
import cv2
from starfm import prediction

F0 = cv2.imread("../Images/sim_Landsat_t1.tif")  # fine image, base date
C0 = cv2.imread("../Images/sim_MODIS_t1.tif")    # coarse image, base date
C1 = cv2.imread("../Images/sim_MODIS_t2.tif")    # coarse image, prediction date

F1 = prediction(F0, C0, C1)                      # shape (150, 150, 3), dtype uint8
```

## Known limitations

- **Images are loaded at 8-bit precision, and not completely.** The sample rasters are single-band, signed 16-bit TIFFs with values from 500 to 4000. `cv2.imread` with its default flags returns them as 8-bit, three-channel arrays. The three channels are identical, so the fusion runs three times over the same data, and every value is divided by 256 and rounded down, which leaves integers no larger than 15. That is why the plotted arrays look almost black. The same default read also returns zeros for 143 of the 150 pixels in each of the six right-most columns (seen with OpenCV 4.7.0 and 5.0.0 on macOS). This shows up as a dark strip in the notebook figures, a false vertical edge that puts the 14 right-most columns into the mask, and a strip of zeros in `results/output.tif`. Reading with rasterio, which is already a dependency, or with `cv2.IMREAD_UNCHANGED` returns the full 16-bit band with every column intact. That is a 2-D `int16` array, which `prediction` does not accept as it is; stacked into three channels and cast to `float32`, the bands run through the code unchanged.
- **Arithmetic is unsigned.** The arrays stay `uint8` through the distance calculations, so a difference that should be negative wraps around to a large positive number. On the sample data this happens to the spectral difference at 2,033 of the 22,500 pixels, where it makes a small difference look like a very large one, and no neighbour darker than the centre pixel is ever counted as similar. The same wraparound inflates the MAE recorded in `compareImages.ipynb`: recomputed with signed arithmetic it is 0.41, not 1.21. Because the output array is a copy of `C1`, predictions are truncated to integers whenever the inputs are integer arrays.
- **The metrics assume an 8-bit scale.** PSNR and SSIM in `compareImages.ipynb` use a 0 to 255 range, far wider than data that tops out at 15, so both come out more flattering than they would on the data's own range. Recomputed with a range of 15, the same arrays give a PSNR of 21.6 dB and an SSIM of 0.876.
- **One window at a time.** The prediction is a Python loop over pixels with no chunking or parallelism. For the 150 x 150 samples that takes about 2 seconds with the mask and about 5 seconds without it on an Apple silicon Mac; it will be slow for real scenes. starfm4py partitions the image with dask, stores the windows as zarr files and processes them in slices. That part was not carried over.
- **One input pair, limited input types.** Only one fine/coarse pair is supported. The edge mask calls OpenCV's BGR-to-gray conversion on `F0`, so a single-band array is rejected, and so are `int16` and `float64` arrays. Three-channel `uint8`, `uint16` and `float32` arrays work.
- **No tests.** There is no test suite. The notebooks are the only record of checks.

## Origin and credits

- STARFM was introduced in F. Gao, J. Masek, M. Schwaller and F. Hall, "On the blending of the Landsat and MODIS surface reflectance: predicting daily Landsat surface reflectance", IEEE Transactions on Geoscience and Remote Sensing, 44(8), 2207-2218, 2006, [doi:10.1109/TGRS.2006.872081](https://doi.org/10.1109/TGRS.2006.872081).
- This repository is derived from [starfm4py](https://github.com/nmileva/starfm4py) by Nikolina Mileva, which is published under the GNU General Public License v3.0. Taken from that project:
  - `spectral_distance`, `temporal_distance`, `spatial_distance`, `combination_distance` (`comb_distance` upstream), `similarity_threshold`, `similarity_pixels`, `filtering` and `weighting` in `src/starfm.py` follow the upstream functions of those names, with the dask array calls replaced by NumPy and each function reduced to a single flattened window. The weighted sum that produces each pixel follows upstream's `predict`. Earlier copies of the same functions are in `src/spectralDistance.ipynb`.
  - `src/parameters.py` is the upstream file with one constant, `padAmount`, added. `src/spectralDistance.ipynb` holds a copy of it with a 3 x 3 window, and its GeoTIFF-writing cell follows upstream's `Tests/test.py`.
  - The six rasters in `Images/` are unmodified copies of files in upstream's `Tests/Test_1` (t1 and t2) and `Tests/Test_2` (t4).
- Not taken from starfm4py: the loop that cuts each window out of zero-padded arrays, the handling of several bands, the Sobel and Otsu edge mask with its fall-back to the coarse value, `saveImage`, and the comparison metrics.
- The paper behind starfm4py is N. Mileva, S. Mecklenburg and F. Gascon, "New tool for spatio-temporal image fusion in remote sensing: a case study approach using Sentinel-2 and Sentinel-3 data", Proc. SPIE 10789, Image and Signal Processing for Remote Sensing XXIV, 2018, [doi:10.1117/12.2327091](https://doi.org/10.1117/12.2327091). `src/spectralDistance.ipynb` links the copy hosted by the University of Augsburg: <https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/deliver/index/docId/78805/file/STARFM_paper.pdf>. The starfm4py README asks that published work using its code cite this paper.
