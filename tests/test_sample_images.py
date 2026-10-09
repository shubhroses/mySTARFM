"""Tests that read the sample rasters in Images/ and predict t2 from them.

The mask size and the errors asserted here are the figures the README quotes
under "Results on the sample images".
"""

import runpy
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

import starfm

REPOSITORY = Path(__file__).resolve().parents[1]
IMAGES = REPOSITORY / "Images"

# The smallest and largest value stored in each file, from gdalinfo -stats.
STORED_RANGE = {
    "sim_Landsat_t1.tif": (500, 1000),
    "sim_Landsat_t2.tif": (500, 4000),
    "sim_Landsat_t4.tif": (1000, 4000),
    "sim_MODIS_t1.tif": (500, 1000),
    "sim_MODIS_t2.tif": (500, 4000),
    "sim_MODIS_t4.tif": (1000, 4000),
}


def read(name):
    return starfm.readImage(IMAGES / name)


def predict_t2():
    return starfm.prediction(
        read("sim_Landsat_t1.tif"), read("sim_MODIS_t1.tif"), read("sim_MODIS_t2.tif")
    )


def rmse_against_true_t2(image):
    truth = read("sim_Landsat_t2.tif").astype(np.float64)
    return float(np.sqrt(np.mean((image - truth) ** 2)))


# Reading


@pytest.mark.parametrize("name", sorted(STORED_RANGE))
def test_read_image_keeps_the_stored_values(name):
    image = read(name)

    assert image.shape == (150, 150, 1)
    assert image.dtype == np.float32
    assert (image.min(), image.max()) == STORED_RANGE[name]
    # OpenCV decodes the same file on its own. With IMREAD_UNCHANGED it
    # returns the signed 16-bit band without converting it.
    stored = cv2.imread(str(IMAGES / name), cv2.IMREAD_UNCHANGED)
    assert stored.dtype == np.int16
    np.testing.assert_array_equal(image[:, :, 0], stored)


def test_read_image_is_quiet_about_the_missing_georeferencing():
    # rasterio warns when it opens a raster without georeferencing, which is
    # what the sample files are. readImage only wants the pixel values.
    with warnings.catch_warnings():
        warnings.simplefilter("error", NotGeoreferencedWarning)
        read("sim_Landsat_t1.tif")


def test_read_image_keeps_every_band_in_file_order(tmp_path):
    bands = np.stack(
        [np.arange(20, dtype=np.uint16).reshape(4, 5) + offset for offset in (0, 30000)]
    )
    path = tmp_path / "two_bands.tif"
    geotiff = dict(driver="GTiff", height=4, width=5, count=2, dtype="uint16")
    with rasterio.open(path, "w", transform=from_origin(0, 0, 30, 30), **geotiff) as f:
        f.write(bands)

    image = starfm.readImage(path)

    assert image.shape == (4, 5, 2)
    np.testing.assert_array_equal(image[:, :, 0], bands[0])
    np.testing.assert_array_equal(image[:, :, 1], bands[1])


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_save_image_writes_what_read_image_reads_back(tmp_path, monkeypatch):
    # saveImage writes to results/output.tif under the working directory.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    image = np.arange(24, dtype=np.float32).reshape(3, 4, 2) + 0.5

    starfm.saveImage(image)

    np.testing.assert_array_equal(starfm.readImage("results/output.tif"), image)


# Prediction


def test_edge_mask_on_the_sample_image():
    mask = starfm.sobel_edge_detection(read("sim_Landsat_t1.tif"))

    assert mask.sum() == 6124
    # The disc keeps 25 pixels clear of every border, further than the mask
    # reaches from its edge.
    assert not mask[:, -14:].any()


def test_prediction_on_the_sample_images():
    predicted = predict_t2()

    assert predicted.shape == (150, 150, 1)
    assert predicted.dtype == np.float32
    assert (predicted.min(), predicted.max()) == (500, 4000)
    assert rmse_against_true_t2(predicted) == pytest.approx(153.9, abs=0.05)
    # For comparison, the coarse image on its own.
    assert rmse_against_true_t2(read("sim_MODIS_t2.tif")) == pytest.approx(
        743.7, abs=0.05
    )


def test_prediction_of_every_pixel_on_the_sample_images(monkeypatch):
    monkeypatch.setattr(
        starfm, "sobel_edge_detection", lambda image: np.ones(image.shape[:2])
    )

    predicted = predict_t2()

    assert rmse_against_true_t2(predicted) == pytest.approx(12.7, abs=0.05)


def test_script_predicts_at_full_depth(monkeypatch, capsys):
    shown = []
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gca().images[0]))
    # The script reads Images/ by relative path and appends to sys.path.
    monkeypatch.chdir(REPOSITORY)
    monkeypatch.setattr(sys, "path", list(sys.path))

    runpy.run_path(str(REPOSITORY / "src" / "starfm.py"), run_name="__main__")

    printed = capsys.readouterr().out
    assert "F0 shape: (150, 150, 1)" in printed
    assert "F1 shape: (150, 150, 1)" in printed
    # The figure is scaled to the data, 500 to 4000, not to 8-bit values.
    assert len(shown) == 1
    assert shown[0].get_clim() == (500, 4000)
    plt.close("all")
