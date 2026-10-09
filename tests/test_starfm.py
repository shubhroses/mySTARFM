"""Tests for the window functions and the edge mask in src/starfm.py.

Every expected value is small enough to work out by hand, and the comments
show the working. The functions that look at the centre of a window read
windowSize, mid_idx and padAmount from the starfm module each time they are
called, so the window3 fixture swaps those three names to shrink the moving
window from 31 x 31 to 3 x 3.
"""

import numpy as np
import pytest

import starfm

# spatial_distance gives each pixel 1 / (1 + d / spatImp), where d is its
# distance from the window centre and spatImp is 150. In a 3 x 3 window the
# four direct neighbours have d = 1 and the four corners have d = sqrt(2).
NEIGHBOUR = 150 / 151
CORNER = 1 / (1 + np.sqrt(2) / 150)

# The worked example from src/spectralDistance.ipynb: a 3 x 3 image, one band,
# predicted with a 3 x 3 window. Flattened, each array is also the window
# around the centre pixel.
F0 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
C0 = np.ones((3, 3))
C1 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


@pytest.fixture
def window3(monkeypatch):
    """Shrink the moving window from 31 x 31 to 3 x 3."""
    monkeypatch.setattr(starfm, "windowSize", 3)
    monkeypatch.setattr(starfm, "mid_idx", 4)
    monkeypatch.setattr(starfm, "padAmount", 1)


def centre_window_distances():
    """Differences and distances for the window around the centre pixel."""
    spec_diff, spec_dist = starfm.spectral_distance(F0.ravel(), C0.ravel())
    temp_diff, temp_dist = starfm.temporal_distance(C0.ravel(), C1.ravel())
    return spec_diff, spec_dist, temp_diff, temp_dist


def predict_band(mask):
    return starfm.predictionPerBand(
        starfm.padImage(F0), starfm.padImage(C0), starfm.padImage(C1), C1, mask
    )


def step_image(rows, cols, start, high=200):
    """Three identical uint8 bands that jump from 0 to high at column start."""
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    image[:, start:] = high
    return image


# Distances


def test_spectral_distance():
    fine = np.array([5.0, 3.0, 1.0])
    coarse = np.array([1.0, 3.0, 4.0])

    difference, distance = starfm.spectral_distance(fine, coarse)

    # Fine minus coarse, then 1 / (|difference| + 1). Equal pixels score 1 and
    # the score falls as the two images disagree more.
    np.testing.assert_array_equal(difference, [4.0, 0.0, -3.0])
    np.testing.assert_allclose(distance, [1 / 5, 1.0, 1 / 4])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="unsigned input wraps around, see Known limitations in the README",
)
def test_spectral_distance_of_unsigned_input_is_signed():
    fine = np.array([1], dtype=np.uint8)
    coarse = np.array([4], dtype=np.uint8)

    difference, _ = starfm.spectral_distance(fine, coarse)

    # 1 - 4 should be -3. In uint8 arithmetic it comes out as 253.
    assert difference[0] == -3


def test_temporal_distance():
    coarse_t0 = np.array([1.0, 3.0, 4.0])
    coarse_t1 = np.array([3.0, 3.0, 0.0])

    difference, distance = starfm.temporal_distance(coarse_t0, coarse_t1)

    # Prediction date minus base date, then 1 / (|difference| + 1).
    np.testing.assert_array_equal(difference, [2.0, 0.0, -4.0])
    np.testing.assert_allclose(distance, [1 / 3, 1.0, 1 / 5])


def test_spatial_distance_in_a_3x3_window(window3):
    # The argument is not used: the result depends on the window size only.
    distance = starfm.spatial_distance(F0.ravel())

    expected = [
        [CORNER, NEIGHBOUR, CORNER],
        [NEIGHBOUR, 1.0, NEIGHBOUR],
        [CORNER, NEIGHBOUR, CORNER],
    ]
    np.testing.assert_allclose(distance, np.ravel(expected))


def test_spatial_distance_in_the_default_window():
    distance = starfm.spatial_distance(np.zeros(31 * 31))

    assert distance.shape == (31 * 31,)
    grid = distance.reshape(31, 31)
    # The centre of a 31 x 31 window is row 15, column 15.
    assert grid[15, 15] == 1.0
    assert distance.argmax() == starfm.mid_idx == 480
    # One pixel away, and in the corner, 15 rows and 15 columns away.
    assert grid[15, 14] == pytest.approx(150 / 151)
    assert grid[0, 0] == pytest.approx(1 / (1 + 15 * np.sqrt(2) / 150))
    # The same in every direction.
    np.testing.assert_array_equal(grid, grid.T)
    np.testing.assert_array_equal(grid, grid[::-1, ::-1])


def test_combination_distance_is_the_product():
    spec_dist = np.array([1.0, 0.5])
    temp_dist = np.array([0.5, 0.25])
    spat_dist = np.array([1.0, 0.8])

    combined = starfm.combination_distance(spec_dist, temp_dist, spat_dist)

    np.testing.assert_allclose(combined, [0.5, 0.1])


def test_combination_distance_with_log_weight(monkeypatch):
    monkeypatch.setattr(starfm, "logWeight", True)

    combined = starfm.combination_distance(
        np.array([1.0]), np.array([1.0]), np.array([0.5])
    )

    # log(1 + 1) for the spectral and for the temporal distance, times 0.5.
    np.testing.assert_allclose(combined, [np.log(2) ** 2 * 0.5])


# Similar pixels


def test_similarity_threshold():
    # 1 to 9 have mean 5 and squared deviations 16, 9, 4, 1, 0, 1, 4, 9, 16,
    # which add up to 60. The standard deviation is sqrt(60 / 9) = 2.58, and
    # the threshold is twice that divided by numberClass (4): 1.29.
    threshold = starfm.similarity_threshold(F0.ravel())

    assert threshold == pytest.approx(np.sqrt(60 / 9) / 2)


def test_similarity_threshold_leaves_out_zeros():
    # Zeros stand for the padding outside the image. Without them the values
    # are 2, 3, 5, 6, 8, 9: mean 5.5, squared deviations 12.25, 6.25, 0.25,
    # 0.25, 6.25, 12.25, sum 37.5, standard deviation sqrt(37.5 / 6) = 2.5.
    window = np.array([2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0])

    assert starfm.similarity_threshold(window) == pytest.approx(1.25)


def test_similarity_pixels(window3):
    similar = starfm.similarity_pixels(F0.ravel())

    # The centre value is 5 and the threshold is 1.29, so 4, 5 and 6 count.
    np.testing.assert_array_equal(similar, [0, 0, 0, 1, 1, 1, 0, 0, 0])


def test_filtering_drops_similar_pixels_that_fit_worse_than_the_centre(window3):
    spec_diff, spec_dist, temp_diff, temp_dist = centre_window_distances()

    kept = starfm.filtering(F0.ravel(), spec_dist, temp_dist, spec_diff, temp_diff)

    # The similar pixels 4, 5 and 6 have spectral differences 3, 4 and 5. A
    # pixel passes if its difference is below the centre's 4 plus the sensor
    # uncertainty of 0.042, so the 6 is dropped.
    np.testing.assert_array_equal(kept, [0, 0, 0, 1, 1, 0, 0, 0, 0])


def test_filtering_with_the_temporal_test_switched_on(window3, monkeypatch):
    # As C1, but the pixel left of the centre changes by 2 instead of 0, and
    # the pixel right of it by 1 instead of 2.
    coarse_t1 = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 2.0, 1.0, 2.0, 3.0])
    spec_diff, spec_dist = starfm.spectral_distance(F0.ravel(), C0.ravel())
    temp_diff, temp_dist = starfm.temporal_distance(C0.ravel(), coarse_t1)
    arguments = (F0.ravel(), spec_dist, temp_dist, spec_diff, temp_diff)

    without_temporal_test = starfm.filtering(*arguments)
    monkeypatch.setattr(starfm, "temp", True)
    with_temporal_test = starfm.filtering(*arguments)

    # The centre changes by 1 between the dates. With the switch on, a pixel
    # also has to change by less than 1 plus the uncertainty of 0.042. That
    # drops the pixel on the left. The pixel on the right passes it and is
    # dropped by the spectral test alone, so the result needs both tests.
    np.testing.assert_array_equal(without_temporal_test, [0, 0, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(with_temporal_test, [0, 0, 0, 0, 1, 0, 0, 0, 0])


# Weights


def test_weighting_shares_the_weight_between_the_filtered_pixels(window3):
    _, spec_dist, _, temp_dist = centre_window_distances()
    comb_dist = starfm.combination_distance(
        spec_dist, temp_dist, starfm.spatial_distance(F0.ravel())
    )
    kept = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])

    weights = starfm.weighting(spec_dist, temp_dist, comb_dist, kept)

    # Left of the centre: spectral 1/4, temporal 1, spatial NEIGHBOUR.
    # The centre itself: spectral 1/5, temporal 1/2, spatial 1.
    left = NEIGHBOUR / 4
    centre = 1 / 10
    expected = np.zeros(9)
    expected[3] = left / (left + centre)
    expected[4] = centre / (left + centre)
    np.testing.assert_allclose(weights, expected)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("zero_difference", ["spectral", "temporal"])
def test_weighting_gives_all_weight_to_a_centre_with_zero_difference(
    window3, zero_difference
):
    # A distance of exactly 1 means a difference of 0.
    spec_dist = np.full(9, 0.5)
    temp_dist = np.full(9, 0.25)
    if zero_difference == "spectral":
        spec_dist[4] = 1.0
    else:
        temp_dist[4] = 1.0

    weights = starfm.weighting(spec_dist, temp_dist, spec_dist * temp_dist, np.ones(9))

    np.testing.assert_array_equal(weights, [0, 0, 0, 0, 1, 0, 0, 0, 0])


# Prediction


def test_pad_image_adds_half_a_window_of_zeros(window3):
    padded = starfm.padImage(np.array([[1, 2], [3, 4]]))

    expected = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
    np.testing.assert_array_equal(padded, expected)


def test_prediction_per_band_on_the_worked_example(window3):
    predicted = predict_band(np.ones((3, 3)))

    # Centre pixel: the two weights from the weighting test above, applied to
    # F0 + (C1 - C0), which is 4 left of the centre and 6 at the centre.
    left, centre = NEIGHBOUR / 4, 1 / 10
    middle = (4 * left + 6 * centre) / (left + centre)
    # The pixel right of it has the value 6, and its window hangs over the
    # edge of the image. Two pixels pass the filter: the 5 (spectral 1/5,
    # temporal 1/2, spatial NEIGHBOUR) and the 6 itself (spectral 1/6,
    # temporal 1/3, spatial 1). F0 + (C1 - C0) is 6 and 8 for them.
    left, centre = NEIGHBOUR / 10, 1 / 18
    right = (6 * left + 8 * centre) / (left + centre)
    # Every other pixel keeps all of the weight, because it has no similar
    # neighbour or because its spectral or temporal difference is zero, and
    # comes out as its own F0 + (C1 - C0).
    expected = [[1.0, 3.0, 5.0], [4.0, middle, right], [7.0, 9.0, 11.0]]
    np.testing.assert_allclose(predicted, expected)
    # The two values as the notebook prints them.
    assert middle == pytest.approx(4.57414449)
    assert right == pytest.approx(6.71733967)


def test_prediction_per_band_keeps_the_coarse_value_outside_the_mask(window3):
    # One pixel of the mask is set, and it is off the diagonal, so a mask read
    # with rows and columns swapped would predict a different pixel.
    right_of_centre = np.zeros((3, 3))
    right_of_centre[1, 2] = 1

    predicted = predict_band(right_of_centre)

    np.testing.assert_array_equal(predict_band(np.zeros((3, 3))), C1)
    changed = predicted != C1
    assert changed[1, 2]
    assert changed.sum() == 1


def test_prediction_treats_each_band_on_its_own(window3):
    # Band 0 is the worked example. In band 1 the coarse image does not change
    # between the dates, so the prediction is the fine image. In band 2 the
    # fine and coarse images agree on the base date, so the prediction is the
    # coarse image of the prediction date. No two bands of an input are the
    # same, so a band taken from the wrong place changes the result.
    twos = np.full((3, 3), 2.0)
    threes = np.full((3, 3), 3.0)
    fine_t0 = np.dstack([F0, 10 * F0, C1]).astype(np.float32)
    coarse_t0 = np.dstack([C0, twos, C1]).astype(np.float32)
    coarse_t1 = np.dstack([C1, twos, threes]).astype(np.float32)
    # The image is smaller than the 15 x 15 box that widens the edges, so one
    # edge pixel is enough to put all nine pixels in the mask.
    assert starfm.sobel_edge_detection(fine_t0).all()

    predicted = starfm.prediction(fine_t0, coarse_t0, coarse_t1)

    assert predicted.shape == (3, 3, 3)
    np.testing.assert_allclose(
        predicted[:, :, 0], predict_band(np.ones((3, 3))), rtol=1e-6
    )
    np.testing.assert_array_equal(predicted[:, :, 1], 10 * F0)
    np.testing.assert_array_equal(predicted[:, :, 2], threes)


def test_prediction_of_a_single_band(window3):
    one_band = [image[:, :, np.newaxis] for image in (F0, C0, C1)]

    predicted = starfm.prediction(*one_band)

    assert predicted.shape == (3, 3, 1)
    np.testing.assert_allclose(predicted[:, :, 0], predict_band(np.ones((3, 3))))


# Edge mask


def test_edge_mask_reaches_seven_pixels_either_side_of_a_step():
    mask = starfm.sobel_edge_detection(step_image(40, 40, start=20))

    # The 3 x 3 Sobel kernel responds in the two columns that touch the jump,
    # 19 and 20. The 15 x 15 box of ones then reaches 7 pixels further each
    # way, so the mask is columns 12 to 27 of every row.
    expected = np.zeros((40, 40))
    expected[:, 12:28] = 1
    np.testing.assert_array_equal(mask, expected)


def test_edge_mask_works_the_same_along_rows():
    upright = step_image(40, 40, start=20)
    on_its_side = np.ascontiguousarray(upright.transpose(1, 0, 2))

    mask = starfm.sobel_edge_detection(on_its_side)

    np.testing.assert_array_equal(mask, starfm.sobel_edge_detection(upright).T)


def test_edge_mask_stops_at_the_image_border():
    mask = starfm.sobel_edge_detection(step_image(40, 40, start=3))

    # Edge columns 2 and 3, widened by 7: columns 0 to 10.
    expected = np.zeros((40, 40))
    expected[:, :11] = 1
    np.testing.assert_array_equal(mask, expected)


def test_edge_mask_is_empty_for_a_flat_image():
    flat = np.full((40, 40, 3), 7, dtype=np.uint8)

    assert not starfm.sobel_edge_detection(flat).any()


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64"])
def test_edge_mask_takes_a_single_band_as_it_is(dtype):
    three_bands = step_image(40, 40, start=20)
    one_band = three_bands[:, :, :1].astype(dtype)
    expected = starfm.sobel_edge_detection(three_bands)

    # One band is already a grayscale image, with or without the third axis.
    np.testing.assert_array_equal(starfm.sobel_edge_detection(one_band), expected)
    np.testing.assert_array_equal(
        starfm.sobel_edge_detection(one_band[:, :, 0]), expected
    )


def test_edge_mask_drops_a_weak_edge_next_to_a_strong_one():
    # A jump of 200 at column 20 and a jump of 10 at column 45. Scaled to
    # 0 to 255, the gradient is 255 in the two columns of the strong edge, 13
    # in the two columns of the weak one and 0 elsewhere. Otsu's method picks
    # the split with the larger between-class variance. Splitting below 13
    # gives 2240/2400 * 160/2400 * 134^2 = 1117. Splitting above 13 gives
    # 2320/2400 * 80/2400 * 254.6^2 = 2088, so the weak edge is left out.
    image = step_image(40, 60, start=20)
    image[:, 45:] = 210

    mask = starfm.sobel_edge_detection(image)

    expected = np.zeros((40, 60))
    expected[:, 12:28] = 1
    np.testing.assert_array_equal(mask, expected)
