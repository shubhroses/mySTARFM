import cv2
import rasterio
from rasterio.transform import from_origin
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from parameters import (
    windowSize,
    path,
    logWeight,
    temp,
    spatImp,
    numberClass,
    uncertaintyFineRes,
    uncertaintyCoarseRes,
    mid_idx,
    specUncertainty,
    tempUncertainty,
    sizeSlices,
    padAmount,
)

sys.path.append("src")


def spectral_distance(F0_window, C0_window):
    spectral_difference_window = F0_window - C0_window
    spectral_distance_window = 1 / (abs(spectral_difference_window) + 1.0)
    return spectral_difference_window, spectral_distance_window


def temporal_distance(C0_window, C1_window):
    temp_difference_window = C1_window - C0_window
    temp_distance_window = 1 / (abs(temp_difference_window) + 1.0)
    return temp_difference_window, temp_distance_window


def spatial_distance(F0_window):
    coord = np.sqrt((np.mgrid[0:windowSize, 0:windowSize] - windowSize // 2) ** 2)
    spat_dist = np.sqrt(((0 - coord[0]) ** 2 + (0 - coord[1]) ** 2))
    rel_spat_dist = spat_dist / spatImp + 1.0  # relative spatial distance
    rev_spat_dist = 1 / rel_spat_dist  # relative spatial distance reversed
    flat_spat_dist = np.ravel(rev_spat_dist)
    return flat_spat_dist


def combination_distance(
    spectral_distance_window, temporal_distance_window, spatial_distance_window
):
    if logWeight == True:
        spectral_distance_window = np.log(spectral_distance_window + 1)
        temporal_distance_window = np.log(temporal_distance_window + 1)
    combination_distance_window = (
        spectral_distance_window * temporal_distance_window * spatial_distance_window
    )
    return combination_distance_window


def similarity_threshold(F0_window):
    F0_window = np.where(F0_window == 0, np.nan, F0_window)

    st_dev = np.nanstd(F0_window, axis=0)

    sim_threshold = st_dev * 2 / numberClass

    return sim_threshold


def similarity_pixels(F0_window):
    sim_threshold = similarity_threshold(F0_window)
    similar_pixels = np.where(
        abs(F0_window - F0_window[mid_idx]) <= sim_threshold, 1, 0
    )

    return similar_pixels


def filtering(
    F0_window,
    spectral_distance_window,
    temporal_distance_window,
    spectral_difference_window,
    temporal_difference_window,
):
    similar_pixels = similarity_pixels(F0_window)

    max_spec_dist = abs(spectral_difference_window)[mid_idx] + specUncertainty + 1
    max_temp_dist = abs(temporal_difference_window)[mid_idx] + tempUncertainty + 1

    spec_filter = np.where(spectral_distance_window > 1.0 / max_spec_dist, 1, 0)

    st_filter = spec_filter

    if temp == True:
        temp_filter = np.where(temporal_distance_window > 1.0 / max_temp_dist, 1, 0)
        st_filter = spec_filter * temp_filter

    similar_pixels = similar_pixels * st_filter
    return similar_pixels


def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered):
    # Assign max weight (1) when the temporal or spectral distance is zero

    zero_spec_dist = np.where(spec_dist[mid_idx] == 1, 1, 0)

    zero_temp_dist = np.where(temp_dist[mid_idx] == 1, 1, 0)

    zero_dist_mid = np.where((zero_spec_dist == 1), zero_spec_dist, zero_temp_dist)

    shape = np.subtract(spec_dist.shape, (0, 1))

    zero_dist = np.zeros(shape[1])

    zero_dist = np.insert(zero_dist, [mid_idx], zero_dist_mid, axis=0)

    weights = np.where((np.sum(zero_dist) == 1), zero_dist, comb_dist)

    # Calculate weights only for the filtered spectrally similar pixels
    weights_filt = weights * similar_pixels_filtered

    # Normalize weights
    norm_weights = weights_filt / (np.sum(weights_filt))
    # print ("Done weighting!", norm_weights)

    return norm_weights

def getImportantPixels(padded_F0_test, padded_C0_test, padded_C1_test, edge_image):
    F0_important_pixels = {}
    for i in range(padAmount, len(padded_F0_test) - padAmount):
        for j in range(padAmount, len(padded_F0_test[0]) - padAmount):
            if edge_image[i-padAmount][j-padAmount] != 0:
                F0_important_pixels[(i, j)] = padded_F0_test[
                    i - windowSize // 2 : i + windowSize // 2 + 1,
                    j - windowSize // 2 : j + windowSize // 2 + 1,
                ].flatten()
    return F0_important_pixels

def padImage(image):
    padded_image = np.pad(
        image, pad_width=windowSize // 2, mode="constant", constant_values=0
    )
    return padded_image

def predictionPerBand(padded_F0_test, padded_C0_test, padded_C1_test, C1_test, edge_image):
    F1_test = C1_test.copy()

    F0_important_pixels = getImportantPixels(padded_F0_test, padded_C0_test, padded_C1_test, edge_image)

    for i, j in F0_important_pixels.keys():
        F0_window = F0_important_pixels[(i, j)]
        C0_window = padded_C0_test[
            i - windowSize // 2 : i + windowSize // 2 + 1,
            j - windowSize // 2 : j + windowSize // 2 + 1,
        ].flatten()
        C1_window = padded_C1_test[
            i - windowSize // 2 : i + windowSize // 2 + 1,
            j - windowSize // 2 : j + windowSize // 2 + 1,
        ].flatten()

        # Spectral Difference/Distance
        spectral_difference_window, spectral_distance_window = spectral_distance(
            F0_window, C0_window
        )

        # Temporal Difference/Distance
        temporal_difference_window, temporal_distance_window = temporal_distance(
            C0_window, C1_window
        )

        # Spatial Distance
        spatial_distance_window = spatial_distance(F0_window)

        # Combination Distance
        combination_distance_window = combination_distance(
            spectral_distance_window, temporal_distance_window, spatial_distance_window
        )

        similar_pixels_window = filtering(
            F0_window,
            spectral_distance_window,
            temporal_distance_window,
            spectral_difference_window,
            temporal_difference_window,
        )

        weights = weighting(
            spectral_distance_window,
            temporal_distance_window,
            combination_distance_window,
            similar_pixels_window,
        )

        pred_refl = F0_window + temporal_difference_window
        weighted_pred_refl = np.sum(pred_refl * weights)

        F1_test[i - padAmount][j - padAmount] = weighted_pred_refl
    return F1_test

def sobel_edge_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    thresh = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # define a 5x5 kernel of ones
    kernel = np.ones((15, 15))

    # perform the convolution operation with the kernel
    output = convolve2d(thresh, kernel, mode='same')

    # threshold the output to set any non-zero values to 1
    output[output > 0] = 1
    return output


def prediction(F0, C0, C1):
    x, y, z = F0.shape
    F0_bands = np.split(F0, z, axis=2)
    C0_bands = np.split(C0, z, axis=2)
    C1_bands = np.split(C1, z, axis=2)

    edge_image = sobel_edge_detection(F0)
    # edge_image = np.ones((150, 150))
    # plt.imshow(edge_image, cmap='gray')
    # plt.show()

    F1 = None

    for i in range(z):
        cur_F0 = np.squeeze(F0_bands[i])
        cur_C0 = np.squeeze(C0_bands[i])
        cur_C1 = np.squeeze(C1_bands[i])

        padded_cur_F0 = padImage(cur_F0)
        padded_cur_C0 = padImage(cur_C0)
        padded_cur_C1 = padImage(cur_C1)

        newBand = predictionPerBand(padded_cur_F0, padded_cur_C0, padded_cur_C1, cur_C1, edge_image)[:, :, np.newaxis]
        if i == 0:
            F1 = newBand
        else:
            F1 = np.concatenate((F1, newBand), axis=2)
    return F1

def saveImage(F1):
    output_file = "results/output.tif"
    xmin, ymax = 0, 0
    xres, yres = 1, 1
    nrows, ncols, nbands = F1.shape
    transform = from_origin(xmin, ymax, xres, yres)

    with rasterio.open(output_file, 'w', driver='GTiff', height=nrows, width=ncols, count=nbands, dtype=F1.dtype, transform=transform) as dst:
        # Write the NumPy array to the file
        for i in range(nbands):
            dst.write(F1[:,:,i], indexes=i+1)


if __name__ == "__main__":
    # F0_test = np.array(
    #     [
    #         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    #         [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]],
    #         [[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]],
    #     ]
    # )
    # C0_test = np.array(
    #     [
    #         [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    #         [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    #         [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    #     ]
    # )
    # C1_test = np.array(
    #     [
    #         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    #         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    #         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    #     ]
    # )

    # F1_test = prediction(F0_test, C0_test, C1_test)

    # print("F0_test shape:", F0_test.shape)
    # print("F1_test shape:", F1_test.shape)

    F0 = cv2.imread("Images/sim_Landsat_t1.tif")
    C0 = cv2.imread("Images/sim_MODIS_t1.tif")
    C1 = cv2.imread("Images/sim_MODIS_t2.tif")

    F1 = prediction(F0, C0, C1)

    print("F0 shape:", F0.shape)
    print("F1 shape:", F1.shape)

    plt.imshow(F1, cmap='gray')
    plt.show()

    # saveImage(F1)