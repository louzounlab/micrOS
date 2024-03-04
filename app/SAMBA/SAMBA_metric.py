import os
import umap
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sewar.full_ref import mse, sam
import matplotlib.pyplot as plt


def load_img(folder_path, tag=None):
    """
    Reads all the images saved in a certain folder path and in the tag file
    :param folder_path: Path of the folder where the images from micro2matrix are saved (str)
    :param Default is None, but if we want to work only on images which we have theri tag
            a tag dataframe or series should be passed too (pandas)
    :return: (1) final array with all the loaded images from the folder (list)
             (2) names list with the loaded images names (list)
    """
    arrays = []
    names = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            if file == "bact_names.npy":
                continue
            file_path = os.path.join(folder_path, file)
            if tag is None:
                arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))
            else:
                if file_path.split("\\")[-1].replace(".npy", "") in tag.index:
                    arrays.append(np.load(file_path, allow_pickle=True, mmap_mode='r'))

            names.append(file_path.split("\\")[-1].replace(".npy", ""))

    final_array = np.stack(arrays, axis=0)
    return final_array, names


def fft_process(x, cutoff):
    """
    Apply FFT on each images with the cutoff given.
    :param x: A single image (ndarray)
    :param cutoff: Cutoff frequency as a fraction of the maximum possible frequency (float)
    :return: A filtered image (ndarray)
    """
    fft = np.fft.fft2(x)

    # Shift the zero-frequency component to the center of the array
    fft_shifted = np.fft.fftshift(fft)

    # Define the cutoff frequency (as a fraction of the maximum possible frequency)
    cutoff_freq = cutoff

    # Create a mask to keep only the low-frequency components
    rows, cols = fft_shifted.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
    ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = True

    # Apply the mask to the FFT
    fft_cutoff = np.copy(fft_shifted)
    fft_cutoff[~mask] = 0

    # Inverse FFT to get the filtered image
    img_filtered_x = np.fft.ifft2(np.fft.ifftshift(fft_cutoff))
    return img_filtered_x


def final_distance(output1, output2, f):
    """
    Calculate the distance between 2 filtered images.
    :param output1: Filtered image 1 (ndarray)
    :param output2: Filtered image 2 (ndarray)
    :param f: Metric to calculate the distance according to
              One of  "d1","d2","d3","sam","mse"
    :return:
    """
    if f == "d1":
        # Euclidean distance
        return np.linalg.norm(output1 - output2)
    elif f == "d2":
        # Absolute difference
        return np.sum(np.abs(np.abs(output1) - np.abs(output2)))
    elif f == "d3":
        # Difference of angles
        return np.sum(np.abs(np.angle(output1) - np.angle(output2)))
    elif f == "sam":
        return sam(output1, output2)
    elif f == "mse":
        return mse(output1, output2)


def build_SAMBA_distance_matrix(folder_path, metric="sam", cutoff=0.8, tag=None):
    """
    Build SAMBA distance matrix of the FFT processed images using the metric as the
    final distance metric between the processed images, and the cutoff as the FFT cutoff
    :param folder_path: Path of the folder where the images from micro2matrix are saved (str)
    :param metric: Metric to calculate the distance according to.
                   One of  "d1","d2","d3","sam","mse"
    :param cutoff: Cutoff frequency as a fraction of the maximum possible frequency (float)
    :param tag: Default is None, but if we want to work only on images which we have theri tag
                a tag dataframe or series should be passed too (pandas)
    :return: Distance matrix dataframe (pandas)
    """
    # Load images from the folder
    imgs, names = load_img(folder_path, tag)

    # Image shape
    x_axis = imgs.shape[-1]
    y_axis = imgs.shape[-2]

    # Function for images adjusting (FFT) and calculating the pairwise distance
    def fft_dist(x, y):
        x = x.reshape(x_axis, y_axis)
        y = y.reshape(x_axis, y_axis)
        x_after = fft_process(x, cutoff)
        y_after = fft_process(y, cutoff)
        return final_distance(x_after, y_after, metric)

    # Build the SAMBA distance matrix
    dm = cdist(imgs.reshape(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1), metric=fft_dist)

    if tag is None:
        dm = pd.DataFrame(dm, index=names, columns=names)
    else:
        dm = pd.DataFrame(dm, index=tag.index, columns=tag.index)

    return dm


def plot_umap(dm, tag, save):
    umap_embedding = umap.UMAP(metric='precomputed').fit_transform(dm)
    umap_embedding_df = pd.DataFrame(data=umap_embedding, index=tag.index, columns=["PCA1", "PCA2"])
    tag0 = tag[tag.values == 0]
    tag1 = tag[tag.values == 1]

    umap_embedding_df0 = umap_embedding_df.loc[tag0.index]
    umap_embedding_df1 = umap_embedding_df.loc[tag1.index]

    plt.scatter(umap_embedding_df0["PCA1"], umap_embedding_df0["PCA2"], color="red", label="Control")
    plt.scatter(umap_embedding_df1["PCA1"], umap_embedding_df1["PCA2"], color="blue", label="Condition")
    plt.xlabel("UMAP dim 1", fontdict={"fontsize": 15})
    plt.ylabel("UMAP dim 2", fontdict={"fontsize": 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save}/umap_plot.png")
