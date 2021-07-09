"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import os
import pickle
import tempfile
import warnings
from typing import Any, Callable

import cv2
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
import shapely
import skimage.graph
import skimage.transform

import xdem.version

TEMP_DIR = tempfile.TemporaryDirectory()


def generate_random_field(shape: tuple[int, int], corr_size: int) -> np.ndarray:
    """
    Generate a semi-random gaussian field (to simulate a DEM or DEM error)

    :param shape: The output shape of the field.
    :param corr_size: The correlation size of the field.

    :examples:
        >>> np.random.seed(1)
        >>> generate_random_field((4, 5), corr_size=2).round(2)
        array([[0.47, 0.5 , 0.56, 0.63, 0.65],
               [0.49, 0.51, 0.56, 0.62, 0.64],
               [0.56, 0.56, 0.57, 0.59, 0.59],
               [0.57, 0.57, 0.57, 0.58, 0.58]])

    :returns: A numpy array of semi-random values from 0 to 1
    """
    field = cv2.resize(
        cv2.GaussianBlur(
            np.repeat(
                np.repeat(
                    np.random.randint(0, 255, (shape[0] // corr_size, shape[1] // corr_size), dtype="uint8"),
                    corr_size,
                    axis=0,
                ),
                corr_size,
                axis=1,
            ),
            ksize=(2 * corr_size + 1, 2 * corr_size + 1),
            sigmaX=corr_size,
        )
        / 255,
        dsize=(shape[1], shape[0]),
    )
    return field


def deprecate(removal_version: str | None = None, details: str | None = None):
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed.
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """

    def deprecator_func(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or removal_version > xdem.version.version

            # Add text depending on the given arguments and 'should_warn'.
            text = (
                f"Call to deprecated function '{func.__name__}'."
                if should_warn
                else f"Deprecated function '{func.__name__}' was removed in {removal_version}."
            )

            # Add the details explanation if it was given, and make sure the sentence is ended.
            if details is not None:
                details_frm = details.strip()
                if details_frm[0].islower():
                    details_frm = details_frm[0].upper() + details_frm[1:]

                text += " " + details_frm

                if not any(text.endswith(c) for c in ".!?"):
                    text += "."

            if should_warn and removal_version is not None:
                text += f" This functionality will be removed in version {removal_version}."
            elif not should_warn:
                text += f" Current version: {xdem.version.version}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func


def cache(func):

    @functools.wraps(func)
    def inner_func(*args, **kwargs):

        should_cache = kwargs.get("cache") in [True, None]
        if "random_seed" in kwargs and kwargs["random_seed"] is None:
            should_cache = False

        if should_cache:
            cache_name = os.path.join(TEMP_DIR.name, hashlib.md5("".join(map(lambda s: str(pickle.dumps(s)), list(args) + list(kwargs.values()))).encode()).hexdigest() + ".pkl")

            if os.path.isfile(cache_name):
                with open(cache_name, "rb") as infile:
                    return pickle.load(infile)

        result = func(*args, **kwargs)

        if should_cache:
            with open(cache_name, "wb") as outfile:
                pickle.dump(result, outfile)

        return result

    return inner_func

@cache
def synthesize_glacier(
    border: int = 1,
    curviness: float = 1.0,
    size_scale: float = 1.0,
    gradient_coeffs: list[float] = [100.0, 0],
    error_size: int = 2,
    error_magnitude: float = 1.0,
    random_seed: int | None = 42,
    cache: bool = True,
) -> np.ma.masked_array:
    if random_seed is None:
        random_seed = np.random.randint(0, np.iinfo(np.int32).max)

    if cache:
        pass

    for i in range(10):
        np.random.seed(random_seed + i)
        # Generate a spatially correlated random field.
        # Threshold the field to generate random "blobs"
        random_blobs = generate_random_field((int(500 * size_scale), int(500 * size_scale)), corr_size=20) < 0.6

        # Remove all "blobs" that intersect the border (only keep the closed ones)
        labels = scipy.ndimage.measurements.label((~random_blobs) & scipy.ndimage.binary_fill_holes(random_blobs))[0]

        u_labels, u_counts = np.unique(labels[labels != 0], return_counts=True)

        mask = labels == u_labels[np.argwhere(u_counts == u_counts.max())][0][0]

        # In some cases, the generated blob is huge.
        if (np.count_nonzero(mask) / mask.size) > 0.3:
            continue

        (ymin, xmin), (ymax, xmax) = np.argwhere(mask).min(axis=0), np.argwhere(mask).max(axis=0)
        mask = mask[ymin - border : ymax + border + 1, xmin - border : xmax + border + 1]

        xcoords, ycoords = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))

        coords = np.dstack([xcoords, ycoords])

        xcoords_inside, ycoords_inside = xcoords[mask], ycoords[mask]
        xcoords_outside, ycoords_outside = xcoords[~mask], ycoords[~mask]

        larger_mask = scipy.ndimage.binary_dilation(mask, iterations=2)

        distance_arr = np.zeros(mask.shape, dtype="float32")
        distance_arr[coords[:, :, 1][mask], coords[:, :, 0][mask]] = scipy.spatial.distance.cdist(
            coords.reshape(-1, 2)[mask.ravel()], coords.reshape(-1, 2)[~mask.ravel() & larger_mask.ravel()]
        ).min(axis=1)

        largest_distance = np.argwhere(distance_arr == distance_arr.max())[0]
        cost_raster = np.where(mask, 1 / (distance_arr + 0.1), 99999)
        smaller_mask = scipy.ndimage.binary_erosion(mask, iterations=2)

        longest_line = shapely.geometry.LineString()
        route_lengths = np.empty((0,), dtype=int)
        descending_order = np.argsort(
            1 - np.linalg.norm(coords.reshape(-1, 2)[smaller_mask.ravel()][:, ::-1] - largest_distance, axis=1)
        )
        for coord in coords.reshape(-1, 2)[smaller_mask.ravel()][descending_order, ::-1]:
            route = np.array(skimage.graph.route_through_array(cost_raster, largest_distance, coord)[0])

            route_lengths = np.append(route_lengths, len(route))

            # If no improvement is seen within the last 10 iterations, stop looking
            if len(route_lengths) > 10 and route_lengths[-10:-5].max() >= route_lengths[-5:-1].max():
                break

            if len(route) > len(longest_line.coords):
                longest_line = shapely.geometry.LineString(route[::-1, [1, 0]])

        splitter = int(len(longest_line.coords) * 0.9)
        extended_line = np.r_[
            longest_line.coords[:splitter],
            shapely.affinity.scale(
                shapely.geometry.LineString(longest_line.coords[splitter:]),
                xfact=10,
                yfact=10,
                origin=longest_line.coords[splitter],
            ).coords,
        ]

        heights = (
            scipy.ndimage.uniform_filter(
                scipy.interpolate.griddata(
                    points=extended_line,
                    values=np.arange(extended_line.shape[0]),
                    xi=(xcoords, ycoords),
                    method="nearest",
                ),
                10,
            )
            / extended_line.shape[0]
        )

        curvature = (2 * heights - 1) * scipy.ndimage.filters.gaussian_filter(distance_arr, 3) * curviness

        dem = np.ma.masked_array(np.poly1d(gradient_coeffs)(heights) - curvature, mask=~mask)

        if error_magnitude > 0.0:
            dem += (generate_random_field(dem.shape, corr_size=error_size) - 0.5) * error_magnitude * dem.std() / 10

        break
    else:
        raise ValueError("This should not be possible. Please raise an issue")

    return dem


@cache
def synthesize_glacier_region(
    shape: tuple[int, int] = (2000, 2000),
    n_glaciers: int = 50,
    random_seed: int | None = None,
    cache: bool = True
):

    from xdem.spatial_tools import subdivide_array
    if random_seed is None:
        random_seed = np.random.randint(0, np.iinfo(np.int32).max)

    if cache:
        pass

    output = np.ma.masked_array(np.empty(shape=shape, dtype="float32"), mask=np.ones(shape=shape, dtype="bool"))

    subdivision = subdivide_array(shape, n_glaciers)

    for i in range(n_glaciers):

        dem = synthesize_glacier(random_seed=random_seed + i, curviness=abs(np.random.randn()), size_scale=1 + abs(np.random.randn()))

        subdivided_part = subdivision == i
        (ymin, xmin), (ymax, xmax) = np.argwhere(subdivided_part).min(axis=0), np.argwhere(subdivided_part).max(axis=0)

        max_height = ymax - ymin
        max_width = xmax - xmin

        if dem.shape[0] > max_height or dem.shape[1] > max_width:
            dem_resized = skimage.transform.resize(dem.data, output_shape=(np.clip(dem.shape[0], 1, max_height), np.clip(dem.shape[1], 1, max_width)), order=1, preserve_range=True)
            mask_resized = skimage.transform.resize(dem.data, output_shape=(np.clip(dem.shape[0], 1, max_height), np.clip(dem.shape[1], 1, max_width)), order=0, preserve_range=True)

            dem = np.ma.masked_array(dem_resized, mask=mask_resized.astype(bool))

        output[ymin: ymin + dem.shape[0], xmin: xmin + dem.shape[1]] = dem

    return output

