from itertools import product

import pandas as pd
import numpy as np

from nilearn.image import math_img, iter_img

def tidy_data(array, labels, depvar=None, columns=None):
    """ Transorm multidimensional array into tidy-style dataframe. 
    
    It is assumed that values of the array represent dependent variable and each
    array dimension represents categorical independent variable with arbitrary 
    number of levels. 
    
    For example if we have three dependent variables with two, three and five 
    levels, array will have shape 3x2x5. Function will then transform this array
    into table with shape 30x4. First dimension size is equal to the total 
    number of observation, whereas second dimension size is equal to the total
    number of variables (dependent and independent). 
    
    Args:
        array (np.array):
            Array of dependent variables. Can have any dimension.
        labels (list of lists of str):
            Labels for independent variables. Lables have to correspond to array
            dimensions.
        depvar (str, optional):
            Name of dependent variable.
        columns (list of str, optional):
            Name of independent variables for each array dimension.
    
    Returns:
        Pandas DataFrame in tidy format.
    """
    if array.shape != tuple(len(lst) for lst in labels):
        raise ValueError("column names are not matching array dimensions: " + 
                         f"{array.shape} != {tuple(len(l) for l in labels)}")
    if depvar is not None and not isinstance(depvar, str):
        raise TypeError("depvar should be a string")
    if columns is not None and not isinstance(columns, list):
        raise TypeError("columns should be a list")
    if (columns is not None and 
        (not all(isinstance(c, str) for c in columns) or 
        len(columns) != array.ndim)):
        raise ValueError("columns should only contain strings and have lenght" +
                         " equal to the number of array dimension")
    
    # Used to retreive values from array
    keys = [
        {label: i for i, label in enumerate(l)} 
        for l in labels
    ]
    if depvar is None:
        depvar = "depvar"
    if columns is None:
        columns = [f"col_{cix}" for cix in range(array.ndim)]
    
    tidy_df = pd.DataFrame(
        list(
            (array[tuple(keys[i][label] for i, label in enumerate(prod))], *prod) 
            for prod in product(*labels)
        ),
        columns=[depvar]+columns
    )
    return tidy_df

def normalize_4d_nifti(img):
    """Normalize 4d nifti image by subtracting mean voxel intensity over time 
    and dividing by voxel intensity standard deviation over time.
    
    Args:
        img (4d Nifti1Image):
            Represent fMRI timecourse.
    
    Returns:
        List of individual timepoint 3d normalized images. To concatenate back
        to 4d use cocnat_images function from nibabel.funcs
    """
    img_std = math_img("np.std(img, axis=-1)", img=img)
    img_mean = math_img("np.mean(img, axis=-1)", img=img)
    img_list = []

    for img_3d in iter_img(img):
        img_list.append(math_img(
            "np.divide(img_3d-img_mean, img_std, where=(img_std!=0))", 
            img_3d=img_3d, 
            img_mean=img_mean, 
            img_std=img_std))

    return img_list