import numpy as np
import cv2
import os

def AutoContrast(im:cv2.Mat) -> cv2.Mat:
    """
    Automatically applies contrast to an image and returns it 

    Parameters
    ----------
    `im : cv2.Mat`
        the image(`im`) that will be applied contrast to

    Returns
    -------
    `cv2.Mat`
        image with contrast applied to it
    """
    #Autocontrast process here
    #if need use this: https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    
    #g(i,j) = a * f(i,j) + b, where alpha and beta can be used to adjust contrast and brightness. 
    
    gray = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    #Calculate grayscale histogram
    hist = cv2.calcHist([im],[0],None,[256],[0,256])
    hist_size = len(hist)

    #Calculate cumulative distrubution from the histogram
    accumulator = []
    accumulator.append(float(hist[0][0]))
    for index in range(1,hist_size):
        accumulator.append(accumulator[index-1]+float(hist[index][0]))

    #Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent = 1
    clip_hist_percent *=(maximum/100.0)
    clip_hist_percent /= 2.0

    #locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    

    #locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)
    return auto_result

def ConvertTo8U(im:cv2.Mat) -> cv2.Mat:
    """
    Takes an image and converts it to uint8.
    Usually to make other cv2 functions work. 

    Parameters
    ----------
    `im : cv2.Mat`
        the image(`im`) to be converted, usually .tiff or .tif.

    Returns
    -------
    `cv2.Mat`
        the converted image.
    """
    #convert with normalizing    
    return cv2.normalize(src=im,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

def ClipArray(array:np.ndarray) -> np.ndarray:
    """
    Clips values away from array based on percentile.

    Extended Sumamry
    ----------
    The clipping is done to values outside 10 - 99 percentile. \n
    Values such as `nan` are changed to `0` and `inf` are changed to `1`  

    Parameters
    ----------
    `array : np.ndarray`
        any form of numpy array with unclipped values.

    Returns
    -------
    `np.ndarray`
        the array with values clipped.
    """
    #Find NaN in the image and change them --> causes TransformECC to crash if not removed/changed
    array = np.where(np.isnan(array),0, array)
    #                  ^               ^     ^ if false remain unchanged
    #            condition    if true replace with

    #Find inf in the image and change them --> causes the image to become black if not removed/change because it divides with inf
    array = np.where(np.isinf(array),1.0,array)

    lowerPercentile = np.percentile(array,10) 
    upperPercentile = np.percentile(array,99)

    clipped_img_array = np.clip(array,lowerPercentile,upperPercentile)

    return clipped_img_array

def GetDistanceFromPath(path:str) -> float:
    """
    Searches path to extract distance along z-axis with underscore(`_`) being the seperator.

    Parameter
    ---------
    `path : str`
        the path that the distance will be extracted from.

    Returns
    -------
    `float`
        a value for the distance that the images was taken from.
    """
    #Breakdown string
    splitPath = path.lower().split('_')

    #Find z
    index = splitPath.index('z') 

    #The space after should be the number
    index += 1
    
    distance = float(splitPath[index])

    return distance

def expand_and_fill_gap_one_image_Medipix(arr: np.ndarray, gap_size_px: 2, crop_to_orginal: True)-> np.ndarray:
    Gap = gap_size_px
    Halfgap = Gap // 2
    H = 256  # chipsize
    K = H + Gap  # chip + gap

    # determine the number of chips on the x-axis
    n_chips_x = int(arr.shape[0]/256)
    # determine the number of chips on the y-axis
    n_chips_y = int(arr.shape[1]/256)

    if (n_chips_x == 1) and (n_chips_y == 1):  # just one chip, so no gaps to fill
        return arr

    new_arr = np.zeros(
        (arr.shape[0] + ((n_chips_x - 1)*Gap), arr.shape[1] + ((n_chips_y - 1)*Gap)))

    # set all the cross pixels to zero in the original image
    for chip_nr_x in range(1, n_chips_x):
        arr[chip_nr_x*H-1:chip_nr_x*H+1, :] = 0
    for chip_nr_y in range(1, n_chips_y):
        arr[:, chip_nr_y*H-1:chip_nr_y*H+1] = 0

    # copy the original imaging data in the new larger array with gaps
    for chip_nr_y in range(n_chips_y):
        for chip_nr_x in range(n_chips_x):
            new_arr[chip_nr_x*K:(chip_nr_x*K)+H, chip_nr_y *
                    K:(chip_nr_y*K)+H] = arr[chip_nr_x*H:(chip_nr_x*H)+H, chip_nr_y*H:(chip_nr_y*H)+H]

    # now fill the gaps by copying the count number of the pixel closest to it
    # first in the x-direction
    for chip_nr_x in range(1, n_chips_x):
        lineindex = (chip_nr_x*H)+((chip_nr_x-1)*Gap) - 2
        for i in range(lineindex+1, lineindex + 2 + Halfgap):
            new_arr[i, :] = new_arr[lineindex, :]
        lineindex = (chip_nr_x*H)+((chip_nr_x-1)*Gap) + Gap + 1
        for i in range(lineindex-Halfgap-1, lineindex):
            new_arr[i, :] = new_arr[lineindex, :]
    # then in the y-direction
    for chip_nr_y in range(1, n_chips_y):
        lineindex = (chip_nr_y*H)+((chip_nr_y-1)*Gap) - 2
        for i in range(lineindex+1, lineindex + 2 + Halfgap):
            new_arr[:, i] = new_arr[:, lineindex]
        lineindex = (chip_nr_y*H)+((chip_nr_y-1)*Gap) + Gap + 1
        for i in range(lineindex-Halfgap-1, lineindex):
            new_arr[:, i] = new_arr[:, lineindex]

    if crop_to_orginal:
        return new_arr[(n_chips_x-1)*Halfgap:-(n_chips_x-1)*Halfgap, (n_chips_y-1)*Halfgap:-(n_chips_y-1)*Halfgap]
    else:
        return new_arr   

def load_and_correct_images(filename):
    print(filename)
    if os.path.exists(filename):
        #load the image
        img = np.loadtxt(filename).astype(np.float32)
        #crop the bad chips off
        img = img[:,0:-256]
        #flip the image
        img = np.flip(img, axis=1)
        #get rid of the crosses in the image and correct the size
        result = expand_and_fill_gap_one_image_Medipix(img, 2, True)
        return result
    