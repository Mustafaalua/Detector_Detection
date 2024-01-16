# Getting started
Before you get started download the required packages to run the script.

## Installation
To download the packages required please navigate to the src folder and run the following line in the terminal. 
```bash
pip install -r requirements.txt
```

## Settings
You can adjust the scripts settings via a json file called settings.json located in the Detector_Detection folder.

Format of the settings folder:
```bash
{
    "folder" : "../20231207_alignment_grid_offset",
    "type": ".npy",
    "scale": 1.0,
    "showProgress": 0,
    "gridMilliMeterLength": 1.5,
    "singlePixelSizeMilliMeter":0.055
}
```

`folder` is the path to the folder that contains the images that will be used to determine the correction. The path given should be written with src as current folder.
***
`type` is the file type that the images are saved as. If a type or certain specifics, such as file naming, is not implemented then they can be implemented with ImageReader interface.
***
`scale` is the scaling that will be done to the images in the process of finding the DSD (detector-source-distance) with 1.0 being no scaling done. Changing the scaling can give some variance to the results.
***
`showProgress` is boolean type 0 or 1 that indicates whether the scripts should show the images throughout the process or not. 

0: Do not show images. 

1: Show images.
***
`gridMilliMeterLength` is the length of a square on the grid in millimeters.
***
`singlePixelSizeMilliMeter` is the size of the detector pixels in millimeters. 


## Nice to know
Here are some things that are nice to know when reading through the code.
The examples will mostly be shown in image.

#### Axis
The Script is written with opencv and the defualt placement of origin is the top left corner of the image.


![!\[Alt text\](XandY.png)](readmeImages/XandYAxis.png)
***
#### Lower and Upper
Throughout the script upper and lower are mentioned a lot. The first use of upper and lower was done in respective to the y-axis but not value based. It was done by looking at the image. 

When you look at the bottom of the image and move your eyes up, it was then considered to an upwards movement even though the y value got smaller. That logic did unfortunately not work when looking at the x-axis. 

CHECK
The principle is still the same. When the axis value gets smaller it is considered upper. When the axis value gets larger it is considered lower. 

![Alt text](readmeImages/UpperAndLower.png)
***
#### Grid length in mm
The length that this references is that of a single square on the grid.
This same length is found in the script but as pixels instead of mm.
![Alt text](readmeImages/gridmillimeter.png)

***
#### Translation/Affine matrix
To edit the images in the script a matrix is needed by openCV. Here is a simple description for each index of x and y.

The initial matrix is an eye matrix:

```math
\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}
```
1 is used as default scaling number.

The combinations of x and y:

```math
\begin{bmatrix}0,0\end{bmatrix} -> Scaling for x, defualt is 1.
```

$\begin{bmatrix}0,1\end{bmatrix}$ -> Shearing for y, defualt is 0.

$\begin{bmatrix}0,2\end{bmatrix}$ -> Translation for x, defualt is 0.

$\begin{bmatrix}1,0\end{bmatrix}$ -> Shearing for x, defualt is 0.

$\begin{bmatrix}1,1\end{bmatrix}$ -> Scaling for y, defualt is 1.

$\begin{bmatrix}1,2\end{bmatrix}$ -> Translation for y, defualt is 0.

Overview with names:

$\begin{bmatrix}ScalingX&ShearingY&TranslationX\\ShearingX&ScalingY&TranslationY\end{bmatrix}$

***
## Walkthrough of the general steps of the script
Here we will see the general and most important steps of the script in finding the correction to the detector.

#### 1.

First step that happens is reading the images into the script and save as ImageObjects. If more reading methods are needed then add a new .py script in the models folder. The script has to implement the interface ImageReader.py. Afterwards add appropriate element for the new script in the dataObjects/enums/ReaderType.py. 
Lastly register the new file type in the ReaderFactory.py under models. 

#### 2.
Second step is processing the DSD (detector-source-distance) by identifying squares and use them as reference to find other squares.

They are identified by doing bluring, thresholding and finding contours.

This is an example of the process. 
![Alt text](readmeImages/stepsSquares.png)

The real contours will match the roundness of the corners of each square.

From here the center point of each square is found.

![Alt text](readmeImages/stepsCenterPoints.png)

Each 2x2 pair points form a new kind of square, which has its own class object.
That is because each pair of points is used to cross over the center of the 2x2 as shown on the next image.

![Alt text](readmeImages/stepsCross.png)

This is done for each pair that can be found on the image.

This creates a new set of points that are on the center of each crossover.

![Alt text](readmeImages/stepsCrossovers.png)

If these points meet certain conditions then the script will try to extrapolate them where a 2x2 pair was not possible. This will add new points to the set.

For this example we will not simulate extrapolation.

These points will be used to create a big square around the outermost points.

![Alt text](readmeImages/stepsBigSquare.png)

This big square has a pixel area that fits 4 squares. Here we can find the length of each square that matches the size of "Grid length in mm" section but in pixels instead of mm.

The lowest distance that the image was taken at that usually is 0 mm and is used as a baseline for both distance and length when calculating DSD. 

The process of processing DSD includes finding the magnification and that is added to the ImageObject attribute. This will be used in step 3.

#### 3.
Third step is processing the shift between the images in pixels. This is done through cross correlation between the images to find a common spot between them.

When comparing images to eachother the ones with lower distances will always be more zoomed out then the higher distances. Therefore the lower distance has to be changed to fit the higher distance.

This is done by looking at the magnifications and change the scale of the lower distance image based on the difference of magnification.

The scale needs to occur with a translation of the x and y to center the image again. This is because the scaling move the image without taking the original center into account.

When the images have similiar magnification the calculations of the shift can begin.

The set up for taking the images is a 3D-printed plastic stand. The stand can hold the physical grid with plastic behind, except a small 1 mm hole. This creates a spot where the beam attenuates less than the rest.

![Alt text](readmeImages/stepsHole.png)

On the final image this appears as higher pixel values than the rest. The red circle shows the higher value pixels.

With simple thresholding and some filtering, we can discern this spot from the rest of the image. Since all images have this spot, we can use it as reference to correlate the images.

An initial correlation is done by taking the medians of the spots for x and y. This gives two sets of x and y that describes where the spot is. The difference is used in a translation matrix to fit the lower distance image to the higher distance image.

Lastly for the fine tuned correlation is done by ECC (Enhanced Cross Correlation), that is because the ECC function can not handle large difference in the two images and therefore used for the final bit. This usaully yield about 1-3 pixels in correction.

We end up with sets of matrices that describes the needed translation/shift from one image the the other. 

#### 4.
Final step is taking the shift and turning it into a correction in mm instead of pixels.

We are not 100% sure about this process because it is not tested yet.

The process of converting the shift to mm is done by using `gridMilliMeterLength` and `singlePixelSizeMilliMeter`.

This output shift tells us how much the image has moved in mm from one to the next. That is not enough information, because we need it to tells us how much the detector has shifted off center.

The detector shift is found by considering a triangle from the source to the detector for both x and y to create a vector. The relations between the distances in the images and the DSD and the shift found before is used to calculate the correction shift.

This correction shift will be printed both for mean and median correction.

***
## TODO
* **Correction direction:** The output from the script is in mm for both x and y. The problem is, that they do not tell if it's the actual direction. This can depend on if the image is flip at some point or not. This will need testing, where the directions is applied and evaluated manually to later adjust the output. 


* **Units in findtranslation REDO:** The function used to translate the pixels to millimeter might not be correct and needs to be evaluated again.

    The current function to translate pixels to millimeter is ```TranslatePixeltoMM```.

    ```python
    def TranslatePixeltoMM(pixelLength:float,gridMillimeterLength:float,realPixelLength:float) -> float:
        #units:
        #         px       *        mm
        #         --------------------  = px??????????
        #                  mm
        return (pixelLength * realPixelLength) / gridMillimeterLength
    ```
    The function consists of 2 constants, that were mentioned in the settings section.

    Therefore the return statement usaully looks like the following.
    ```python
    return (pixelLength * 0.055) / 1.5
    ```
    Units:
    * ```pixelLength``` = [ px ]
    * ```realPixelLength``` = [ mm ]
    * ```gridMillimeterLength``` = [ mm ]

    The function has to use cases in the script, it is used to find magnification and shift. Here is a clear conflict and as magnification is without a unit and shift is with a unit.

    They should be separated. The name ```TranslatePixeltoMM``` should stay in ```CalcShiftMM```. 

* **magnification:** The magnification is used to scale an image onto another image. This has partly some of the same problems as translation.

* **Spelling mistakes** The script comments has some spelling mistakes, weird sentences or bad grammar, that need to be run through and corrected. 