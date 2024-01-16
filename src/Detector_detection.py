# -Detector is postioned 6 mm too high
# -The detector has 1176x1104 pixels, and they are each 0.0495x0.0495 mm
# -The detector positions on the x and the y-axis were: X = 188.925 mm Y =   40.000 mm 
# -The distance between the source and the detector is approximately 197.5 mm (when the detector Z position is 0)
# - Each square and including the edge is 1.5 mm
import json
import numpy as np
import os
from dataObjects.ImageObject import ImageObject
from dataObjects.Settings import Settings
from models.interface.ImageReader import ImageReader
import pandas as pd
import time

#Functions
from functions.Detector_detection_functions.Detector_detection_funcs import *

def Main(parameters):

    #information test
    information = []

    #Set variables from parameters into settings object
    folder:str = parameters['folder']
    fileType: str = parameters['type']
    scale:int = int(parameters['scale'])
    showProgress:bool = bool(parameters['showProgress'])
    GRID_MM_LENGTH:float = float(parameters['gridMilliMeterLength']) 
    SINGLE_PIXEL_SIZE_MM:float = float(parameters['singlePixelSizeMilliMeter'])

    #Create and put information into settings
    settingsData:Settings = Settings(folder,fileType,scale,showProgress,GRID_MM_LENGTH,SINGLE_PIXEL_SIZE_MM)

    #Interface
    reader: ImageReader

    #Get files - based on file type -- Should be done as strategy pattern with a interface method that returns imageobject
    reader = MakeReader(settingsData.FILE_TYPE)

    #Read Images
    imageObjects = reader.ReadImage(settingsData)

    #Chosing of the images
    im1:ImageObject = imageObjects[0]

    #PROCESSING############################### 
    DSD,DSD_VAR,DSD_SIGMA = ProcessDSD(imageObjects) 
    
    print("DSD: ",DSD)
    print("DSD var: ",DSD_VAR)
    print("DSD sigma: ",DSD_SIGMA)

    #region
    #Do I need ((DSD - DSO)/DSO) -1 and ((DSD+100 - DSO)/DSO) -1??
    #B: the larger distance
    #A: the shorter distance
    #2. Find DSD with: ((B-A)/A) = %, where 50% is 100 mm and 100% is 200 mm
    # #Percentage difference between two images
    # percentage = ((im2.magnification-im1.magnification)/im1.magnification)

    # #DSD = FindDSD(percentage,Distances[ii]-Distances[i])

    # #3. Used the percentage change to apply a affine matrix to change the scale and at the same time keep the image centered (for image1)
    # appliedScale = np.round(percentage,3)

    # #Create identity matrix
    # #Positions for scale are x = [0,0] and y = [1,1]. Positions for translation are x = [0,2] and y = [1,2]
    # warpScaleMatrix = np.eye(2,3,dtype=np.float32)

    # #Scale
    # warpScaleMatrix[0,0] += appliedScale
    # warpScaleMatrix[1,1] += appliedScale

    # #Apply percentage change to the x and y so it can used in the translation part of the matrix and keep it centered
    # xApplied = im1.x * appliedScale/2
    # yApplied = im1.y * appliedScale/2

    # #translation - this reason for it being -= is because we are always going from small to larger distance, so 0 mm to 200 mm which will always require 
    # #a scaling up and therefore the y and x will increase and we mitigate this by subtracting x/y applied
    # #If we reverse the process so 200 mm to 0 mm then we would be scaling down and therefore += is needed
    # warpScaleMatrix[0,2] -= xApplied
    # warpScaleMatrix[1,2] -= yApplied
    
    #endregion
    
    #Distance difference #IS this right? 
    maxDistance = imageObjects[-1].distance - imageObjects[0].distance 

    #Go through and compare all images
    for i in range(1,len(imageObjects)):

        #Create warp matrix for im1 to create a new image object
        warpScaleMatrix = CreateMagnificationMatrix(im1,imageObjects[i])

        #Warp im1 using affine
        im1ScaledUp:ImageObject = im1.WarpNew(warpScaleMatrix,imageObjects[i].distance) #Add target distance

        #Checking magnification of the scaled up image and the target image
        AddSpacePrint()
        ShowImage(im1ScaledUp)
        FindMagnification(im1ScaledUp)
        print(f"Im2 magnification compared to the scaled up im1 magnification: {imageObjects[i].magnification} == {im1ScaledUp.magnification}")
        ShowImages([im1ScaledUp,imageObjects[i]],["im1 scaled up","im2"])

        #4. With the imaged up/down the next step is to find the offset(translation) and magnification(scale) that will change image1 to fit onto image2

        #Process final shift matrix
        FinalShiftMatrix = ProcessShift(im1ScaledUp,imageObjects[i])

        #Changing the first matrix to be the final one 
        warpScaleMatrix[0,2] += FinalShiftMatrix[0,2]
        warpScaleMatrix[1,2] += FinalShiftMatrix[1,2]

        #Take im1 which is unchanged and apply the combined matrix to it to see if it fits on im2
        confirmationIm:ImageObject = im1.WarpNew(warpScaleMatrix,imageObjects[i].distance)

        ShowImages([confirmationIm,imageObjects[i],ImageObject(imageObjects[i].image - confirmationIm.image,imageObjects[i].distance,imageObjects[i].settings)],[f"Confirmation image ({confirmationIm.distance} mm)",f"Image ({imageObjects[i].distance} mm)",f"Difference between of the two images"])
        
        #Copy to give a better name
        FinalWarpMatrix = warpScaleMatrix.copy()

        #Printing
        AddSpacePrint()
        print("Final Warp Matrix:")
        print(FinalWarpMatrix) #Does not get used, is the translation and magnification combined
        AddSpacePrint()
        print("Final Shift Warp Matrix:")
        print(FinalShiftMatrix)

        #5.--------------------- #TODO: Correct?
        
        #Change the pixel to mm for both x and y
        mmShiftX,mmShiftY = CalcShiftMM(im1,imageObjects[i],FinalShiftMatrix,maxDistance,DSD)

        #print information to a panda dataframes
        information.append([
                    imageObjects[i].distance,
                    np.round(mmShiftX,4),
                    np.round(mmShiftY,4)])
    
    names = [   'Distance [mm]',
                'Shift X [mm]',
                'Shift Y [mm]']
    
    df = pd.DataFrame(information,columns=names).to_excel('output.xlsx')
    
    AddSpacePrint()
    outputString: str = f"""
    \b\b\b\b----------------------------------------------------------------------
    \b\b\b\bMean Shift X: {statistics.mean([info[1] for info in information])}
    \b\b\b\bMedian Shift X: {statistics.median([info[1] for info in information])}
    \b\b\b\b----------------------------------------------------------------------
    \b\b\b\bMean Shift Y: {statistics.mean([info[2] for info in information])}
    \b\b\b\bMedian Shift Y: {statistics.median([info[2] for info in information])}
    \b\b\b\b----------------------------------------------------------------------
"""
    print(outputString)
#^Being used

#region main
#Clear terminal - does not work
if(os.name == 'nt'):
    os.system('cls')
else:
    os.system('clear')

#Set perferred print option
np.set_printoptions(suppress=True)

#Timer - remove later
start = time.time()

#Parameters from settings
settingsPath =  os.path.join(os.path.curdir,'..//settings.json') 
settings = open(settingsPath)

parameters = json.load(settings)

Main(parameters)

end = time.time()
print(f"Took {end - start} sec(s) to complate")
#endregion main