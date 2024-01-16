from models.interface.ImageReader import ImageReader
from dataObjects.ImageObject import ImageObject
from dataObjects.Settings import Settings
import numpy as np
import os
from functions.ImageReadFunctions.ImageReadFuncs import *

#Concrete
class ReadNpy(ImageReader):


    def ReadImage(self, settings:Settings) -> list[ImageObject]:

        #Find paths to the within the folder/directory
        path : str = os.path.join(os.path.curdir,settings.FOLDER)
        directory:list[str] = os.listdir(path)

        #Filter for safety
        files : list[str] = list(filter(lambda x: x.__contains__(".npy"),directory))

        #Combine path and files
        filePaths: list[str] = []
        for file in files:
            #Take the single file path and append
            filePaths.append(os.path.join(path,file))

        #Seperate the .npy files into grid and open files
        npyGrid : list[str] = list(filter(lambda x: x.__contains__("grid"),filePaths))
        npyOpen : list[str] = list(filter(lambda x: x.__contains__("open"),filePaths))

        #Should only contain one element since the .npy file should be able to contain all images
        gridArray = np.load(npyGrid[0])
        openArray = np.load(npyOpen[0])

        #Correct image by dividing 
        correctedGridArray = gridArray/openArray

        #ex: data dimensions are 9 x 512 x 1280
        #                        ^ distances 
        #                             ^ y
        #                                    ^ x 

        #Use shape (aka dimensions) to define the distances and images
        steps = np.shape(correctedGridArray)[0]

        #The distance the is between each step in mm
        stepDistance:float = 25.0

        #Baseline distance
        distance:float = 0.0

        #List of ImageObjects to be returned
        imageObjects:list[ImageObject] = []

        #Go through steps, range(1,steps), 1 because the baseline distance has already been added
        for i in range(steps):
            #With each step we can also extract the image from correctedGridArray
            im:np.ndarray = correctedGridArray[i]

            #The image needs some processing before being turned into imageObject
            clipped_img_array = ClipArray(im)

            #Convert to make using opencv functions possible/easier as some of the methods used do not support 32bit float or 32bit int files
            ConvertedImage = ConvertTo8U(clipped_img_array)

            #Autocontrast
            ConvertedImage = AutoContrast(ConvertedImage)

            #Set pixel value 10 to 0
            ConvertedImage = np.where(ConvertedImage <= 10,0,ConvertedImage)

            #Take im and distance and make an imageobject
            imageObjects.append(ImageObject(ConvertedImage,distance,settings))

            #Move distance one step
            distance += stepDistance

        return imageObjects




