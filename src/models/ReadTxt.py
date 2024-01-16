from models.interface.ImageReader import ImageReader
from dataObjects.ImageObject import ImageObject
from dataObjects.Settings import Settings
import os
import cv2
from functions.ImageReadFunctions.ImageReadFuncs import *

#Concrete
class ReadTxt(ImageReader):
        
    def ReadImage(self, settings: Settings) -> list[ImageObject]:
        """
        Searches a specific folder for any .txt files and converts them to images/numpy arrays.

        Parameters
        ----------
        `folder : str`
            the folder path were each indivual file path will be extracted.

        Returns
        -------
        `tuple[list[cv2.Mat],list[float]]`
            the first `list[cv2.Mat]` contains the detector images.\n
            the second `list[float]` contains the distances for marked by each file path.\n
        """
        #Find folder
        print(f"Searching for the folder: {settings.FOLDER}")
        path : str = os.path.join(os.path.curdir,settings.FOLDER)
        print(f"Path found: {path}")
        
        openFolder = os.path.join(path, 'grid')
        print(f"Open folder: {openFolder}")

        #Images
        images : list[cv2.Mat] = [] 

        #Distances
        distances : list[float] = []

        #Extract files
        files = os.listdir(openFolder)
        
        #Sort files
        sortedFiles = sorted(files,key=lambda x: GetDistanceFromPath(x))

        for file in sortedFiles:
            split_tup = os.path.splitext(file)
            if (split_tup[1].lower() == '.txt'):
            
                img_array = load_and_correct_images(os.path.join(openFolder, file))

                #ShowImage(img_array,"Array as is")

                clipped_img_array = ClipArray(img_array)

                #ShowImage(img_array,"inf and nan removed")

                #Convert to make using opencv functions possible/easier as some of the methods used do not support 32bit float or 32bit int tif/tiff files
                ConvertedImage = ConvertTo8U(clipped_img_array)

                #Autocontrast
                ConvertedImage = AutoContrast(ConvertedImage)

                #Save the image
                images.append(ConvertedImage)
                    
                #Get distances
                distance = GetDistanceFromPath(file)
                distances.append(distance)

                #ShowImage(ConvertedImage, "uint8")
        
        #Convert to image object
        imObj:list[ImageObject] = []
        for i in range(len(images)):
            imObj.append(ImageObject(images[i],distances[i],settings))

        return imObj