from models.interface.ImageReader import ImageReader
from dataObjects.ImageObject import ImageObject
from dataObjects.Settings import Settings
import os
import cv2
from functions.ImageReadFunctions.ImageReadFuncs import *

##Concrete
class ReadTiff(ImageReader):

    #Helper method
    def ProcessImage(tifDetector:list[str],tifOpen:list[str],darkFolder:str,AutoContrast:bool = False,ShowProcess:bool = False) -> list[cv2.Mat]:
        """
        Takes image paths and process the image to a usable state.

        Extended Summary
        ----------
        Process the image by doing a series of sequences:
        1. open - dark = openResult
        2. detector - dark = detectorResult
        3. detectorResult / openResult
        4. if true: autocontrast 

        Parameters
        ----------
        `tifDetector : list[str]`
            the paths for the detector images. 

        `tifOpen : list[str]`
            the paths for the open images.

        `darkFolder : str`
            the path to the folder containing the dark image.

        `AutoContrast : bool`, `optional`
            this sets if autocontrast should be done or not before returning the image

        `ShowProcess : bool`, `optional`
            this sets the function to show every processed image.

        Returns
        -------
        `list[cv2.Mat]`
            a list containing all the processed images in order of distances.
        """
        #resultOpen = open - dark
        #resultProjection = tifDetector - dark
        #correctedImage = resultProjection / resultOpen
        #correctedImage --> autocontrast needed

        #List to be returned later
        correctedImage : [cv2.Mat] = []

        #Get images from path - doing two loop as sizes might not match. Do not know when that is the case
        DetectorImages : [cv2.Mat] = []
        for tif in tifDetector:
            #Read file, -1 is enum for UNCHANGED
            im = cv2.imread(tif,-1)

            DetectorImages.append(im)

        OpenImages : [cv2.Mat] = []
        for tif in tifOpen:
            #Read file, -1 is enum for UNCHANGED
            im = cv2.imread(tif,-1)

            OpenImages.append(im)

        #Get dark from folder
        listOfFiles = os.listdir(darkFolder)

        #Filter everything out that is not dark 
        listOfDarks = list(filter(lambda x: x.__contains__("dark") | x.__contains__("Dark"),listOfFiles))
        
        #Extract first instance of a dark image
        dark = cv2.imread(os.path.join(darkFolder,listOfDarks[0]),-1)


        #Do each calculation
        for (detector,open,i) in zip(DetectorImages,OpenImages,range(len(DetectorImages))):
            
            #Subtract open with dark
            resultOpen = cv2.subtract(open,dark)

            #Subtract detector with dark
            resultDetector = cv2.subtract(detector,dark)

            #Divide the two results
            result = cv2.divide(resultDetector,resultOpen)
            
            #Clip values - is most likely the reason that the image looks graining
            result = ClipArray(result)

            #Convert to make using opencv functions possible/easier as some of the methods used do not support 32bit float or 32bit int tif/tiff files
            Convertedresult = ConvertTo8U(result)

            #Auto contrast the image 
            if(AutoContrast):
                Convertedresult = AutoContrast(Convertedresult)

            #append
            correctedImage.append(Convertedresult)
            # if(ShowProcess):
            #     ShowImage(Convertedresult,str(i))

        return correctedImage

    #Concrete method from interface
    def ReadImage(self, settings:Settings) -> list[ImageObject]:
        
        #Find folder
        print(f"Searching for the folder: {settings.FOLDER}")
        path : str = os.path.join(os.path.curdir,settings.FOLDER)
        print(f"Path found: {path}")

        #darkfolder
        darkFolder = os.path.join(path,"dark")

        #Take all elements from the path and filter out what is not .tiff or .tif
        directory = os.listdir(path)
        files : list[str] = list(filter(lambda x: x.__contains__(".tiff") or x.__contains__(".tif"),directory))

        #Add path to the files, so cv2.imread can work, x is each element in the list
        files = list(map(lambda x: os.path.join(path,x), files))


        #Seperate the .tif/.tiff files into the detector files and open files
        tifDetector : list[str] = list(filter(lambda x: x.__contains__("detector"),files))
        tifOpen : list[str] = list(filter(lambda x: x.__contains__("open"),files))


        #Sort, might be reduntant but it is to assure matching indexes
        tifDetector.sort()
        tifOpen.sort()
        
        #Conform that they are matching, can be expanded upon
        matching = (len(tifDetector)==len(tifOpen))

        print("Are they matching: ",matching)

        print("All files found") 
        print("Showing files:")
        
        #List to hold distances as they are printed out
        tifDisance = []

        #Print the files that are matching
        for i in range(len(tifDetector)):

            #mm matching to see if each file really is matching by distance
            #ex: xxxxx_Z_000_mm.tiff --> xxxxx  Z   000    mm.tiff
            #                             [0]  [1]  [2]      [3]  
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Happens inside functions now.

            #Get distance for detector
            mmDetector = GetDistanceFromPath(tifDetector[i])

            #Get distance for open
            mmOpen = GetDistanceFromPath(tifOpen[i])

            #Check condition
            mmMatch = mmDetector == mmOpen
            print(f"found {tifDetector[i]} and {tifOpen[i]}   -->  mm match is {mmMatch}")

            #Save mm to use later
            tifDisance.append(float(mmDetector))


        #Get image and process, change ShowProcess to true to get every processed image to verify them manually
        images = self.ProcessImage(tifDetector,tifOpen,darkFolder)

        #Create image object
        imageObject = list[ImageObject] = []
        for i in range(len(images)):
            imageObject.append(ImageObject(images[i],tifDisance[i],settings))

        return imageObject



