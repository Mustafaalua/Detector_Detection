import numpy as np
import cv2
from dataObjects.Settings import Settings

class ImageObject:
    image : np.ndarray
    distance : float
    x : int
    y : int
    shape : np.ndarray
    magnification : float
    settings : Settings

    def __init__(self,image:np.ndarray,distance:float,settings:Settings) -> None:
        self.image = image
        self.distance = distance
        self.settings = settings

        #Create rest of the information
        self.y,self.x = image.shape
        self.shape = np.array([self.x,self.y])

    def SetMagnification(self,magnification:float) -> None:
        self.magnification = magnification
    def WarpSelf(self,warpMatrix:np.ndarray) -> None:
        self.image = cv2.warpAffine(self.image,warpMatrix,(self.x,self.y), flags=cv2.INTER_LINEAR)

    def WarpNew(self,warpMatrix:np.ndarray,targetDistance:float):
        return ImageObject(cv2.warpAffine(self.image,warpMatrix,(self.x,self.y),flags=cv2.INTER_LINEAR),targetDistance,self.settings)

    def __str__(self) -> str:
        return f"""
                   Distance is {self.distance},
                   x is {self.x},
                   y is {self.y},
                   Shape is {self.shape},
                   Magnification is {np.round(self.magnification,3)}.
                """

    def __repr__(self) -> str:
        return self.__str__()
        

    