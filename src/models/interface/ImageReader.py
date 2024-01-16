from abc import ABC,abstractmethod
from dataObjects.Settings import Settings
from dataObjects.ImageObject import ImageObject
import numpy as np
import cv2

## Strategy interface
class ImageReader(ABC):
    @abstractmethod
    def ReadImage(self,settings:Settings) -> list[ImageObject]:
        pass