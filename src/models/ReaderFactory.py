from models.ReadTiff import ReadTiff
from models.ReadTxt import ReadTxt
from models.ReadNpy import ReadNpy
from dataObjects.enums.ReaderType import ReaderType
from models.interface.ImageReader import ImageReader


class ReaderFactory:

    def CreateReader(self, rType:ReaderType) -> ImageReader:
        #Check type and return class object
        if(rType == ReaderType.TIFF):
            return ReadTiff()
        elif(rType == ReaderType.TXT):
            return ReadTxt()
        elif(rType == ReaderType.NPY):
            return ReadNpy()