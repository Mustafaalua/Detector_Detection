class Settings:
    
    FOLDER : str
    FILE_TYPE : str
    SCALE: float
    SHOW_PROGRESS:bool
    GRID_MILLIMETER_LENGTH: float
    SINGLE_PIXEL_SIZE_MILLIMETER : float
    
    def __init__(self,folder:str,fileType:str,scale:float,showProgress:int,gridMilliMeterLength:float,singlePixelSizeMilliMeter:float) -> None:
        
        self.FOLDER = folder
        self.FILE_TYPE = fileType
        self.SCALE = scale
        self.SHOW_PROGRESS = bool(showProgress)
        self.GRID_MILLIMETER_LENGTH = gridMilliMeterLength
        self.SINGLE_PIXEL_SIZE_MILLIMETER = singlePixelSizeMilliMeter
