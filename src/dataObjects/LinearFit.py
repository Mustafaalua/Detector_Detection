from sklearn import linear_model
import numpy as np
from dataObjects.Point import Point
from dataObjects.enums.Axis import Axis

class LinearFit:
    interpolationLimitUpper:float
    interpolationLimitLower:float
    extrapolationLimitLower:float
    extrapolationLimitUpper:float
    fit:linear_model.LinearRegression
    shape:np.ndarray
    medianStep:float
    r:float
    equation:str
    axis: Axis
    availableUpper: list[Point]
    availableLower: list[Point]
    extendable: list[Point]
    line:list[Point]

    def __init__(self,fit:linear_model.LinearRegression,medianStep:float,interpolationLimitUpper:float,interpolationLimitLower:float,r:float,line:list[Point],shape:np.ndarray,axis:Axis) -> None:
        #Make copies because the variables are not necessarily static but reference
        self.fit = fit
        self.interpolationLimitLower = interpolationLimitLower.copy()
        self.interpolationLimitUpper = interpolationLimitUpper.copy()
        self.medianStep = medianStep.copy()
        self.r = r.copy()
        self.line = line.copy()
        self.shape = shape.copy()
        self.axis = axis
        

        #Create equation for rows 
        if(self.axis == Axis.X):
            self.equation = f"eq: y = {np.round(self.fit.coef_[0],3)} • x + {np.round(self.fit.intercept_,3)} ----------------- R^2: {np.round(self.r,3)}"
        #Create equation for columns
        else:
            self.equation = f"eq: x = {np.round(self.fit.coef_[0],3)} • y + {np.round(self.fit.intercept_,3)} ----------------- R^2: {np.round(self.r,3)}"

        #Create limit for extrapolation as to not go out to the egde of the image
        step = medianStep/10
        #Egde is 0 and we add to it
        self.extrapolationLimitUpper = 0 + step

        #Egde for lower is based on shape size and the axis
        self.extrapolationLimitLower = self.shape[self.axis.value] - step


        #Do calculations
        self.availableUpper = []
        self.availableLower = []
        self.extendable = []
        self.__calcExtendable()
        self.__calcAvailability()

    def PredictScalar(self,scalar:float) -> float|Point:
        #Check input scalar
        valid,point = self.__checkPredictionIsValid(scalar)
        if(not valid): 
            if(point == None): return -1.0
            else:
                return point

        #Wrap in numpy array so it can be inserted into fit 
        predictionInput = np.array([scalar]).reshape(-1,1)

        #Make prediction from linear fit
        prediction = self.fit.predict(predictionInput)

        #Take index 0 to make a float
        return prediction[0]

    def __checkPredictionIsValid(self,input:float) -> tuple[bool, Point|None]:
        #Check if prediction does not already exist
        isValid = True

        #Check through row or column and depending on axis - bheck both line and extendable 

        #Check the independet value - x
        if(self.axis == Axis.X):
            #Check line
            for cr in self.line:
                if(cr.x - 5 <= input <= cr.x + 5):
                    isValid = False

                    #Return extrapolation is not valid and None because it is already in the line
                    return [isValid,None]
                
            #Check extendable
            for cr in self.extendable:
                if(cr.x - 5 <= input <= cr.x + 5):
                    isValid = False
                    
                    #Return extrapolation is not valid and what point already exist
                    return [isValid, cr]
                
        
        #check the independet value - y
        elif(self.axis == Axis.Y):
            #Check line
            for cr in self.line:
                if(cr.y - 5 <= input <= cr.y + 5):
                    isValid = False

                    #Return extrapolation is not valid and None because it is already in the line
                    return [isValid,None]
                
            #Check extendable
            for cr in self.extendable:
                if(cr.y - 5 <= input <= cr.y + 5):
                    isValid = False

                    #Return extrapolation is not valid and what point already exist
                    return [isValid, cr]
        
        #Gets here if all is valid
        return [isValid, None]

    def __calcExtendable(self) -> None:
        #Check extendable line points based on axis
        
        #ROWS
        if(self.axis == Axis.X):
            #Expand rows if possible
            
            #Expand upper
            up = self.line[0]
            while(up.left != None):
                up = up.left
                self.extendable.append(up)

            #Expand lower
            down = self.line[-1]
            while(down.right != None):
                down = down.right
                self.extendable.append(down)
        
        #COLUMNS
        elif(self.axis == Axis.Y):
            #Expand columns if possible

            #Expand upper
            up = self.line[0]
            while(up.above != None):
                up = up.above
                self.extendable.append(up)

            #Expand lower
            down = self.line[-1]
            while(down.under != None):
                down = down.under
                self.extendable.append(down)



    def __calcAvailability(self) -> None:
        #Calculate from interpolation limit possible upper and lower points

        #Check up
        evalUp = self.interpolationLimitUpper.copy()
        while(evalUp - self.medianStep > self.extrapolationLimitUpper):
            #Move to next step
            evalUp -= self.medianStep

            #Calc prediction and make a point
            pred = self.PredictScalar(evalUp)

            #Case if not valid pred == -1.0
            if(pred == -1.0):
                #Do nothing
                return
            #Case if pred is a point already and therefore can be appended
            elif(isinstance(pred,Point)):
                self.availableUpper.append(pred)
            #X axis
            elif(self.axis == Axis.X):
                self.availableUpper.append(Point(evalUp,pred))
            #Y axis
            elif(self.axis == Axis.Y):
                self.availableUpper.append(Point(pred,evalUp))

        #Check down
        evalDown = self.interpolationLimitLower.copy()
        while(evalDown + self.medianStep < self.extrapolationLimitLower):
            #Move to next step
            evalDown += self.medianStep
            
            #Calc prediction and make a point
            pred = self.PredictScalar(evalDown)
            
            #Case if not valid pred == -1.0
            if(pred == -1.0):
                #Do nothing
                return
            #Case if pred is a point already and therefore can be append
            elif(isinstance(pred,Point)):
                self.availableLower.append(pred)
            #X axis
            elif(self.axis == Axis.X):
                self.availableLower.append(Point(evalDown,pred))
            #Y axis
            elif(self.axis == Axis.Y):
                self.availableLower.append(Point(pred,evalDown))

    def __str__(self) -> str:
        return self.equation
    
    def __repr__(self) -> str:
        return self.equation
