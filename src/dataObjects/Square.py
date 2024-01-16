from dataObjects.Point import Point
from dataObjects.LinFunc import LinFunc
class Square:
    upperRightCorner:Point
    lowerLeftCorner:Point

    upperLeftCorner:Point
    lowerRightCorner:Point
    

    def __init__(self,upperRightCorner:Point,upperLeftCorner:Point,lowerRightCorner:Point,lowerLeftCorner:Point):
        self.upperRightCorner = upperRightCorner
        self.upperLeftCorner = upperLeftCorner
        self.lowerRightCorner = lowerRightCorner
        self.lowerLeftCorner = lowerLeftCorner

    def CreateLinearFunction(self,point1:Point,point2:Point) -> LinFunc:
        a = (point1.y - point2.y) / (point1.x - point2.x)
        b = point1.y - a * point1.x

        return LinFunc(a,b)
    
    def Intersection(self,func1:LinFunc,func2:LinFunc) -> Point:
        a1 = func1.a
        b1 = func1.b

        a2 = func2.a
        b2 = func2.b

        x = (b2 - b1) / (a1 - a2)

        y = b1 - (a1*b1 - a1*b2) / (a1-a2)

        return Point(x,y)

    def GetCenterPoint(self) -> Point:
        if(self.upperRightCorner is None and self.upperLeftCorner is None and self.lowerRightCorner is None and self.lowerLeftCorner is None):
            return None

        func1:LinFunc = self.CreateLinearFunction(self.upperLeftCorner,self.lowerRightCorner)

        func2:LinFunc = self.CreateLinearFunction(self.upperRightCorner,self.lowerLeftCorner)
        
        intersec = self.Intersection(func1,func2)

        return intersec
