from __future__ import annotations
from typing import Optional
from math import sqrt
from dataObjects.enums.Direction import Direction

class Point:
    x:float
    y:float

    above: Optional[Point]
    under: Optional[Point]
    left: Optional[Point]
    right: Optional[Point]

    distanceToAbove:float
    distanceToUnder:float
    distanceToLeft: float
    distanceToRight:float

    isExtrapolated:bool

    def __init__(self,x:float,y:float):
        self.x = x
        self.y = y

        #DEFAULT:
        self.above = None
        self.under = None
        self.left = None
        self.right = None

        self.distanceToAbove = 0.0
        self.distanceToUnder = 0.0
        self.distanceToLeft = 0.0
        self.distanceToRight = 0.0


    def AddNeighboringPoints(self,point:Point,position:Direction):
        #Check if None
        if(point == None): return

        #Check if "point" and "self" already know eachother - redundant if a check is done before using this method
        selfHasPoint = self.GetNeighboringPoints().__contains__(point)
        pointHasSelf = point.GetNeighboringPoints().__contains__(self)

        if(selfHasPoint and pointHasSelf): return

        distance = self.__calcDistance(point)

        if(position == Direction.ABOVE):
            self.above = point
            self.distanceToAbove = distance

            #Also update point given
            self.__adjustOpposite(point,Direction.UNDER,distance)

        elif(position == Direction.UNDER):
            self.under = point
            self.distanceToUnder = distance

            #Also update point given
            self.__adjustOpposite(point,Direction.ABOVE,distance)

        elif(position == Direction.LEFT):
            self.left = point
            self.distanceToLeft = distance

            #Also update point given
            self.__adjustOpposite(point,Direction.RIGHT,distance)

        elif(position == Direction.RIGHT):
            self.right = point
            self.distanceToRight = distance

            #Also update point given
            self.__adjustOpposite(point,Direction.LEFT,distance)
    
    def GetNeighboringPoints(self) -> list[Point]:
        points = []

        if(self.above == None):
            points.append(None)
        else:
            points.append(self.above)

        if(self.under == None):
            points.append(None)
        else:
            points.append(self.under)

        if(self.left == None):
            points.append(None)
        else:
            points.append(self.left)

        if(self.right == None):
            points.append(None)
        else:
            points.append(self.right)

        return points


    def GetNeighboringDistances(self) -> list[float]:
        return [self.distanceToAbove,self.distanceToUnder,self.distanceToLeft,self.distanceToRight]


    def __calcDistance(self,point:Point) -> float:
        #Vector calc d(self,point)
        dx = (self.x - point.x)**2
        dy = (self.y - point.y)**2

        dxy = dx + dy
    
        d = sqrt(dxy)

        return d
    

    def __adjustOpposite(self,point:Point,position:Direction,distance:float) -> None:
        #Should only be called by "AddNeighboringPoints", since if "this" point has a right point, then the point to right of it must have "this" to the left of it
        
        #point should have "self" added to it
        if(position == Direction.ABOVE):
            point.above = self
            point.distanceToAbove = distance

        elif(position == Direction.UNDER):
            point.under = self
            point.distanceToUnder = distance

        elif(position == Direction.LEFT):
            point.left = self
            point.distanceToLeft = distance

        elif(position == Direction.RIGHT):
            point.right = self
            point.distanceToRight = distance


    def __str__(self) -> str:
        string = f"""Center is:\nx is {self.x}\ny is {self.y}
                  """

        if(self.above != None):
            string += f"""
                      Above is:
                      x is {self.above.x}
                      y is {self.above.y}
                      distance is {self.distanceToAbove}
                    """
        if(self.under != None):
            string += f"""
                      Under is:
                      x is {self.under.x}
                      y is {self.under.y}
                      distance is {self.distanceToUnder}
                    """
        if(self.left != None):
            string += f"""
                      Left is:
                      x is {self.left.x}
                      y is {self.left.y}
                      distance is {self.distanceToLeft}
                    """
        if(self.right != None):
            string += f"""
                      Right is:
                      x is {self.right.x}
                      y is {self.right.y}
                      distance is {self.distanceToRight}
                    """
        return string
    
    def __repr__(self) -> str:
        return f"Extrapolated point: (x:{int(self.x)}, y:{int(self.y)})" if self.isExtrapolated else  f"Point: (x:{int(self.x)}, y:{int(self.y)})"