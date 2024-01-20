import numpy as np
import matplotlib.pylab as plt
import cv2
import statistics
from skimage.util import invert
from dataObjects.ImageObject import ImageObject
from dataObjects.Point import Point
from dataObjects.enums.Direction import Direction
from dataObjects.Square import Square
from dataObjects.LinearFit import LinearFit
from dataObjects.enums.Axis import Axis
from dataObjects.enums.RowsAndColumns import RowsAndColumns
from dataObjects.enums.ReaderType import ReaderType
from models.interface.ImageReader import ImageReader
from models.ReaderFactory import ReaderFactory
from sklearn import linear_model

#Functions used in Detector_detection.py
def ShowImage(im:ImageObject,name:str="") -> None:
    """
    Description
    ----------
    Shows a choosen image with a custom title. If show progress is false then it wont be shown. 

    Parameters
    ----------
    `im : ImageObject`
        the image object(`im`) that will be shown.

    `name : str`, optional
        the title above the image. `Default = ""`

    Returns
    -------
    `None`
        
    """
    #Black bared for readme
    #im.image = cv2.copyMakeBorder(src=im.image, top=100, bottom=100, left=100, right=100, borderType=cv2.BORDER_CONSTANT) 
    if(im.settings.SHOW_PROGRESS):
        plt.imshow(im.image,cmap='gray')
        plt.title(name)
        x,y = im.shape
        plt.ylabel(f"{y} :px")
        plt.xlabel(f"{x} :px")
        plt.show()
def ShowImages(ims:list[ImageObject],names:[str]=[]) -> None:
    """
    Description
    ----------
    Shows images with a custom title. If show progress is false then they wont be shown. 

    Parameters
    ----------
    `ims : list[ImageObject]`
        the image objects(`ims`) that will be shown.

    `name : str`, optional
        the title above each image. `Default = ""`

    Returns
    -------
    `None`
        
    """
    #Create subplot split with the length of ims
    if(ims[0].settings.SHOW_PROGRESS):
        fig, axes = plt.subplots(nrows=1, ncols=len(ims), figsize=(10, 6), sharex=True, sharey=True)

        ax = axes.ravel()

        #Check if sizes add up
        while(len(ims) >= len(names)):
            names.append("")

        for i in range(len(ims)):
            x,y = ims[i].shape

            ax[i].imshow(ims[i].image)
            ax[i].set_ylabel(f"{y} :px")
            ax[i].set_xlabel(f"{x} :px")
            ax[i].set_title(names[i],fontsize=10)

        fig.tight_layout()
        plt.show()
def RemoveDots(contours:list) -> list:
    """
    Description
    ----------
    Remove contours that are under a specific area. Typically countours that are dot sized.

    Parameters
    ----------
    `contours : list`
        the contours to be searched through.

    Returns
    -------
    `list`
        a new list with the small/dot contours removed.
    """
    newContours = []
    areas = []

    for c in contours:
        #Calculate area
        area = cv2.contourArea(c)
        areas.append(area)

        #Add
        newContours.append(c)
    
    areas = np.array(areas)
    meanArea = np.mean(areas)

    #meanArea = statistics.mean(areas) <-- OLD

    #Filtering
    filteredContours = list(filter(lambda x: cv2.contourArea(x) >  meanArea/3, newContours))

    #Returning
    return filteredContours
def RemoveEgdeContours(contours:list,shape:np.ndarray,pixelIncrement:int = 10) -> list:
    """
    Description
    ----------
    Goes through contours and removes the ones closest to the egde that do not form a complete square.

    Parameters
    ----------
    `contours : list`
        the contours to be searched through.

    `shape : np.ndarray`
        the shape(x,y) of the image where the contours were derived from.
        
    `pixelIncrement : int, optional`
        the increment from the egde of the image that should be removed.

    Returns
    -------
    `list`
        a new list of contours with the egde cases removed.
    """
    #Assuming y = 1104, x = 1176, will be removing by default 10 pixels in
    
    #Shape of image
    xMax,yMax = shape

    new_contours = [] 

    #No idea, followed: https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
    for c in contours:
        #Indications whether or not a contour should be added or not
        removalMark = False
        
        #Dimensions of the contour
        x,y,w,h = cv2.boundingRect(c)

        #Checking all points
        removalMark = (x < pixelIncrement or (xMax - pixelIncrement) < x ) or (y < pixelIncrement or (yMax - pixelIncrement) < y) 
        if(removalMark is True): continue

        #Adding the width and height the get the next point
        x+=w
        y+=h
        
        #Checking again
        removalMark = (x < pixelIncrement or (xMax - pixelIncrement) < x ) or (y < pixelIncrement or (yMax - pixelIncrement) < y)
        if(removalMark is True): continue

        #Add if it was never marked for removal
        new_contours.append(c)

    return new_contours
def Contouring(imObj:ImageObject,doThreshold:bool=True) -> list:
    """
    Description
    ----------
    Takes an image and returns contours derived from it.

    
    Extended Summary
    ----------
    All contours are derived and then egde and small contours are removed,\n
    leaving only whole contours that fill the grid.

    Parameters
    ----------
    `imObj : ImageObject`
        the image object(`imObj`) which the contours are derived from.

    `doThreshold : bool`, `optional`
        if the method should do the thresholding.

    Returns
    -------
    `list`
        list of contours
    """
    #As the methods used actually change the orignal image a copy will be created instead
    im_clone = imObj.image.copy()

    #Threshold created with values determine from canny egde testing
    #Find contours - values chosen with canny edge detection in mind - might need adjustment
    if(doThreshold):
        ret, threshold = cv2.threshold(im_clone,128,255,0)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(im_clone,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    #remove contours at the sides of the image that dont form a complete square
    contours = RemoveEgdeContours(contours=contours,shape=imObj.shape.copy())

    #remove small unwanted contours that are not at the egde
    contours = RemoveDots(contours)

    return contours
def AdjustToGrid(contours:list,axis:int=1,limit:int=15) -> tuple[list,int]:
    """
    Description
    ----------
    Takes an unordered list of contours and orders them based on x,y and axis.

    Parameters
    ----------
    `contours : list`
        the unordered list of `contours`

    `axis : int`, `optional` 
        the axis at which to order the the contours, for x-axis = 1, for y-axis = 0
    
    `limit : int`, `optional`
        the `limit` is used to combat variations along the x and y axis of the contours.\n
        For `axis` = 0 then `limit` = 25 is recommended. \n
        For `axis` = 1 then `limit` = 15 is recommended.

    Returns
    -------
    `tuple[list,int]`
        a tuple with the ordered list of contours and a int for amount of valid rows
    """
    #Create empty numpy array that fits the amount of contours
    coords = np.empty((len(contours),2),dtype=np.int32)
    coords.fill(0)


    #List to hold x and y values and index will match contours
    indexList: list([int,int]) =  []

    #Save coords
    for i in range(len(contours)):
        x,y = contours[i][0][0]
        coords[i] = x,y
        indexList.append([x,y])

    #Change some pixel values to make them fit eachother, some values are next to eachother on the grid but is slighty higher or lower by one/two pixels
    #Which messes up the ordering of the grid contours
    
    #Where to start the change
    startChange = 0
    
    #Where to apply change x = 0 and y = 1, this changes the sorting order and limit has to accounted. Try limit = 20 for y and limit = 5 for x
    axis = axis

    #Where to end the change
    endChange = 0
    
    #Applied change
    change = 0
    
    #Limit to determine which values to apply change to
    limit = limit
    
    #Index to make it adjustable outside of the loops influence
    index = 0

    #Rows that are valid in the grid - used for another method so it might aswell be returned now instead of trying to find it again
    rows = 0

    #Sort coords based on y and x
    if(axis == 0):
        #Sort for x first then y
        sortedCoords = sorted(list(coords), key=lambda k: [k[0], k[1]])
    else:
        #Sort for y first then x
        sortedCoords = sorted(list(coords),key=lambda k: [k[1],k[0]])


    #Extra column to hold old and new.       
    #Old have positions x=[0] and y=[1], new have x=[2] and y=[3]
    extraColumn = np.empty((len(sortedCoords),2),dtype=np.int32)
    extraColumn.fill(0)

    #Keep old values as they are needed to reverse the change after the sorting. We cant match the values with a contour if we dont reverse.
    reverseCoords = np.append(sortedCoords.copy(),extraColumn,1)


    for i in range(len(sortedCoords)):
        #Take out the first value and check all the values after it is still within a limit
        value = sortedCoords[index][axis]

        #Where change is starting
        startChange = index

        #upper limit for y values that are accepted
        upperLimit = value + limit

        #lower limit for y values that are accepted
        lowerLimit = value - limit

        #New loop to search ahead in the current 2d array and determine an endChange
        #Set index to our know startChange plus 1 as the precise startChange is our baseline  
        for j in range((startChange + 1),len(sortedCoords)):
            
            #Evaluate
            eval = sortedCoords[j][axis] 
            if(not (lowerLimit <= eval & eval <= upperLimit) or j == len(sortedCoords)-1):
                #If here then values have exceeded limit
                    
                #j was exceeded and therefore the value just before was inside, this will be handled by range()
                endChange = j
                break

        #redundant but name makes more sense at this point
        change = value

        #Sequence
        seq = range(startChange,endChange)

        #loop through startChange until endChange and apply change
        for j in seq:
            
            #Change y value
            sortedCoords[j][axis] = change

            #Save new value
            #x
            reverseCoords[j][2] = sortedCoords[j][0]
            #y
            reverseCoords[j][3] = sortedCoords[j][1]

        #Set next starting point and apply rows for i=0
        if(i == 0):
            rows = endChange - 1
        startChange = endChange
        index = startChange


        #Check if changes has reached its end as we dont always know the size of the grid before hand, so we use the whole length of the 2d array
        if(startChange == len(sortedCoords)-1):

            #Last number is an egde case 
            sortedCoords[index][axis] = change

            #Save new value
            #x
            reverseCoords[index][2] = sortedCoords[index][0]
            #y
            reverseCoords[index][3] = sortedCoords[index][1]

            break

    #Sort again based on y and x
    if(axis == 0):
        #Sort for x first then y
        correctSortedCoords = sorted(list(sortedCoords), key=lambda k: [k[0], k[1]])
    else:
        #Sort for y first then x
        correctSortedCoords = sorted(list(sortedCoords),key=lambda k: [k[1],k[0]])
    
    correctSortedCoords = np.array(correctSortedCoords)


    #Reverse
    for i in range(len(reverseCoords)):
        #Since correctSortedCoords has been changed, we need to change it back by comparing new values with eachother and overwrite with the old
        #Therefore will the order remain the same but the values will change back and then we can compare the values back to their orignal contour

        #Extract values - can be improved
        oldX = reverseCoords[i][0]
        oldY = reverseCoords[i][1]
        newX = reverseCoords[i][2]
        newY = reverseCoords[i][3]

        #Compare
        for index in range(len(correctSortedCoords)):
            
            if(correctSortedCoords[index][0] == newX and correctSortedCoords[index][1] == newY):
                #Change back
                correctSortedCoords[index][0] = oldX
                correctSortedCoords[index][1] = oldY
                break
       

    #Clone/Copy of orignal array
    contourCopy = contours.copy()

    #Create new contour array but in order
    for i in range(len(sortedCoords)):
        #Extract every array
        x = correctSortedCoords[i][0]
        y = correctSortedCoords[i][1]

        #Search index and save index
        sortingIndex = indexList.index([x,y])

        #Move all information from one contour to the correct place instead of the first x and y only
        contourCopy[i] = contours[sortingIndex]
    
    return contourCopy,rows
def CalcShiftPX(pixelLength:float,gridMillimeterLength:float,realPixelLength:float) -> float:
    """
    Description
    ----------
    Translates a pixel length to a pixel shift using a known pixel size. 

    Parameters
    ----------
    `pixelLength : float`
        the amount of pixels to be translated to a pixel shift.

    `gridMillimeterLength : float`
        the length of the grid-squares in the real world in millimeters.    

    `realPixelLength : float`
        the size of a pixel in the real world in millimeters.

    Returns
    -------
    `float`
        the calculated shift.
    """
    #units:
    #         px       *        mm
    #         --------------------  = px?
    #                  mm
    return (pixelLength * realPixelLength) / gridMillimeterLength
def FindMagnification(imObj:ImageObject) -> float: 
    """
    Description
    ----------
    Determines the magnification of an image by assuming it is a mesh/grid image.

    Parameters
    ----------
    `imObj : ImageObject`
        the image object(`imObj`) that will be used to calculated the magnification.
       
    Returns
    -------
    `float`
        the calculated magnification
    """
    #Threshold
    medianblur = cv2.medianBlur(imObj.image.copy(),5)
    invMedianBlur = invert(medianblur)
    _,binInvBlur = cv2.threshold(invMedianBlur,200,255.0,cv2.THRESH_BINARY)

    #Resize
    shape = imObj.shape * imObj.settings.SCALE
    resizedImg = cv2.resize(binInvBlur,shape,interpolation=cv2.INTER_LINEAR)

    #Insert into image object
    resizedImg = ImageObject(resizedImg,imObj.distance,imObj.settings)

    #Define squares with contours
    contours = Contouring(resizedImg,doThreshold=False)

    #Sort
    sortedContours,_ = AdjustToGrid(contours,axis=0,limit=25*resizedImg.settings.SCALE)
    
    #Find center point for each sqaure
    points = [Point(*np.mean(sc,axis=0)[0]) for sc in sortedContours]

    #Map points so they have references to eachother
    MapPoints(points,imObj)

    #Use points to create big squares formed from 4 center points together
    sq = CreatingSquares(points)
    
    #Save centerPoints between intersections
    centerPoints = [s.GetCenterPoint() for s in sq] 

    #Fit sqaure
    MapPoints(centerPoints,imObj)
    
    #Fits a square on the image with a certain amount of rows and columns
    vertices,rows,columns = FitSquare(centerPoints,resizedImg)

    #Create blank
    blank = resizedImg.image.copy() * 0

    #Convert to numpy to use method
    vertices_np = np.array(vertices)

    #Fill sqaure
    b = blank.copy() * 0
    cv2.fillPoly(b,[vertices_np],255)

    #Show points and their number
    j = 0
    for p in vertices:  
        cv2.circle(blank,p,0,(125,125,125),7*resizedImg.settings.SCALE)
        cv2.putText(blank,f"{j}",p,1,resizedImg.settings.SCALE*2,(125,125,125),4)
        j += 1

    #Calculate area
    area = cv2.contourArea(vertices_np)

    #Scale down
    areaScaledDown = area / (resizedImg.settings.SCALE * resizedImg.settings.SCALE)

    #Calculate amount of squares
    amountOfSquares = rows*columns
    if(amountOfSquares == 0):
        amountOfSquares = columns

    #Calculate average Area per sqaure
    averageAreaPrSqr = areaScaledDown/amountOfSquares

    #Calculate length from averageAreaPrSqr
    length = np.sqrt(averageAreaPrSqr) 

    #Calculate magnification for specific image
    magnification = (length * resizedImg.settings.SINGLE_PIXEL_SIZE_MILLIMETER) / resizedImg.settings.GRID_MILLIMETER_LENGTH # +-1

    #Add length and magnification to calculate dsd later
    imObj.SetMagnification(magnification)

    #ShowImages([blank,b])

    return magnification
def FindTranslation(pointIm1:Point,pointIm2:Point) -> np.ndarray:
    """
    Description
    ----------
    Finds a translation(difference) between two points(x,y), assuming that they are comparable.

    Parameters
    ----------
    `pointIm1 : Point`
        point object that contains `x` and `y` for image 1

    `pointIm2 : Point`
        point object that contains `x` and `y` for image 2

    Returns
    -------
    `np.ndarray`
        a 3x2 numpy array that can be applied directly as a warp matrix
    """
    #Create output warp matrix
    matrix = np.eye(2,3,dtype=np.float32)

    #break down points into respective x and y values
    xIm1,yIm1 = [pointIm1.x,pointIm1.y]
    xIm2,yIm2 = [pointIm2.x,pointIm2.y]

    #Calculate translation
    xTranslation = xIm2 - xIm1
    yTranslation = yIm2 - yIm1

    #Insert into matrix
    matrix[0,2] = xTranslation
    matrix[1,2] = yTranslation

    return matrix
def AddSpacePrint() -> None:
    """
    Description
    ----------
    Adds 5 spaces to making printing look nicer

    Returns
    -------
    `None`
    """
    for i in range(3):
        print("")
def CreatingSquares(points:list[Point]) -> list[Square]:
    """
    Description
    ----------
    Takes a series of points, finds 4 points that make out a square and returns all squares found.
    
    Parameters
    ----------
    `points : list[Point]`
        the list of points that are going to be used to create squares.  

    Returns
    -------
    `list[Square]`
        returns the list of all the created squares.
    """
    #Using built in point left, right, above and under to identitfy squares 
    sq = []

    #Go through every point and try to create a sqaure from the information inside the point object
    for p in points:
        #p is always upperLeftCorner
        #below p is always lowerLeftCorner

        #right for p is always upperRightCorner
        #under "right" is always lowerRightCorner

        #Points at the egdes might not have below or right points and therefore will throw an exception
        upperLeftCorner = p 

        #Check under
        if (p.under == None): continue
        lowerLeftCorner = p.under

        #Check right
        if (p.right == None): continue
        upperRightCorner = p.right

        #Check under right
        if(upperRightCorner.under == None): continue
        lowerRightCorner = upperRightCorner.under


        #Create sqaure
        square = Square(upperRightCorner,upperLeftCorner,lowerRightCorner,lowerLeftCorner)
        
        #Append
        sq.append(square)

    return sq
def FitSquare(centerPoints:list[Point],imObj:ImageObject) -> (list[(int,int)],int,int):
    """
    Description
    ----------
    Takes a series of points and maps them to eachother with built-in directions.
    
    Parameters
    ----------
    `points : list[Point]`
        the list of points that are going to be mapped to eachother. 

    `imObj : ImageObject`
        the image object that the points are derived from to map according to distance and scale. 

    Returns
    -------
    `None`
        returns none, the `Point` attributes with directions are changed inside the list.
    """
    #Lists to hold the points that will form the different lines
    UpperHoriLine:list[Point] = []
    LowerHoriLine:list[Point] = []
    LeftVertiLine:list[Point] = []
    RightVertiLine:list[Point] = []

    #Starting point
    p:Point = centerPoints[0]

    #Counter to determine best rows
    outerCounter = 0
    innerCounter = 0

    #Starting with upper horizontal line 
    # - outer
    p1:Point = centerPoints[0]
    
    #Check
    if(p1 != None):
        while(p1.right != None):
            p1 = p1.right
            outerCounter+=1
        outerCounter+= 1

    # - inner
    p2:Point = centerPoints[0].under
        
    #Check
    if(p2 != None):
        while(p2.right != None):
            p2 = p2.right
            innerCounter+=1
        innerCounter +=1
    
    #Evaluate the inner and outer to eachother and use
    if(outerCounter < innerCounter):
        #Use inner point as reference
        reference = centerPoints[0].under
        count = innerCounter            

    else:
        #Use outer point as reference
        reference = centerPoints[0]
        count = outerCounter


    #Count columns with for loop
    highPColumns:list[Point] = []
    pColumns:list[list[Point]] = []
    counterColumns = []

    #Add the highest point from every column
    for _ in range(count):
        highPColumns.append(reference)
        reference = reference.right
    
    #Go through every highest point for every columns and find the points under
    for point in highPColumns:
        p = point
        pColumn:list[Point] = []
        
        #Add points until there are no more points under
        while(p.under != None):
            p = p.under
            pColumn.append(p)

        #Insert point into the first position
        pColumn.insert(0,point)

        #Add ponts to a collective list
        pColumns.append(pColumn)

        #Add the length to another list to use mode
        counterColumns.append(len(pColumn))
    

    #Find mode
    mode = statistics.mode(counterColumns)

    #Cut off for the beginnig and ending, ex 1-5 where index 0 is remove. 1 through 5 must have the same amount of points as mode.
    startCutOff = 0
    endCutOff = len(pColumns) - 1

    #Evalaute every column by size
    #for i in range(0,endCutOff+1): #CHANGE TO WHILE for(i = 0; i < endCutOff+1; i++)
    i = 0
    while(i < endCutOff+1):
        
        #pColumn can be larger, the same or smaller.

        #the same -> do nothing
        
        #Larger
        if(counterColumns[i] > mode):
            
            #Take the difference and reduce pColumn until it matches the mode
            diff = counterColumns[i] - mode

            #Remove the last elements with indices
            pColumns[i] = pColumns[i][:-diff] #DO NOT IF THIS ACTUALLY WORKS

        #Smaller
        elif(counterColumns[i] < mode):

            #There two cases here. if i is at the cutcoff numbers then we remove that part of the list
            #If it is within the start and endcutoff then we reduce every element to fit with the current list
            
            #At the cutoff
            if(i == startCutOff or i == endCutOff):
                #Remove elements
                del pColumns[i]
                del counterColumns[i]

                #Set cutoff again
                startCutOff = 0
                endCutOff = len(pColumns) - 1
            
            #Inside cutoff
            else:
                #Reduce the size of every pColumn list from 
                for j in range(len(pColumns)):
                    
                    while(len(pColumns[j]) > counterColumns[i]):
                        del pColumns[j][-1]

                    counterColumns[j] = len(pColumns[j])
                
                #Readjust mode
                mode = statistics.mode(counterColumns)
        
        i += 1

    ##################################### EXTRAPOLATION ROWS ########################################################
    #Amount of columns determines the amount of points on each row there is
    IsRowsExtrapolated = False
    if(len(pColumns) > 2): 
        #Since we dont have a list for rows like we do for columns, we have to create it -> right to left 

        #Creating rows list from columns - the mode is equal to how many rows there are
        pRows:list[list[Point]] = []
        
        for i in range(mode):
            
            pRow:list[Point] = []

            for pc in pColumns:
                pRow.append(pc[i])

            pRows.append(pRow)

        #Create fits based on points being columns or rows -> Rows
        linearFits = CreateFits(pRows,imObj,RowsAndColumns.ROWS)

        #Extrapolate rows
        holderPoints = ExtrapolateRowsOrColumns(pRows,linearFits,centerPoints,RowsAndColumns.ROWS)
       
        ###################################### REDO MAPING(adjust points) ###########################################
        MapPoints(holderPoints,imObj)

        #Extrapoled has been done and therefore redoing 
        IsRowsExtrapolated = True
        
        #Use mode to redo columns later on - can be done with min()
        modeRows = min(len(x) for x in pRows)

    ##################################### EXTRAPOLATION COLUMNS ########################################################
    #Only makes sense to do fit if columns have more than 2 points
    if(mode > 2):
        
        #Redo columns since they have been extrapolted
        if(IsRowsExtrapolated):
            #Clear Columns()
            pColumns.clear()

            #Creating columns list from rows - the mode is equal to how many columns there are
            for i in range(modeRows):
            
                pColumn:list[Point] = []

                for pr in pRows:
                    pColumn.append(pr[i])

                pColumns.append(pColumn)
            

        #Create fits based on points being columns or rows -> Columns
        linearFits = CreateFits(pColumns,imObj,RowsAndColumns.COLUMNS)

        #Extrapolate columns
        holderPoints = ExtrapolateRowsOrColumns(pColumns,linearFits,centerPoints,RowsAndColumns.COLUMNS)
       
        ###################################### REDO MAPING(adjust points) ###########################################
        MapPoints(holderPoints,imObj)

    
    #Find corners
    #First column used to extract left side corner
    upperLeftCorner:Point = pColumns[0][0]
    lowerLeftCorner:Point = pColumns[0][-1]

    #Last column used to extract right side corner
    upperRightCorner:Point = pColumns[-1][0]
    lowerRightCorner:Point = pColumns[-1][-1]


    #Only the horizontal lines should be formed as the columns is already formed - could do this before the corners
    LeftVertiLine = pColumns[0][::-1]
    RightVertiLine = pColumns[-1]


    #Upper horizontal line
    UpperHoriLine = []
    p = upperLeftCorner
    UpperHoriLine.append(p)
    while(p.right != None):
        p = p.right
        UpperHoriLine.append(p)

        #Check if p is the end corner, as a point to the right of the corner can be cutoff
        if(p == upperRightCorner): break


    #Lower horizontal line
    LowerHoriLine = []
    p = lowerRightCorner
    LowerHoriLine.append(p)
    while(p.left != None):
        p = p.left
        LowerHoriLine.append(p)

        #Check if p is the end corner, as a point to the right of the corner can be cutoff
        if(p == lowerLeftCorner):break

    assert (len(UpperHoriLine) == len(LowerHoriLine) and len(RightVertiLine) == len(LeftVertiLine))

    print(len(UpperHoriLine)-1,len(LowerHoriLine)-1,len(RightVertiLine)-1,len(LeftVertiLine)-1)
    
    #Combine lists
    points = UpperHoriLine + RightVertiLine + LowerHoriLine + LeftVertiLine


    #Removing duplicates with dict and turning back to list 
    points = list(dict.fromkeys(points))


    #Break down points object into ints
    vertrices = []
    for point in points:
        vertrices.append((int(point.x),int(point.y)))

    #Return
    rows = len(RightVertiLine)-1
    columns = len(UpperHoriLine)-1
    return vertrices,rows,columns
def MapPoints(points:list[Point],imObj:ImageObject) -> None:
    """
    Description
    ----------
    Takes a series of points and maps them to eachother with built-in directions.
    
    Parameters
    ----------
    `points : list[Point]`
        the list of points that are going to be mapped to eachother. 

    `imObj : ImageObject`
        the image object that the points are derived from to map according to distance and scale. 

    Returns
    -------
    `None`
        returns none, the `Point` attributes with directions are changed inside the list.
    """
    #Extract scale and distance
    distance = imObj.distance
    scale = imObj.settings.SCALE

    #Variance is for how many pixels we go out to still determine a point as part of a line
    #Is done with this linear fit: Y = 0.3576*X + 41.00
    pxDistance = 0.3576 * distance + 41

    pxDistance *=scale

    variance = pxDistance * 0.35

    #go through every point
    for center in points:
        
        #Center coords 
        x = center.x
        y = center.y

        #Limit to determine same vertical line
        upperXLimit = x + variance
        lowerXLimit = x - variance

        #Limit to determine same horizontal line
        upperYLimit = y + variance
        lowerYLimit = y - variance

        #Check if above has already been set
        if(center.above == None):
            #Find above - comparing y values and checking with x ####################################
            threshold = 9999
            candidate = None
            for neighbor in points:

                #Skip if center and neighbor is the same
                if(center == neighbor): continue    

                Nx = neighbor.x
                Ny = neighbor.y

                #If difference is a positive number, it means that neighbor is above
                if(y - Ny > 0):

                    #Check that neighbor and center is on the same vertical line
                    if(lowerXLimit < Nx and Nx < upperXLimit):
                        
                        #Lastly check if its a lower number than the threshold that came before it
                        if(threshold > y - Ny):
                            candidate = neighbor
                            threshold = y - Ny

            #Adding the candidate whether its None or a Point
            center.AddNeighboringPoints(candidate,Direction.ABOVE)

        #Check if under has already been set
        if(center.under == None):
            #Find under - comparing y values and checking with x ####################################
            threshold = -9999
            candidate = None
            for neighbor in points:

                #Skip if center and neighbor is the same
                if(center == neighbor): continue    

                Nx = neighbor.x
                Ny = neighbor.y

                #If difference is a negative number, it means that neighbor is under
                if(y - Ny < 0):

                    #Check that neighbor and center is on the same vertical line
                    if(lowerXLimit < Nx and Nx < upperXLimit):
                        
                        #Lastly check if its a higher number than the threshold that came before it
                        #We want the negative number closest to zero
                        if(threshold < y - Ny): 
                            candidate = neighbor
                            threshold = y - Ny

            #Adding the candidate whether its None or a Point
            center.AddNeighboringPoints(candidate,Direction.UNDER)
        
        #Check if left has already been set
        if(center.left == None):
            #Find left - comparing x values and checking with y ####################################
            threshold = 9999
            candidate = None
            for neighbor in points:

                #Skip if center and neighbor is the same
                if(center == neighbor): continue    

                Nx = neighbor.x
                Ny = neighbor.y

                #If difference is a positive number, it means that neighbor is to the left
                if(x - Nx > 0):

                    #Check that neighbor and center is on the same horizontal line
                    if(lowerYLimit < Ny and Ny < upperYLimit):
                        
                        #Lastly check if its a lower number than the threshold that came before it
                        if(threshold > x - Nx):
                            candidate = neighbor
                            threshold = x - Nx

            #Adding the candidate whether its None or a Point
            center.AddNeighboringPoints(candidate,Direction.LEFT)

        #Check if right has already been set
        if(center.right == None):
            #Find right - comparing x values and checking with y ####################################
            threshold = -9999
            candidate = None
            for neighbor in points:

                #Skip if center and neighbor is the same
                if(center == neighbor): continue    

                Nx = neighbor.x
                Ny = neighbor.y

                #If difference is a negative number, it means that neighbor is under
                if(x - Nx < 0):

                    #Check that neighbor and center is on the same vertical line
                    if(lowerYLimit < Ny and Ny < upperYLimit):
                        
                        #Lastly check if its a higher number than the threshold that came before it
                        #We want the negative number closest to zero
                        if(threshold < x - Nx): 
                            candidate = neighbor
                            threshold = x - Nx

            #Adding the candidate whether its None or a Point
            center.AddNeighboringPoints(candidate,Direction.RIGHT)
def ProcessDSD(imageObjects:list[ImageObject]) -> tuple[float,float,float]:
    """
    Description
    ----------
    Takes a series of image objects and calculates DSD, DSD variance and DSD standard deviation.
    
    Parameters
    ----------
    `imageObjects : list[ImageObject]`
        image objects where the DSD is going to be derived from. 

    Returns
    -------
    `tuple[float,float,float]`
        a tuple with 3 numbers being DSD, DSD variance and DSD standard deviation in that order.
    """
    #1. Find Magnification of both images, which is returned as a float
    #Lengths used to do the calculations
    lengths = []

    zeroSquareElements = []
    
    for k in range(len(imageObjects)):    
        #FindMagnification(imageObjects[k],SINGLE_PIXEL_SIZE,GRID_MM_LENGTH) <--- OLD

        #Threshold
        medianblur = cv2.medianBlur(imageObjects[k].image.copy(),5)
        invMedianBlur = invert(medianblur)
        _,binInvBlur = cv2.threshold(invMedianBlur,200,255.0,cv2.THRESH_BINARY)

        #Resize
        shape = imageObjects[k].shape * imageObjects[k].settings.SCALE
        resizedImg = cv2.resize(binInvBlur,shape,interpolation=cv2.INTER_LINEAR)

        #Insert into image object
        resizedImg = ImageObject(resizedImg,imageObjects[k].distance,imageObjects[k].settings)

        #Define squares with contours
        contours = Contouring(resizedImg,doThreshold=False)

        #Sort
        sortedContours,_ = AdjustToGrid(contours,axis=0,limit=25*imageObjects[k].settings.SCALE)

        #region points for loop
        # #Find point for each sqaure
        # points = []
        # for j in range(len(sortedContours)):
        #     x,y = np.mean(sortedContours[j],axis=0)[0]

        #     points.append(Point(x,y))

        #     cv2.circle(resizedImg.image, (int(x),int(y)), radius=0, color=(255, 255, 255), thickness=7*scale)
        #endregion
        
        #Find center point for each sqaure
        points = [Point(*np.mean(sc,axis=0)[0]) for sc in sortedContours]

        #Map points so they have references to eachother
        MapPoints(points,imageObjects[k])

        #Use points to create big squares formed from 4 center points together
        sq = CreatingSquares(points)

        #region centerPoints for loop
        # centerPoints = []
        # #Save center point
        # for j in range(len(sq)):
        #     intersec = sq[j].GetCenterPoint()

        #     centerPoints.append(intersec)

        #     #cv2.putText(resizedImg,f"{i}",(int(intersec.x)-50,int(intersec.y)),1,scale*2,(0,0,0),4)
        #     #print(intersec)

        #     cv2.circle(resizedImg.image,(int(intersec.x),int(intersec.y)),0,(125,125,125),7*scale)
        #endregion
        
        #Save centerPoints between intersections
        centerPoints = [s.GetCenterPoint() for s in sq] 

        #Fit sqaure
        MapPoints(centerPoints,imageObjects[k])

        #Fits a square on the image with a certain amount of rows and columns
        vertices,rows,columns = FitSquare(centerPoints,resizedImg)

        #Check if rows and columns are zero then continue
        if(rows == 0 or columns == 0):
            zeroSquareElements.append(imageObjects[k])
            continue

        #Create blank
        blank = resizedImg.image.copy() * 0

        #Convert to numpy to use method
        vertices_np = np.array(vertices)

        #Fill sqaure
        b = blank.copy() * 0
        cv2.fillPoly(b,[vertices_np],255)

        #Show points and their number
        j = 0
        for p in vertices:  
            cv2.circle(blank,p,0,(125,125,125),7*imageObjects[k].settings.SCALE)
            cv2.putText(blank,f"{j}",p,1,imageObjects[k].settings.SCALE*2,(125,125,125),4)
            j += 1

        #Calculate area
        area = cv2.contourArea(vertices_np)

        #Scale down
        areaScaledDown = area / (imageObjects[k].settings.SCALE * imageObjects[k].settings.SCALE)

        #Calculate amount of squares
        amountOfSquares = rows*columns
        if(amountOfSquares == 0):
            amountOfSquares = columns

        #Calculate average Area per sqaure
        averageAreaPrSqr = areaScaledDown/amountOfSquares

        #Calculate length from averageAreaPrSqr
        length = np.sqrt(averageAreaPrSqr) 

        #Calculate magnification for specific image
        magnification = (length * resizedImg.settings.SINGLE_PIXEL_SIZE_MILLIMETER) / resizedImg.settings.GRID_MILLIMETER_LENGTH # +-1

        #Printing information, can be removed
        print("k: ",k)
        print("row: ",rows)
        print("column: ",columns)
        print("area scaled down: " ,areaScaledDown)
        print("average area per sqr: ",averageAreaPrSqr)
        print("length: ", np.sqrt(averageAreaPrSqr))
        print("mag: ",magnification)
        #ShowImages([resizedImg.image,blank,b],[f"resized: {imageObjects[k].distance}","points","area"])
        AddSpacePrint()

        #Add length and magnification to calculate dsd later
        imageObjects[k].SetMagnification(magnification)
        lengths.append(np.round(length,3))


    #Remove zero square elements
    for imObj in zeroSquareElements:
        imageObjects.remove(imObj)


    #Baseline - baseline being the smallest distance
    baselineLength = lengths[0]
    baselineDistance = imageObjects[0].distance

    #Run through and calculate the dsd for each
    dsdHolder = []
    for j in range(1,len(imageObjects)):#Skip index=0
        dsd = (imageObjects[j].distance - baselineDistance)/((lengths[j] - baselineLength)/baselineLength) - baselineDistance
        dsdHolder.append(dsd)

    #=(88-48)/((72.08-57.68)/57.68)-48)
    #dsd = (88 - baselineDistance)/((72.08 - baselineLength)/baselineLength) - baselineDistance
    #       ^distance eval             ^length eval

    #Convert numbers to numpy array
    dsdHolder = np.array(dsdHolder)

    #Take median as reference for dsd
    dsd = np.median(dsdHolder)
    
    dsdVar = statistics.variance(dsdHolder)

    dsdSigma = np.sqrt(dsdVar)

    return dsd,dsdVar,dsdSigma
def CreateFits(pCR:list[list[Point]],imObj:ImageObject,rowsOrColumns:RowsAndColumns) -> list[LinearFit]:
    """
    Description
    ----------
    Create linear fits for rows or columns.
    
    Parameters
    ----------
    `pCR : list[list[Point]]`
        2d list where the first index indicate a row or column and the second index indicate the specific 
         point within the row or column.

    `imObj : ImageObject`
        image object where the rows or columns are derived from. 

    `rowsOrColumns : RowsAndColumns`
        enum that indicate whether `pCR` are rows or columns.    

    Returns
    -------
    `list[LinearFit]`
        list of the linear fit that matches the `pCR` list.
    """
    #Columns -> distanceUnder 
    #Rows -> distanceRight

    #List of fits and their information
    linearFits:list[LinearFit] = []

    #To hold x values
    xList:list[float] = []
    
    #To hold y values
    yList:list[float] = []

    #To hold distance so a step can be calculated
    step:list[float] = []

    #Depending on rowsOrColumns the approach is different
    if(rowsOrColumns == RowsAndColumns.COLUMNS):
        #Set axis
        axis = Axis.Y

        #Take each column and do linear regression
        for pcr in pCR:
            
            #Extract x,y and distance and place into list and convert to numpy array later
            for p in pcr:
                xList.append(p.x)
                yList.append(p.y)
                step.append(p.distanceToUnder)

            #Remove lastest step as it always 0.0 
            step.pop()

            #Place into numpy array - invert since its vertical - so: x = y • a + b
            x = np.array(xList) # x depends on y
            y = np.array(yList).reshape(-1, 1) # y does not depend on anything

            #Calculate median step to extrapolation
            medianStep = np.median(np.array(step))

            #Fit
            fit = linear_model.LinearRegression().fit(y,x)
            
            #Finding r
            r = fit.score(y,x) 

            #region show fit on image
            #Show on image
            # y_pred = np.array(list(range(0,imObj.shape[1]+1))).reshape(-1,1) #IS ACTUALLY Y VALUES ON CV2 PLOT AXIS
            # x_pred = fit.predict(y_pred) #IS ACTUALLY X VALUES ON CV2 PLOT AXIS
            # for ii in range(0,imObj.shape[1]+1):
            #     cv2.circle(imObj.image,(int(x_pred[ii]),int(y_pred[ii])),0,(0),1)
            # plt.plot(x_pred,y_pred) #show plot
            #endregion

            #Extract upper y value
            upEvalY = pcr[0].y.copy() #Apparently they are reference and not static...

            #Extract lower y value 
            lowEvalY = pcr[-1].y.copy() #Apparently they are reference and not static...

            #Create linearfit
            linearFits.append(LinearFit(fit,medianStep,upEvalY,lowEvalY,r,pcr,imObj.shape,axis))

            #Reset
            xList = []
            yList = []
            step = []
    
    elif(rowsOrColumns == RowsAndColumns.ROWS):
        #Set axis
        axis = Axis.X

        #Take each row and do linear regression
        for pcr in pCR:
            
            #Extract x,y and distance and place into list and convert to numpy array later
            for p in pcr:
                xList.append(p.x)
                yList.append(p.y)
                step.append(p.distanceToRight)

            #Remove lastest step as it is always 0.0 
            step.pop()

            #Place into numpy array - invert since its vertical - so: y = x • a + b
            y = np.array(yList) # y depends on x
            x = np.array(xList).reshape(-1, 1) # x does not depend on anything

            #Calculate median step to extrapolation
            medianStep = np.median(np.array(step))

            #Fit
            fit = linear_model.LinearRegression().fit(x,y)
            
            #Finding r
            r = fit.score(x,y) 

            #region show fit on image
            # x_pred = np.array(list(range(0,imObj.shape[0]+1))).reshape(-1,1) #IS ACTUALLY Y VALUES ON CV2 PLOT AXIS
            # y_pred = fit.predict(x_pred) #IS ACTUALLY X VALUES ON CV2 PLOT AXIS
            # for ii in range(0,imObj.shape[0]+1):
            #     cv2.circle(imObj.image,(int(x_pred[ii]),int(y_pred[ii])),0,(0),1)
            # plt.plot(x_pred,y_pred) #show plot
            #endregion

            #Extract upper x value
            upEvalY = pcr[0].x.copy() #Apparently they are reference and not static...

            #Extract lower y value
            lowEvalY = pcr[-1].x.copy() #Apparently they are reference and not static...

            #Create linearfit
            linearFits.append(LinearFit(fit,medianStep,upEvalY,lowEvalY,r,pcr,imObj.shape,axis))

            #Reset
            xList = []
            yList = []
            step = []

    return linearFits    
def ExtrapolateRowsOrColumns(pCR:list[list[Point]],linearFits:list[LinearFit],points:list[Point],rowsOrColumns:RowsAndColumns) -> list[Point]:
    """
    Extrapolate rows or columns based on linear fits.
    
    Parameters
    ----------
    `pCR : list[list[Point]]`
        2d list where the first index indicate a row or column and the second index indicate the specific point within the row or column.

    `linearFits : list[LinearFit]`
        linear fits that are derived from the rows or columns and element-wise should match the first index of `pCR`. 

    `points : list[Point]`
        list of the points that already are on image so extrapolated points can be added to it.

    `rowsOrColumns : RowsAndColumns`
        enum that indicate whether `pCR` are rows or columns.    

    Returns
    -------
    `list[Point]`
        a copy of `points` with the extrapolated points added to it.
    """
    #Go through each linear fit and check how many times we extrapolate upwards and downwards
    countsUp = []
    countsDown = []
    for fit in linearFits:
        countsUp.append(len(fit.availableUpper))
        countsDown.append(len(fit.availableLower))

    #Use set and take lowest number as reference to how many times we can extrapolate in a uniform manner
    minUp = min(countsUp) 
    minDown = min(countsDown)

    #Holder to add new points to and to do remaping
    holderPoints = points.copy()

    #Process depends on if its rows or columns
    if(rowsOrColumns == RowsAndColumns.COLUMNS):

        #Do for both upper and lower
        for i in range(len(linearFits)):
            
            #Do for upper (or lower if you consider actual y values and not the image) 
            for ii in range(minUp):
                
                #Take prediction out of linearFit
                newPoint = linearFits[i].availableUpper[ii]

                #Insert into specific column - make first element
                pCR[i].insert(0,newPoint)
                
                #holder Points to remap
                holderPoints.append(newPoint)
            
            #Do for lower (or upper if you consider actual y values and not the image)
            for ii in range(minDown):

                #Take prediction out of linearFit
                newPoint = linearFits[i].availableLower[ii]

                #Insert into specific column - make last
                pCR[i].append(newPoint)
                
                #holder Points to remap
                holderPoints.append(newPoint)

    elif(rowsOrColumns == RowsAndColumns.ROWS):
        #Do for both upper and lower
        for i in range(len(linearFits)):
            
            #Do for upper (or lower if you consider actual y values and not the image) 
            for ii in range(minUp):
                
                #Take prediction out of linearFit
                newPoint = linearFits[i].availableUpper[ii]

                #Insert into specific row - make first element
                pCR[i].insert(0,newPoint)
               
                #holder Points to remap
                holderPoints.append(newPoint)
            
            #Do for lower (or upper if you consider actual y values and not the image)
            for ii in range(minDown):

                #Take prediction out of linearFit
                newPoint = linearFits[i].availableLower[ii]

                #Insert into specific row - make last
                pCR[i].append(newPoint)

                #holder Points to remap
                holderPoints.append(newPoint)
    
    return holderPoints
def CreateMagnificationMatrix(imObj1:ImageObject,imObj2:ImageObject) -> np.ndarray:
    """
    Takes two `ImageObject` and finds the magnification matrix to match the two images magnification-wise.
    
    Parameters
    ----------
    `imObj1 : ImageObject`
        the image object to be scaled to match `imObj2`.

    `imObj2 : ImageObject`
        the image object that `imObj1` has to match magnification-wise. 

    Returns
    -------
    `np.ndarray`
        the translation matrix to overlap `imObj1Scaled` with `imObj2`.
    """
    #Percentage difference between two images
    percentage = ((imObj2.magnification-imObj1.magnification)/imObj1.magnification)

    #DSD = FindDSD(percentage,Distances[ii]-Distances[i]) <- old

    #Use the percentage change to apply a affine matrix to change the scale and at the same time keep the image centered (for image1)
    appliedScale = np.round(percentage,3)

    #Create identity matrix
    #Positions for scale are x = [0,0] and y = [1,1]. Positions for translation are x = [0,2] and y = [1,2]
    warpScaleMatrix = np.eye(2,3,dtype=np.float32)

    #Scale
    warpScaleMatrix[0,0] += appliedScale
    warpScaleMatrix[1,1] += appliedScale

    #Apply percentage change to the x and y so it can used in the translation part of the matrix and keep it centered
    xApplied = imObj1.x * appliedScale/2
    yApplied = imObj1.y * appliedScale/2

    #translation - this reason for it being -= is because we are always going from small to larger distance, so 0 mm to 200 mm which will always require 
    #a scaling up and therefore the y and x will increase and we mitigate this by subtracting x/y applied
    #If we reverse the process so 200 mm to 0 mm then we would be scaling down and therefore += is needed
    warpScaleMatrix[0,2] -= xApplied
    warpScaleMatrix[1,2] -= yApplied

    return warpScaleMatrix
def ProcessShift(imObj1Scaled:ImageObject,imObj2:ImageObject) -> np.ndarray:
    """
    Takes two `ImageObject` and finds a common spot between them and process the shift needed to overlap the two images.
    
    Parameters
    ----------
    `imObj1Scaled : ImageObject`
        the image object scaled to match `imObj2` and shift will overlap `imObj1Scaled` with `imObj2`.

    `imObj2 : ImageObject`
        the image object that `imObj1Scaled` has to match. 

    Returns
    -------
    `np.ndarray`
        the translation matrix to overlap `imObj1Scaled` with `imObj2`.
    """
    #With the imaged up/down the next step is to find the offset(translation) and magnification(scale) that will change image1 to fit onto image2
    #Find comparable points for both im1 and im2
    pointIm1S = FindCenterHole(imObj1Scaled)
    pointIm2 = FindCenterHole(imObj2)

    #Find translation based on the given points
    warp_mat_calc = FindTranslation(pointIm1S,pointIm2)

    #Warp with the found translation
    imObj1Scaled.WarpSelf(warp_mat_calc)

    ShowImages([imObj1Scaled,imObj2,ImageObject(imObj2.image - imObj1Scaled.image,imObj2.distance,imObj2.settings)],[f"Scaled up image ({imObj1Scaled.distance} mm)",f"Image ({imObj2.distance} mm)",f"Difference between of the two images"])

    #Use the findtransform ecc to match the images and give a translation, it has a tougher time with large difference so we did some scaling up and translation for it
    #So it should only do a translation ideally which would be a small translation(adjustment only)

    #Define the motion model - allowing the method to use a complete MOTION_AFFINE messes up the final image as it would want to apply shear and scale 
    warp_mode = cv2.MOTION_AFFINE
    #warp_mode = cv2.MOTION_TRANSLATION
   
    #Define 2x3 matrix and initialize the matrix to identity matrix I (eye)
    #[1     0       0]
    #[0     1       0]
    warp_matrix = np.eye(2,3,dtype=np.float32)

    #specify the number of iterations 
    number_of_iterations = 5000

    #Sepcifiy the threshold of the increment
    #in the correlation coefficient between two iterations
    termination_eps = 1e-3

    #Define termination criteria
    criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    #Run the ECC algorithm. The results are stored in warp_matrix.
    #ECC = Enchanced Correlation Coefficient Maximization
    (cc, warp_matrix) = cv2.findTransformECC(imObj1Scaled.image,imObj2.image,warp_matrix,warp_mode,criteria,None,1)

    #Warp im1 using affine
    imObj1Scaled.WarpSelf(warp_matrix)
    #region how the warp matrix works:
    #Affine warp matrix: 
    #                     A = [a_00  a_01]          B = [b_00]
    #                         [a_10  a_11]              [b_10]
    #
    #                     M = [A B] = [a_00  a_01  b_00]
    #                                 [a_10  a_11  b_10]
    #
    #                         scale x   shear y  translation x
    #                     T = [a_00*x   a_01*y   b_00]
    #                         [a_10*x   a_11*y   b_10]
    #                         shear x   scale y  translation y
    #
    #Translation warp matrix: [1  0  t_x]
    #                         [0  1  t_y]
    #endregion
    
    #ShowImages([im1ScaledUp.image,im2.image,im2.image - im1ScaledUp.image],[f"Aligned image ({im1ScaledUp.distance} mm)",f"Image ({im2.distance} mm)",f"Difference between of the two images"]) 

    AddSpacePrint()
    print("Calculated:")
    print(warp_mat_calc)
    AddSpacePrint()
    print("ECC:")
    print(warp_matrix)

    #Matrix to be used for calculations and the matrix returned
    ShiftMatrix = np.eye(2,3,dtype=np.float32)

    #Changing the matrix to be the final one 
    ShiftMatrix[0,2] = (warp_mat_calc[0,2] + warp_matrix[0,2])
    ShiftMatrix[1,2] = (warp_mat_calc[1,2] + warp_matrix[1,2])

    return ShiftMatrix
def CalcShiftMM(imObj1:ImageObject,imObj2:ImageObject,ShiftMatrix:np.ndarray,maxDistance:float,dsd:float):
    """
    Takes two `ImageObject` and calculates the real life shift to align the detector.
    
    Parameters
    ----------
    `imObj1 : ImageObject`
        the image object with smaller distance than `imObj2`.

    `imObj2 : ImageObject`
        the image object with larger distance than `imObj1`.

    `ShiftMatrix : np.ndarray`
        the shift matrix between the images.
    
    `maxDistance : float`
        the maximum distance in millimeter.    

    `dsd : float`
        the detector-source distance.   

    Returns
    -------
    `Tuple[float,float]`
        the x and y position wrapped in a tuple.
    """ 
    #Translate shift from pixels to mm
    xShift = CalcShiftPX(ShiftMatrix[0,2],imObj1.settings.GRID_MILLIMETER_LENGTH,imObj1.settings.SINGLE_PIXEL_SIZE_MILLIMETER) #Side opposite
    yShift = CalcShiftPX(ShiftMatrix[1,2],imObj1.settings.GRID_MILLIMETER_LENGTH,imObj1.settings.SINGLE_PIXEL_SIZE_MILLIMETER) #Side opposite 

    #Difference in distance
    distance = imObj2.distance - imObj1.distance #Adjacent

    #Use inverse tan to find angle 
    xAngle = np.arctan(xShift/distance)
    yAngle = np.arctan(yShift/distance)

    #Angle should apply to the triangle from the source to detector dsd/distance * x/y shift
    mmShiftX = (dsd/distance) * xShift
    mmShiftY = (dsd/distance) * yShift 

    #mmShiftX = (dsd + imObj1.distance + maxDistance) * np.tan(xAngle)
    #mmShiftY = (dsd + imObj1.distance + maxDistance) * np.tan(yAngle)
    
    # AddSpacePrint()
    # print("print x shift ",np.round(xShift,3))
    # print("print y ",np.round(yShift,3))
    # print("print distance ",np.round(distance,3))
    # print("angle x",np.round(xAngle,3))
    # print("angle y",np.round(yAngle,3))
    # print("mm shift x",np.round(mmShiftX,3))
    # print("mm shift y",np.round(mmShiftY,3))

    return [mmShiftX,mmShiftY] 
def FindCenterHole(imObj:ImageObject) -> Point:
    """
    Takes an `ImageObject` and finds the hole on the image through thresholding and return the x and y position.

    Parameters
    ----------
    `imObj : ImageObject`
        the image object with the hole on it.

    Returns
    -------
    `Point`
        point object with the x and y position.
    """
    _,thresh = cv2.threshold(imObj.image,250,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contourImage = cv2.drawContours(thresh.copy()*0,contours,-1,(125),1)

    #Worst case is 4 contours of the hole because it is split up
    
    #Find area and combine with the element to create tuple
    areasAndContours =  [[cv2.contourArea(item),item] for item in contours if cv2.contourArea(item) != 0.0]
    
    #Sort the tuple based on area, aac (areas and contours)
    aacSorted = sorted(areasAndContours, key=lambda k: [k[0]],reverse=True)

    #Since worst case is 4 parts of the hole on the image, we take out the 4 last elements
    aacSorted = aacSorted[:4]

    #Check to see what case we are working with, checking based on largest area and comparing with the 3 other
    #Tolarance 50%
    tolarance = aacSorted[0][0] - aacSorted[0][0] * 0.5

    #Filter out what is smaller than the tolarance
    filteredAC = list(filter(lambda x: x[0] > tolarance,aacSorted))
    
    #Go back to contours only, ac (areaContour)
    holeContours = [ac[1] for ac in filteredAC ] 

    #Draw on empty image to get contours
    filtered = cv2.drawContours(thresh.copy()*0,holeContours,-1,(255),1)

    #Fill contours on image
    cv2.fillPoly(filtered,holeContours,(255))

    #Find all the filled pixels
    indices = np.where(filtered == 255)

    #Take mean of indices    
    y = np.round(np.mean(indices[0],axis=0),0)
    x = np.round(np.mean(indices[1],axis=0),0)

    #Draw point 
    cv2.circle(filtered,(int(x),int(y)),0,(125),1)

    ShowImages([imObj,ImageObject(thresh,imObj.distance,imObj.distance),ImageObject(contourImage,imObj.distance,imObj.distance),ImageObject(filtered,imObj.distance,imObj.distance)],["before","thresh","contour","filtered contours"])

    return Point(x, y)
def MakeReader(fileType:str) -> ImageReader:
    """
    Factory method that takes a filetype and create the appropriate `ImageReader`.

    Parameters
    ----------
    `fileType : str`
        the file extension of the images.

    Returns
    -------
    `ImageReader`
        the interface with the ReadImage method that can read the input file extension.
    """
    #Create factory
    readerFactory = ReaderFactory()

    if(fileType == ".tif" or fileType == ".tiff"):
        #tif based
        reader = readerFactory.CreateReader(ReaderType.TIFF)

    elif(fileType == ".txt"):
        #txt based
        reader = readerFactory.CreateReader(ReaderType.TXT)

    elif(fileType == ".npy"):
        #numpy based
        reader = readerFactory.CreateReader(ReaderType.NPY)

    return reader
