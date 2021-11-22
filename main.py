
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from modules.path_planning.astar import AStar
import time
import random
import math
import os
from os.path import isfile,join

#Global variables to store the mouse position coordinates from the image
startX, startY, endX, endY = 0,0,0,0
selectStart = True
img = None

def select_point(event,x,y,flags,param):
    """
    Grabs points selected from the cv2 image shown on screen
    """
    global startX,startY,endX,endY,img,selectStart

    if event == cv2.EVENT_LBUTTONDBLCLK: # captures left button double-click
        ix,iy = x,y

        #Check to see if the user selected a point inside a road
        if not np.array_equal(img[iy][ix],[255,255,255]):
            print("Please select a valid point inside the road!")
            return
            
        #Set the start coordinate if selectStart is still true
        if selectStart:
            img = cv2.circle(img,(ix,iy),radius=5,color=(0,255,0),thickness=-1)
            selectStart = False
            startX,startY = ix,iy
            cv2.imshow('image',img)
            cv2.waitKey(10)
        #Set the endPoint and begin path planning computation
        else:
            img = cv2.circle(img,(ix,iy),radius=5,color=(0,0,255),thickness=-1)
            endX,endY = ix,iy
            cv2.imshow('image',img)
            cv2.waitKey(10)

            #Start the path planning algoirthm
            runPathPlanning()

            #Display the results to the user
            #Resize the image if needed
            #img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_AREA)
            cv2.imshow('image',img)
            cv2.waitKey(0)

        
def grabRandomImage():
    #Returns an image path to use for the path processing

    imgFolder = join(os.getcwd() + "\data\processedImages")
    imgs = [f for f in os.listdir(imgFolder) if isfile(join(imgFolder,f))]

    return join(imgFolder,random.choice(imgs))

def runPathPlanning(algoName="astar"):
    
    start = time.time()
    startPos = (startX,startY)
    endPos = (endX,endY)

    if algoName == "astar":

        #Run the pathfinding algorithm
        aStarAlgo = AStar(startPos,endPos,img)
        path = aStarAlgo.run()
    #else algoName == "LPA":
    #Future path finding algorithms go here..each should return a list of coordinates
    

    end = time.time()
    duration = end - start
    
    #Plot the path found onto the image
    solution_quality = 0.0
    map_path = []
    for pos in path:
        map_path.append(pos)
        cv2.rectangle(img,pos,(pos[0]+2,pos[1]+2),color=(0,255,0),thickness=-1)
        if pos != path[0]:
            solution_quality += math.hypot((map_path[-1][0] - map_path[-2][0]),(map_path[-1][1] - map_path[-2][1]))
        #img[pos[1]][pos[0]] = [0,255,0]

    print('Quality/Cost of solution: ', solution_quality)
    print("Time took: ",duration)
    



def main():
    global img 
    imgPath = grabRandomImage()

    #Load the image
    img = cv2.imread(imgPath)

    #Resize the image if desired (Speeds up computation)
    #img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
    #Convert the image to GrayScale
    #img = cv2.cvtColor(cv2.imread('path1.png'),cv2.COLOR_BGR2GRAY)

    #Initialize the image window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',select_point)


    #Display image and wait until the user selects two valid points on the image 
    cv2.imshow('image',img)
    cv2.waitKey(0) #The bulk of the program logic is run through the select_point() function that processes the mouse clicks from the user

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()