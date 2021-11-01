import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
import random


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

def genereateGrid(image, width, height, points):
    
    
    for x in range(40, width, 40):

        for y in range(40, height, 40):
            point = Point(random.randrange(-80, 80)/10 + x, random.randrange(-80, 80)/10 + y)   
            image[int(point.x)][int(point.y)] = (0, 0, 255)
            points.append(point)
    
   # pointsCount = points.size()
 #   pointsArray = [cv2.sqrt(pointsCount)][cv2.sqrt(pointsCount)]

    #for point in points:



            
    
points = list()
height = 400
width = 400
blank_image = np.zeros((height,width,3), np.uint8)
blank_image[:] = (255, 255, 255)

genereateGrid(blank_image, width, height, points)
img = cv2.cvtColor(blank_image, cv2.CV_32FC3)
cv2.imshow('Picture', img)


cv2.waitKey(0)
cv2.destroyAllWindows()