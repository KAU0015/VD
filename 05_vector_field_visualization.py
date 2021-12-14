from random import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

PATH = 'C:/Users/Moje/Desktop/VD/Python/flow_field'

class Point:

    def __init__(self, x_, y_, streak_ = False):
        self.x = float(x_)
        self.y = float(y_)
        self.streak = streak_

    def __add__(self, p: 'Point'):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p: 'Point'):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, c: float):
        return Point(self.x * c, self.y * c)


    def move(self, n_x_y):
        self.x = self.x + n_x_y[0]
        self.y = self.y + n_x_y[1]

        if(self.x < 0.0): self.x = 0.0
        if(self.x > 255.0): self.x = 255.0

        if(self.y < 0.0): self.y = 0.0
        if(self.y > 255.0): self.y = 255.0

    def rounded(self):
        x_t, y_t = np.int16(np.around((self.x, self.y)))
        return  y_t, x_t

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def to_array(self): 
        return [self.x, self.y]

def mag(a):
	return np.sqrt(pow(a[0], 2) + pow(a[1], 2))

def generatePoints(count):
    points = []
    for i in range(count):
        x = random() * 255.0
        y = random() * 255.0
        while(x >= 50 and x <= 195 and y >= 60 and y <= 105):
            x = random() * 255.0
            y = random() * 255.0
        points.append(Point(x, y))
    return points

def main():
    plt.ion()
    randomPoints = generatePoints(100)
    for time in range(0, 1000, 1):
        
        file_open = cv2.FileStorage(PATH + "/u" + "{:05d}".format(time) + ".yml", cv2.FILE_STORAGE_READ)
        file_node: int = file_open.getNode("flow")
        vector_mat = file_node.mat()

        print("File " + "{:05d}".format(time))

        # Draw to raster
        testRaster = []
        for y in range(256):
            testRaster.append([])
            for x in range(256):
                testRaster[y].append(mag(vector_mat[y, x]))

        # Plot raster
        plt.figure("Result")
        plt.imshow(testRaster, interpolation='nearest')
        plt.colorbar()
        # Cycle through every point and move it
        for x_t in randomPoints:
            # Find closest vector to point
            K1 = vector_mat[x_t.rounded()]

            K2_p_y = x_t.y + (K1[0] * 0.5)
            K2_p_x = x_t.x + (K1[1] * 0.5)
            K2_x, K2_y = np.int16(np.around((K2_p_x, K2_p_y)))
            K2 = vector_mat[K2_y, K2_x]

            K3_p_y = x_t.y + (K2[0] * 0.5)
            K3_p_x = x_t.x + (K2[1] * 0.5)
            K3_x, K3_y = np.int16(np.around((K3_p_x, K3_p_y)))
            K3 = vector_mat[K3_y, K3_x]

            K4_p_y = x_t.y + K3[0]
            K4_p_x = x_t.x + K3[1]
            K4_x, K4_y = np.int16(np.around((K4_p_x, K4_p_y)))
            K4 = vector_mat[K4_y, K4_x]

            next_pos = ((K1[0] + K2[0] + K3[0] + K4[0]) * 1.0 / 6.0, (K1[1] + K2[1] + K3[1] + K4[1]) * 1.0 / 6.0)
            
            x_t.move(next_pos)


        
        # Cycle through every point and draw it
        for point in randomPoints:
            if not point.streak: 
                plt.scatter(point.x, point.y, zorder=10000000, c="green", s=12)


        # Filter only streak points
        streakPoints = []
        for point in randomPoints: 
            if point.streak: 
                streakPoints.append(point)

        # Draw streak line
        for i in range(len(streakPoints) - 1):
            p = streakPoints[i]
            p_next = streakPoints[i + 1]
            plt.plot([p.x, p_next.x], [p.y, p_next.y], zorder=100000000, c="red")


        # Add next point to same spot
        randomPoints.append(Point(40.0, 250.0, True))

        plt.savefig(PATH + "_out/" + "{:05d}".format(time) + "_rk2.jpg")
        plt.draw()
        plt.pause(0.00001)
        plt.clf()


if __name__ == "__main__":    
    main()
