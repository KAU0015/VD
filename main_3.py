import numpy
import matplotlib.pyplot as plt
from random import seed
from random import random
import math
import functools
import cv2
from shapely.geometry import Polygon
from shapely.geometry import Polygon, Point

class MyPoint:
    def __init__(self, x, y):
	    self.x = float(x)
	    self.y = float(y)

    def getArray(self):
        return [self.x, self.y]

    def __mul__(self, c: float):
        return MyPoint(self.x * c, self.y * c)

    def __add__(self, p):
        return MyPoint(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return MyPoint(self.x - p.x, self.y - p.y)

class Quad:
    def __init__(self, points):
        self.points = points

    def getPoint(self, index):
        return self.points[index]

    def setPoint(self, index, point):
        self.points[index] = point

    

def drawSinCosFunction(plot, size = 10.0):
	xlist = numpy.linspace(-1.0, size, 100)
	ylist = numpy.linspace(-1.0, size, 100)
	X, Y = numpy.meshgrid(xlist, ylist)
	Z = numpy.sin(X) + numpy.cos(Y)
	plot.contourf(xlist, ylist, Z, levels=100)

def halfRandom():	
	return (random() - 0.5) * 0.2

def generateGridQuads(size): 
	points = []
	for y in range(size):
		for x in range(size):
			points.append(MyPoint(float(x) + halfRandom(), float(y) + halfRandom()))
				
	quads = []
	for y in range(0, size - 1, 1):
		for x in range(0, size - 1, 1):
			p1 = points[(y * size) + x]
			p2 = points[(y * size) + x + 1]
			p3 = points[((y + 1) * size) + x + 1]
			p4 = points[((y + 1) * size) + x]
			quads.append(Quad([p1, p2, p3, p4]))

	return quads

def renderPoints(plot, grid):
    for quad in grid:
        for p in quad.points:
            plot.scatter(p.x, p.y, zorder=10000000, c="black", s=8)

def renderQuads(plot, grid):
    for quad in grid:
        xs = []
        ys = []

        for p in quad.points:
            xs.append(p.x)
            ys.append(p.y)

        xs.append(xs[0])
        ys.append(ys[0])
        plot.plot(xs, ys, c="black")

def generateGridPoints(size = 10, step = 0.1): 
    points = []
    count = 0
    y = 0.0
    while y <= size - 0.1:
        x = 0.0
        while x <= size - 0.1:
            points.append(Point(x ,y))
            x += step
        y += step
        count += 1
  #  print(count)
    return points


def T_quad(r, s, quad):
	return (quad.points[0] * (1.0 - r) + quad.points[1] * r) * (1.0 - s) + (quad.points[2] * r + quad.points[3] * (1.0 - r)) * s

def J_matrix(r, s, quad):
	#				(s - 1.0) * (quad.p1 - quad.p2) + s * (quad.p3 - quad.p4)
	diffR = (quad.points[0] - quad.points[1]) * (s - 1.0) + (quad.points[2] - quad.points[3]) * s
	diffS = (quad.points[0] - quad.points[3]) * (r - 1.0) + (quad.points[2] - quad.points[1]) * r

	return numpy.array([
			[ diffR.x, diffS.x ], 
			[ diffR.y, diffS.y ], 
		])

def sinCos(point):
	return numpy.sin(point.x) + numpy.cos(point.y)

def newtonMetod(point, quad):
	
    r = 0.5
    s = 0.5

    iter = 0
    while iter < 20: #20 3
        iter += 1

        T_rs = T_quad(r, s, quad)
        J = J_matrix(r, s, quad)
        J_inv = numpy.linalg.inv(J)

        nextSolution = (r, s) - (J_inv @ (T_rs - p).getArray())
        
        r = nextSolution[0]
        s = nextSolution[1]
   # print(r)
    #print(s)
    return MyPoint(r, s)





CANVAS_SIZE = 10
PIXEL_DPI = 0.1
PIXEL_PER_ROW = int(CANVAS_SIZE / PIXEL_DPI)

plt.figure(1)

drawSinCosFunction(plt, CANVAS_SIZE)

quadsGrid = generateGridQuads(CANVAS_SIZE)


renderPoints(plt, quadsGrid)
renderQuads(plt, quadsGrid)

pixelGrid = generateGridPoints(CANVAS_SIZE, PIXEL_DPI)


rasterized = []
for p in pixelGrid:
    f_lambda = -2.0
    for quad in quadsGrid:
        poly = Polygon([quad.points[0].getArray(), quad.points[1].getArray(), quad.points[2].getArray(), quad.points[3].getArray()])
        if poly.contains(Point(p.x, p.y)):
            # Work the magic
            r, s = newtonMetod(p, quad).getArray()

            
            f_i_1 = sinCos(quad.points[0])
            f_i_2 = sinCos(quad.points[1])
            f_i_3 = sinCos(quad.points[2])
            f_i_4 = sinCos(quad.points[3])
            
            phi_1 = (1.0 - r) * (1.0 - s)
            phi_2 = r * (1.0 - s)
            phi_3 = r * s
            phi_4 = (1.0 - r) * s

            f_lambda = (f_i_1 * phi_1) + (f_i_2 * phi_2) + (f_i_3 * phi_3) + (f_i_4 * phi_4)

    rasterized.append(f_lambda)


PIXEL_PER_ROW = numpy.sqrt(len(pixelGrid))
rasterizedMN = []
for y in range(int(PIXEL_PER_ROW)):
    rasterizedMN.append([])
    for x in range(int(PIXEL_PER_ROW)):
        idx = y * PIXEL_PER_ROW + x
        rasterizedMN[y].append(rasterized[int(idx)])


plt.figure(2)
plt.imshow(rasterizedMN, cmap='viridis', interpolation='nearest')
	




plt.show()
