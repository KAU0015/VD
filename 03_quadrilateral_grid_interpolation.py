import numpy
import matplotlib.pyplot as plt
from random import random
from shapely.geometry import Polygon, Point

class MyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getArray(self):
        return [self.x, self.y]

    def __mul__(self, c):
        return MyPoint(self.x * c, self.y * c)

    def __add__(self, p):
        return MyPoint(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return MyPoint(self.x - p.x, self.y - p.y)

class Quad:
    def __init__(self, points):
        self.points = points


def drawSinCosFunction(size):
    xlist = numpy.linspace(-1.0, size, 100)
    ylist = numpy.linspace(-1.0, size, 100)
    x, y = numpy.meshgrid(xlist, ylist)
    z = numpy.sin(x) + numpy.cos(y)
    plt.figure("Grid")
    plt.contourf(xlist, ylist, z, levels=100, cmap='hot')

def renderPoints(quadsGrid):
    for quad in quadsGrid:
        for p in quad.points:
            plt.scatter(p.x, p.y, c="black", s=8)

def renderQuads(quadsGrid):
    for quad in quadsGrid:
        xs = []
        ys = []

        for p in quad.points:
            xs.append(p.x)
            ys.append(p.y)

        xs.append(xs[0])
        ys.append(ys[0])
        plt.plot(xs, ys, c="black")


def generateGridQuads(size): 
	points = []
	for y in range(size):
		for x in range(size):
			points.append(MyPoint(x + (random()-0.5) * 0.25, y + (random()-0.5) * 0.25))
				
	quads = []
	for y in range(0, size - 1, 1):
		for x in range(0, size - 1, 1):
			p1 = points[(y * size) + x]
			p2 = points[(y * size) + x + 1]
			p3 = points[((y + 1) * size) + x + 1]
			p4 = points[((y + 1) * size) + x]
			quads.append(Quad([p1, p2, p3, p4]))

	return quads

def generateGridPoints(size, step): 
    points = []

    y = 0.0
    while y <= size - 1:
        x = 0.0
        while x <= size - 1:
            points.append(MyPoint(x ,y))
            x += step
        y += step
    return points

def sinCos(point):
    return numpy.sin(point.x) + numpy.cos(point.y)

def TQuad(r, s, quad):
    return (quad.points[0] * (1.0 - r) + quad.points[1] * r) * (1.0 - s) + (quad.points[2] * r + quad.points[3] * (1.0 - r)) * s

def JT(r, s, quad):
	diffR = (quad.points[0] - quad.points[1]) * (s - 1.0) + (quad.points[2] - quad.points[3]) * s
	diffS = (quad.points[0] - quad.points[3]) * (r - 1.0) + (quad.points[2] - quad.points[1]) * r

	return numpy.array([[ diffR.x, diffS.x ], [ diffR.y, diffS.y ]])

	

def newtonMetod(p, quad):	
    r = 0.5
    s = 0.5
    i = 0

    while i < 20: 
        T_rs = TQuad(r, s, quad)
        J = JT(r, s, quad)
        J_inv = numpy.linalg.inv(J)

        nextSolution = (r, s) - (J_inv @ (T_rs - p).getArray())
        r = nextSolution[0]
        s = nextSolution[1]

        i += 1
    return MyPoint(r, s)


def getFLamba(r, s, quad):
    f_i_1 = sinCos(quad.points[0])
    f_i_2 = sinCos(quad.points[1])
    f_i_3 = sinCos(quad.points[2])
    f_i_4 = sinCos(quad.points[3])
    
    phi1 = (1.0 - r) * (1.0 - s)
    phi2 = r * (1.0 - s)
    phi3 = r * s
    phi4 = (1.0 - r) * s

    return (f_i_1 * phi1) + (f_i_2 * phi2) + (f_i_3 * phi3) + (f_i_4 * phi4)

def interpolate(pixelGrid, quadsGrid):
    interpolated = []
    for p in pixelGrid:
        f_lambda = -2.0
        for quad in quadsGrid:
            poly = Polygon([quad.points[0].getArray(),quad.points[1].getArray(), quad.points[2].getArray(), quad.points[3].getArray()])
            if poly.contains(Point(p.x, p.y)):
                r, s = newtonMetod(p, quad).getArray()
                f_lambda = getFLamba(r, s, quad)

        interpolated.append(f_lambda)
    return interpolated


def getResult(interpolated, pixelGrid):
    pixelsOnRow = numpy.sqrt(len(pixelGrid))
    arr = []
    for y in range(int(pixelsOnRow)):
        arr.append([])
        for x in range(int(pixelsOnRow)):
            idx = y * pixelsOnRow + x
            arr[y].append(interpolated[int(idx)])


    plt.figure("Result")
    
    plt.imshow(arr, cmap='hot', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.show()


def main():
    size = 10

    drawSinCosFunction(size)
    quadsGrid = generateGridQuads(size)

    renderPoints(quadsGrid)
    renderQuads(quadsGrid)
    pixelGrid = generateGridPoints(size, 0.1)

    getResult(interpolate(pixelGrid, quadsGrid), pixelGrid )


if __name__ == "__main__":    
    main()