import numpy
import matplotlib.pyplot as plt
from random import seed
from random import random
import math
import functools
import cv2
from shapely.geometry import Polygon, Point
# seed random number generator
seed(1)


class P:
	x = 0.0
	y = 0.0
	def __init__(self, x_, y_):
		self.x = float(x_)
		self.y = float(y_)

	def __add__(self, p: 'P'):
		return P(self.x + p.x, self.y + p.y)
	
	def __sub__(self, p: 'P'):
		return P(self.x - p.x, self.y - p.y)


	def __mul__(self, c: float):
		return P(self.x * c, self.y * c)

	
	def __str__(self):
		return "(" + str(self.x) + ", " + str(self.y) + ")"

	def to_array(self): 
		return [self.x, self.y]


# Quad vizualization
#
#		(p4) ----------------- (p3)	(p4) ----------------- (p3)
#		 |						 |	 |						 |
#		 |						 |	 |						 |
#		 |						 |	 |						 |
#		 |						 |	 |						 |
#		 |			C			 |	 |			C			 |
#		 |						 |	 |						 |
#		 |						 |	 |						 |
#		 |						 |	 |						 |
#		(p1) ----------------- (p2)	(p1) ----------------- (p2)

class Quad:
	points = [] 
	def __init__(self, points_ = []):
		self.points = points_


	@property
	def p1(self) -> P: return self.points[0]
	@property
	def p2(self) -> P: return self.points[1]
	@property
	def p3(self) -> P: return self.points[2]
	@property
	def p4(self) -> P: return self.points[3]

	@p1.setter
	def p1(self, p): self.points[0] = p
	@p2.setter
	def p2(self, p): self.points[1] = p
	@p3.setter
	def p3(self, p): self.points[2] = p
	@p4.setter
	def p4(self, p): self.points[3] = p

	


def halfRandom():	
	return (random() - 0.5) * 0.2

def generateGridPoints(size = 10, step = 0.1): 
	points = []

	y = 0.0
	while y <= size - 1:
		x = 0.0
		while x <= size - 1:
			points.append(P(x ,y))
			x += step
		y += step
	return points

def generateGridQuads(size): 
	points = []
	for y in range(size):
		for x in range(size):
			points.append(P(float(x) + halfRandom(), float(y) + halfRandom()))
				
	quads = []
	for y in range(0, size - 1, 1):
		for x in range(0, size - 1, 1):
			p1 = points[(y * size) + x]
			p2 = points[(y * size) + x + 1]
			p3 = points[((y + 1) * size) + x + 1]
			p4 = points[((y + 1) * size) + x]
			quads.append(Quad([p1, p2, p3, p4]))

	return quads



def drawDebugSinCos(plot, size = 10.0):
	xlist = numpy.linspace(-1.0, size, 100)
	ylist = numpy.linspace(-1.0, size, 100)
	X, Y = numpy.meshgrid(xlist, ylist)
	Z = numpy.sin(X) + numpy.cos(Y)
	cp = plot.contourf(X, Y, Z, levels=100)


def sinCos(p: P) -> float:
	return numpy.sin(p.x) + numpy.cos(p.y)

def sampleFunction(quadsGrid):
	ret = []
	for quad in quadsGrid:
		for point in quad.points:
			ret.append(sinCos(point))
	return ret


def T_quad(r: float, s: float, quad: Quad) -> P:
	return (quad.p1 * (1.0 - r) + quad.p2 * r) * (1.0 - s) + (quad.p3 * r + quad.p4 * (1.0 - r)) * s

def J_matrix(r: float, s: float, quad: Quad) -> P:
	#				(s - 1.0) * (quad.p1 - quad.p2) + s * (quad.p3 - quad.p4)
	diffR = (quad.p1 - quad.p2) * (s - 1.0) + (quad.p3 - quad.p4) * s
	diffS = (quad.p1 - quad.p4) * (r - 1.0) + (quad.p3 - quad.p2) * r

	return numpy.array([
			[ diffR.x, diffS.x ], 
			[ diffR.y, diffS.y ], 
		])


	

def newtonMetod(p: P, quad: Quad) -> P:
	
	r = 0.5
	s = 0.5

	iter = 0
	while iter < 20:
		iter += 1

		T_rs = T_quad(r, s, quad)
		J = J_matrix(r, s, quad)
		J_inv = numpy.linalg.inv(J)

		nextSolution = (r, s) - (J_inv @ (T_rs - p).to_array())
		
		r = nextSolution[0]
		s = nextSolution[1]
		
	
	return P(r, s)

def main():

	CANVAS_SIZE = 10
	PIXEL_DPI = 0.1
	PIXEL_PER_ROW = int(CANVAS_SIZE / PIXEL_DPI)

	plt.figure(1)
	# Draw original function
	drawDebugSinCos(plt, CANVAS_SIZE)



	# Generate quads grid
	quadsGrid = generateGridQuads(CANVAS_SIZE)
	
	# Plot quads grid
	for quad in quadsGrid:
		xs = []
		ys = []
		for p in quad.points:
			xs.append(p.x)
			ys.append(p.y)
		xs.append(xs[0])
		ys.append(ys[0])
		plt.plot(xs, ys, c="black")
	
	# Plot quads points
	for quad in quadsGrid:
		for p in quad.points:
			plt.scatter(p.x, p.y, zorder=10000000, c="black", s=8)



	# Generate pixel sampling grid
	pixelGrid = generateGridPoints(CANVAS_SIZE, PIXEL_DPI)

	#samples = sampleFunction(quadsGrid)

	# Plot pixel samples
	#for p in pixelGrid:
		#plt.scatter(p.x, p.y, zorder=100000000000, c="red", s=4)


	rasterized = []
	for p in pixelGrid:
		f_lambda = -2.0
		for quad in quadsGrid:
			poly = Polygon([quad.p1.to_array(), quad.p2.to_array(), quad.p3.to_array(), quad.p4.to_array()])
			if poly.contains(Point(p.x, p.y)):
				# Work the magic
				r, s = newtonMetod(p, quad).to_array()

				
				f_i_1 = sinCos(quad.p1)
				f_i_2 = sinCos(quad.p2)
				f_i_3 = sinCos(quad.p3)
				f_i_4 = sinCos(quad.p4)
				
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
	
	
	
main()
