import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numpy.random.mtrand import multivariate_normal


def generatePointsXY(count):
	points = []
	for i in range(count):
		x,y = np.random.normal(loc=0, scale=100, size=2)
		points.append([x, y])

	return np.array(points)


def getNormalizedGaussian(x=0, y=0, mx=0, my=0, scale=100):
    return 1. / (2. * np.pi * scale * scale) * np.exp(-((x - mx)**2. / (2. * scale**2.) + (y - my)**2. / (2. * scale**2.)))


def generatePointsZ(pointsXY):
	points = []
	for i in range(len(pointsXY)):
		x,y = pointsXY[i]
		z = getNormalizedGaussian(x, y)
		points.append(z)

	return np.array(points)

def scatteredPointInterpolation(pointsZ, knn):
    LAMBDA = 0.001
    MAX_DISTANCE = 75
    raster = []
    y_id = 0
    for y in range(-200, 200, 1):
        raster.append([])
        for x in range(-200, 200, 1):
            query_point = np.array([x, y])

            nearest_points_result = knn.kneighbors(query_point.reshape(1, -1), return_distance=True)
            nearest_points_idx = nearest_points_result[1][0]
            nearest_distances = nearest_points_result[0][0]
            
            R_p = nearest_distances[0]


            sum_numerator = 0.0
            sum_denominator = 0.0
            for i in range(len(nearest_points_idx)):
                if nearest_distances[i] > MAX_DISTANCE: break
                f_i = pointsZ[nearest_points_idx[i]]

                sum_numerator += f_i * np.exp(LAMBDA * pow( abs(nearest_distances[i] / R_p), 2))
                sum_denominator += np.exp(LAMBDA * pow( abs(nearest_distances[i] / R_p), 2))

            f_lambda = sum_numerator / sum_denominator if sum_denominator != 0.0 else 0.0
            raster[y_id].append(f_lambda)
        y_id += 1
    return raster

def main():
    plt.figure(1)

    pointsXY = generatePointsXY(100)
    pointsZ = generatePointsZ(pointsXY)

    sample_domain_plot = plt.axes(projection='3d')
    sample_domain_plot.scatter3D(pointsXY[:,0], pointsXY[:,1], pointsZ, c=pointsZ, cmap="hot")

    plt.figure(2)
    plt.scatter(pointsXY[:,0], pointsXY[:,1], c=pointsZ, cmap="hot")
    plt.colorbar()

    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(pointsXY)


    raster=scatteredPointInterpolation(pointsZ, knn)
            

    plt.figure(3)
    plt.imshow(raster, cmap="hot", interpolation='nearest')
    plt.colorbar()


    plt.show()


if __name__ == "__main__":    
  main()