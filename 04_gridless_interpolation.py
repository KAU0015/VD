import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def generatePoints(count):
    pointsXY = []
    for i in range(count):
        x,y = np.random.normal(0, 100, 2)
        pointsXY.append([x, y])

    pointsZ = []
    for i in range(len(pointsXY)):
        x,y = pointsXY[i]
        z = getNormalizedGaussian(x, y)
        pointsZ.append(z)

    return np.array(pointsXY), np.array(pointsZ)

def getNormalizedGaussian(x, y, mx=0, my=0, scale=100):
    return 1.0 / (2.0 * np.pi * scale**2) * np.exp(-((x - mx)**2 / (2.0 * scale**2) + (y - my)**2 / (2.0 * scale**2)))


def showPointsIn3D(pointsXY, pointsZ):
    plt.figure("3D points")
    sample_domain_plot = plt.axes(projection='3d')
    sample_domain_plot.scatter3D(pointsXY[:,0], pointsXY[:,1], pointsZ, c=pointsZ, cmap="hot")

def showResult(result):
    plt.figure("Result")
    plt.imshow(result, cmap="hot", interpolation='nearest')
    plt.colorbar()


def scatteredPointInterpolation(pointsZ, knn):
    LAMBDA = 0.001
    MAX_DISTANCE = 75
    result = []
    y_id = 0
    for y in range(-200, 200, 1):
        result.append([])
        for x in range(-200, 200, 1):
            query_point = np.array([x, y])

            nearestPoints = knn.kneighbors(query_point.reshape(1, -1), return_distance=True)
            nearestPointsIndexes = nearestPoints[1][0]
            nearestPointsDistances = nearestPoints[0][0]
            
            R = sum(nearestPointsDistances)/5.0


            sum_numerator = 0.0
            sum_denominator = 0.0
            for i in range(len(nearestPointsIndexes)):
                if nearestPointsDistances[i] > MAX_DISTANCE: break
                f_i = pointsZ[nearestPointsIndexes[i]]
#np.power((actual_value-predicted_value),2)
                sum_numerator += f_i * np.exp(1 * pow( abs(nearestPointsDistances[i] / R), 2))
                sum_denominator += np.exp(1 * pow( abs(nearestPointsDistances[i] / R), 2))

            f_lambda = sum_numerator / sum_denominator if sum_denominator != 0.0 else 0.0
            result[y_id].append(f_lambda)
        y_id += 1
    return result




def main():

    pointsXY, pointsZ = generatePoints(100)
    showPointsIn3D(pointsXY, pointsZ)
    
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(pointsXY)

    showResult(scatteredPointInterpolation(pointsZ, knn))
            
    plt.show()

if __name__ == "__main__":    
    main()