import matplotlib.pyplot as plt
import numpy as np



def generateData(count):
    data = []
    for i in range(count):
        data.append(np.random.normal(20, 10, 500))
    return data

def drawPercentils(data):
    for i in range(len(data)):
        p25, p50, p75 = np.percentile(data[i], [25, 50, 75])
        plt.vlines(i+1, p25, p75, color='k', linestyle='-', capstyle='round', lw=5)
        plt.scatter(i+1, p50, zorder=1000, color='w', s=20)

data_count = 3
data = generateData(data_count)

fig = plt.figure()
plt.xticks(range(1, data_count+1, 1))
violin = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

plt.title('VD plot')
plt.xlabel('Month')
plt.ylabel('Temperature [°C]')

drawPercentils(data)

plt.show()
