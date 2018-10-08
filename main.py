from matplotlib import pyplot
from gmm import *

# load data
Y = np.loadtxt("gmm.data")
matY = np.matrix(Y, copy=True)
print(Y.shape)
pyplot.plot(Y[:, 0], Y[:, 1],'ko', label="class1")
pyplot.show()

# number of components
K = 2

# calculate parameters of GMM
mu, cov, alpha = gmm_em(matY, K, 100)

# According to GMM, culstering observations
N = Y.shape[0]
# calculating the matrix of belief
gamma = e_step(matY, mu, cov, alpha)
# for every observations. label the most probable component
category = gamma.argmax(axis=1).flatten().tolist()[0]
# put every observations into array
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])

print("____________")
print(class1.shape)
print(class2.shape)
# plot
pyplot.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
pyplot.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")

pyplot.legend(loc="best")
pyplot.title("GMM Clustering By EM Algorithm")
pyplot.show()
