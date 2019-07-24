import FunctionEx_01 as ex1
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

## ================ Part 1: Feature Normalization ================


print('Loading data ...\n')

# Load Data
data = np.loadtxt("ex1data2.txt",delimiter=',')
X = data[:,[0,1]]
y = data[:,2]
m = len(y)
#print('First 10 examples from the dataset: \n')
#print('X = %0%, y = {1}',x_train[10:],y[10:])
print('Normalizing Features ...\n')

X, mu, sigma = ex1.featureNormalize(X)

# Add a column of ones to x
ones_column = np.transpose(np.matrix(np.repeat(1, m)))
X = np.column_stack((ones_column, X))
print(mu)
## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1))
theta, J_history = ex1.gradientDescentMulti(X, y, theta, alpha, num_iters)
# Plot the convergence graph
n = 400

for i in range(n):
	plt.plot(i, J_history[i],'rx', linewidth=2)
plt. xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta)

data = np.loadtxt("ex1data2.txt",delimiter=',')
X = data[:,[0,1]]
y = data[:,2]
m = len(y)

# Add a column of ones to x
ones_column = np.transpose(np.matrix(np.repeat(1, m)))
X = np.column_stack((ones_column, X))

theta = ex1.normalEqn(X, y)
#theta = np.linalg.pinv(np.transpose(X) * X)
print("theta noraml: \n")
print(theta)

