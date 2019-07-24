import FunctionEx_01 as ex1
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

## ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... \n');
print('5x5 Identity Matrix: \n');
ex1.warmUpExercise()
##======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt("ex1data1.txt",delimiter=',')
x_train = data[:,0]
y = data[:,1]
m = len(y); 
ex1.plotData(x_train,y)
## =================== Part 3: Cost and Gradient descent ===================

 # Add a column of ones to x
x_vals_column = np.transpose(np.matrix(x_train))
 
ones_column = np.transpose(np.matrix(np.repeat(1, m)))
 
X = np.column_stack((ones_column, x_vals_column))
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = ex1.computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ',J)
print('Expected cost value (approx) 32.07\n')
# further testing of the cost function
k = np.matrix('-1.;2.')
J = ex1.computeCost(X, y, k)
print('\nWith theta = [-1 ; 2]\nCost computed = ', J);
print('Expected cost value (approx) 54.24\n');
print('\nRunning Gradient Descent ...\n')
#run gradient descent
theta = ex1.gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ',theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')


point = plt.scatter(x_train,y)
line = plt.plot(X[:,1], X*theta, 'r')
plt.gca().legend(('Linear regression','Training data'))
plt.show()
p1 = np.matrix('1;3.5')
predict1 = np.transpose(p1)*theta
print('For population = 35,000, we predict a profit of %f\n',predict1*10000)
#predict2 = np.matrix('1;7')*theta
#print('For population = 70,000, we predict a profit of %f\n',predict2*10000)

##============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = [[ theta0_vals[i]],[ theta1_vals[j]]]
		J_vals[i,j] = ex1.computeCost(X, y, np.matrix(t))

J_vals_T = np.transpose(J_vals)

#print(J_vals_T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#tmp_planes = ax.zaxis._PLANES 
#ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], tmp_planes[0], tmp_planes[1], tmp_planes[4], tmp_planes[5])


X, Y = np.meshgrid(theta0_vals,theta1_vals)

ax.plot_surface(X, Y, J_vals_T)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(theta0_vals, theta1_vals, J_vals_T, np.logspace(-2, 3, 20))
ax.clabel(CS, inline=1, fontsize=10)


ax. plot(theta[0], theta[1], 'rx', linewidth=2, markersize=10)

plt.show()
