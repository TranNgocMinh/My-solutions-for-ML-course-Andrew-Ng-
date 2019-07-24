import numpy as np
import matplotlib.pyplot as plt

def warmUpExercise():
	A = []
	A = np.eye(5)
	print(A)
def plotData(x, y):
	plt.scatter(x, y)
	plt.ylabel('Profit in %$10,000s');
	plt.xlabel('Population of City in 10,000s');
	plt.show()
def computeCost(X, y, theta):
	m = len(y)
	J = 0
	h = X * theta
	for i in range(m):
		J += (h[i] - y[i])**2
	J = J / (2 * m)
	return J
def computeCostMulti(X, y, theta):
	m = len(y)
	J = 0
	h = X * theta
	J = np.transpose(h-np.transpose(np.matrix(y)))*(h-np.transpose(np.matrix(y)))
	J = J / (2 * m)
	return J
def gradientDescent(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))
	for iter in range(num_iters):
		old_theta = theta
		theta_0 = 0
		theta_1 = 0
		for i in range(m):
			theta_0 += ((X[i, :] * old_theta) - y[i]) * X[i, 0]
			theta_1 += ((X[i, :] * old_theta) - y[i]) * X[i, 1]    
		theta[0] = theta[0] - (alpha * (1 / m) * theta_0)
		theta[1] = theta[1] - (alpha * (1 / m) * theta_1)
		#Save the cost J in every iteration    
		J_history[iter] = computeCost(X, y, theta)
		#print("Cost in iter %d is %f\n", iter, J_history(iter))
	return theta

def featureNormalize(X):
	X_norm = X
	mu = np.zeros((1, np.size(X, 1)))
	sigma = np.zeros((1, np.size(X, 1)));
	n = np.size(X, 1)
	for i in range(n):
		mu[0,i] = np.mean(X[:,i])
		sigma[0,i] = np.std(X[:,i])
		X_norm[:, i] = X_norm[:, i] - mu[0, i]
		X_norm[:, i] = X_norm[:, i] / sigma[0, i] 
		#np.disp(X_norm)
	return X_norm, mu, sigma
def gradientDescentMulti(X, y, theta, alpha, num_iters):
	m = len(y)
	J_history = np.zeros((num_iters, 1))
	n = np.size(X,1)
	for iter in range(num_iters):
		theta_tmp = np.zeros((np.size(theta, 0),1))
		for j in range(n):
			for i in range(m):
				theta_tmp[j] =  theta_tmp[j] + ((X[i,:]*theta) - y[i])*X[i,j]  
		for i in range(n):
			theta[i] = theta[i] - (alpha * (1 / m) * theta_tmp[i])
		J_history[iter] = computeCost(X, y, theta)
	return theta, J_history
def normalEqn(X, y):
	theta = np.zeros((np.size(X, 1), 1))
	theta = np.linalg.pinv(np.transpose(X) * X) * np.transpose(X) * np.transpose(np.matrix(y))
	return theta
