
import numpy as np
import matplotlib.pyplot as plt

def partial_derivative(i, theta, x, y):
	a = np.dot(x, theta.T)-y
	#print(a)
	#print(x[:,i])
	return np.dot((np.dot(x, theta.T)-y),x[:,i])/x.shape[0]
	
def feature_scaling(x):
	n = x.shape[1]
	m = x.shape[0]
	feature_scales = np.array([1.0]*n)
	scaled_x = np.copy(x)
	
	for i in range(1, n):
		col_max = max(abs(x[:,i]))
		if abs(col_max) < 0.0001:
			continue
		feature_scales[i] = 1.0/col_max
		for j in range(m):
			o = scaled_x[j,i]
			scaled_x[j,i] = o*feature_scales[i]
	
	return feature_scales, scaled_x

def gradient_descent(alpha, x, y):
	
	thetas = x.shape[1]
	theta_temp = np.empty(thetas, dtype=float)
	theta = np.array([0.0]*thetas)
	#print(theta_temp)
	iter_count = 0
	max_iter = 20000
	derivative_sum = 99999
	required_derivative_sum = 0.00001
	
	feature_scales, scaled_x = feature_scaling(x)
	print("feature_scales = " ,feature_scales,"scaled_x = ", scaled_x)
	
	
	while iter_count < max_iter and derivative_sum > required_derivative_sum:
		derivative_sum = 0
		for i in range(thetas):
			pd = partial_derivative(i, theta, scaled_x, y)
			derivative_sum += abs(pd)
			theta_temp[i] = theta[i] - alpha*pd
			#print(theta_temp[i])
		
		for i in range(thetas):
			theta[i] = theta_temp[i]
			
		iter_count += 1
		
		#plt.plot(x[:,1], np.dot(x,theta), ':')
	print('iter_count = ',iter_count, ', derivative_sum = ', derivative_sum)
	theta = theta * feature_scales
	
	print('theta = ', theta)
		
	return theta
	

x = np.array([1.0,2,2,3,3,4,5,6,6,6,8,10])
xx = np.array([len(x)*[1], x]).T
#print(xx[:,0])
y = np.array([-890.0, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])
print(xx.shape, y.shape)

theta = gradient_descent(0.1, xx, y)
hx = np.dot(xx, theta)
plt.plot(x, hx, 'r')

plt.ylabel('Second year A\'s')
plt.xlabel('First year A\'s')
plt.plot(x, y, 'ro')

plt.show()