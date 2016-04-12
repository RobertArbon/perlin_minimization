import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)


class Perlin(object):

	def __init__(self,x_max):
		"""
		"""
		self.x_ini = 0
		self.x_max = int(x_max)
		self.grid_grad = np.zeros(self.x_max+1)
		self.add_gradients()

	def value(self, x):
		"""
		Returns value of Perlin function at the point x
		"""
		# get grid coordinate
		x0 = self.grid_point(x)
		x1 = x0 + 1


		# determin interpolation weights
		sx = x - x0

		# Interpolate
		n0 = self.dot_grid_grad(x0,x)
		n1 = self.dot_grid_grad(x1,x)
		ix0 = self.lerp(n0, n1, sx)
		# print 'x0 ~ x ~ x1: {0}~{1}~{2}'.format(x0, x, x1)
		# print 'n0: {0},n1: {1},sx: {2} -> ix0: {3}'.format(n0, n1, sx, ix0)
		# print 

		return ix0

	def lerp(self, a0, a1, w):
		"""
		Linear interpolation between a0 and a1 with weight, w
		"""
		return a0 + (6*w**5 - 15*w**4 + 10*w**3)*(a1-a0)

	def add_gradients(self):
		"""
		Adds random graident vectors
		"""
		for i in range(len(self.grid_grad)):
			self.grid_grad[i] = np.random.uniform(-1, 1, 1)[0]
			# self.grid_grad[i] = np.random.random_integers(-1, 1)
		

	def dot_grid_grad(self, xi, x):
		"""
		Returns the value of the dot product of the gradient vector at specified point, x 
		"""
		dx = x - xi
		return dx*self.grid_grad[xi]
	
	def grid_point(self, x):
		try: 
			if x>=self.x_ini and x<self.x_max:
				return int(x)
			else:
				raise ValueError
		except ValueError:
			print "x ({0}) must be in [{1} and {2})".format(x, 0, self.x_max)



x_max = 50.0
one_d_perlin = Perlin(x_max)


# plt.scatter(one_d_perlin.grid, one_d_perlin.grid_grad)
# plt.show()
xs = []
ys = []
for i in range(10000):
	x = np.random.random(1)*x_max
	xs.append(x)
	ys.append(one_d_perlin.value(x))

plt.scatter(xs, ys)
plt.show()










