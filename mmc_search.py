import random 
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np



def penalty(x):
	"""
	Penalty function. Here an asymmetric double well.
	
	X-vals of stationary points
	np.roots([2, 0, -4, -1])
	Out[30]: array([ 1.52568712, -1.2670351 , -0.25865202])
	
	Y-vals at stationary points:
	In [31]: penalty(np.roots([2, 0, -4, -1]))
	Out[31]: array([  3.05602694,   8.68979677,  10.2541763 ])

	Global minimum at (1.52568712, 3.05602694)
	"""
	return 	x**4 - 4*x**2 - 2*x + 10

def probability(y, T):
	"""
	Metropolis acceptance probability. 
	"""
	return np.exp(-y/T)

def montecarlo(x, dx, num_iter, Temp):
	"""
	Main Montecarlo routine. 
	Returns: all x values
		   : acceptance ratio
	"""
	coors = []
	acceptance = 0
	for i in range(num_iter):
		# Trial move x
		delta_x = random.uniform(-dx, dx)
		x_trial = x + delta_x

		# work out penalty and probability of move
		y_trial = penalty(x_trial)
		y = penalty(x)
		prob = min(1, probability(y_trial, Temp)/probability(y, Temp))
		
		# Move x with above probability
		if (random.uniform(0, 1) < prob):
			x = x_trial
			y = y_trial
			acceptance += 1
		
		coors.append([x,y])
	acceptance /= float(num_iter)
	return coors, acceptance

def estimate_dx(x, dx_start, num_iter, Temp, tol, ddx, max_iter):
	"""
	Estimates a dx given a temperature and an initial starting 
	point (dx_start) and a position, such that the acceptance ratio is 
	50% +/- tol
	"""
	count = 0
	target = 0.5
	dx = dx_start
	while count < max_iter: 
		_, ar = montecarlo(x, dx, num_iter, Temp)
		if (ar < target - tol):
			dx -= ddx
			count += 1
		elif (ar > target + tol):
			dx += ddx
			count += 1
		else:
			# print 'convergence at', count
			return dx
	# print 'convergence at', count
	return dx

def main(T_start, T_end, num_T_steps, num_iter, dx_ini):
	"""
	Main loop.  
	T_start, T_end: starting and ending temperature
	num_T_steps = number of different temps. i.e. cooling rate
	num_iter = number of different steps at each temp
	"""
	# parameters for estimating dx each temperature decrease
	scan_iter = 100
	max_iter = 100
	tol = 0.005
	ddx = 0.01

	Temps = np.linspace(T_start, T_end, num_T_steps)
	results = np.empty([5, num_T_steps])
	coors = []
	x = random.uniform(-2, 0)
	dx = dx_ini	
	for i in range(len(Temps)): 
		T = Temps[i]

		# Estimate dx at new temperature
		new_dx = estimate_dx(x, dx, scan_iter, T, tol, ddx, max_iter)
		dx = new_dx

		# Main Montecarlo routine
		new_coors, new_ar = montecarlo(x, dx, num_iter, T)
		coors.extend(new_coors)
		new_coors = map(list, zip(*new_coors))
		# Record results:
		results[0,i] = (i+1)*num_iter
		results[1,i] = new_ar
		results[2,i] = T
		results[3,i] = np.mean(new_coors[0])
		results[4,i] = np.std(new_coors[0])

		# Get last position to start the next iteration. 
		x = coors[-1][0]

	return coors, results


def print_results(results, xs):
	"""
	prints results
	"""		
	pad = 250
	fig, axs = plt.subplots(3, sharex = True)
	n = range(int(results[0,-1]))
	dn = results[0,1]-results[0,0]
	# Raw x values with the 
	axs[0].plot(n, xs, label = 'x', alpha = 0.2 )
	axs[0].errorbar(results[0], results[3], yerr=results[4], \
		lw=0, elinewidth=2, capsize = 3, fmt='o', color='k')
	axs[0].set_xlim(-pad, n[-1]+pad)
	axs[0].set_ylabel('X-Value')

	axs[1].bar(results[0]-dn, results[2], width = dn)	
	axs[1].set_ylabel('Temperature')

	axs[2].scatter(results[0], results[1])
	axs[2].set_ylim(0,1)
	axs[2].set_ylabel('Acceptance ratio')
	axs[2].set_xlabel('Iteration')

	fig.suptitle('MC simulated annealing')
	plt.savefig('MCMC_Sim_Ann_DoubleWell.png')
	fig.show()
	return None


	
T_start = 1e0
T_end = 1e-2
num_T_steps = 10
num_iter = 1000
dx_ini = .5

coors, results = main(T_start, T_end, num_T_steps, num_iter, dx_ini)
coors = map(list, zip(*coors))

# print_results(results, coors[0])
#
# ANIMATION
#
"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# The data to plot and animate
skip = 10
coors[0] = coors[0][::skip]
coors[1] = coors[1][::skip]
num_frames = len(coors[0])

xs = np.linspace(-3,3, 1000)
ys = penalty(xs)
Ts = np.linspace(T_start, T_end, num_T_steps)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-3, 3), ylim=(0, 20))
line, = ax.plot([], [], 'o-', lw=4,alpha = 0.8)
temp_text = ax.text(0.2, 0.95, '', transform=ax.transAxes)

# Static elements
ax.plot(xs, ys, lw=2, color = 'k', alpha = 0.3)
ax.text(0.2, 0.9, r'$y = x^{4} - 4\cdot x^{2} - 2\cdot x + 10$', transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.suptitle('Monte Carlo Simulated Annealing')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    temp_text.set_text('')
    return line,

# animation function.  This is called sequentially
def animate(i):
	x = coors[0][(i-10):i]
	y = coors[1][(i-10):i]
	line.set_data(x, y)
	temp_text.set_text(r'$\mathrm{Temperature} = %.2f $' % Ts[i*skip/num_iter])
	return line,

# call the animator.  blit=True means only re-draw the parts that have changed.

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames, interval=1, blit=False)
anim.save('MCMC_Sim_Ann_DoubleWell.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

fig.show()