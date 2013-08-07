# Flash and Hogan minimum jerk
#
# Mehrdad Yazdani, Summer 2013

from scipy.linalg import toeplitz
from numpy import *
from cvxopt import matrix, spmatrix, sparse, solvers
from cvxopt.modeling import variable, op
import matplotlib.pyplot as plt

# setup sample rate
N = 500 # number of samples
delta = 1/float_(N) # sample rate
n = arange(0, 1, delta) #one second worth of samples


def minimum_effort_control():
	'''
	Minimum effort control problem
	
	minimize	max{||D_jerk * x||}
	subject to	A_eq * x == b_eq
	'''
	#create matrix data type for cvxopt
	D_sparse = sparse([matrix(1/float_(power(N, 3))*D_jerk)]) # we multiply
	# D_jerk with 1/float_(power(N, 3)) to ensure numerical stability
	A_eq = sparse([matrix(Aeq)])
	b_eq = matrix(beq)

	t = variable()  #auxiliary variable
	x = variable(N) #x position of particle
	op(t, [-t <= D_sparse*x, D_sparse*x <= t, A_eq*x == b_eq]).solve() #linear program
	
	return x
		

def plot_dynamics(position_profile):
	#acceleration derivaties
	row_acceleration = hstack((array([[1,-2,1]]), zeros((1, N-3)))) 
	col_acceleration = vstack((array([[1]]), zeros((N-3,1))))
	D_acceleration = power(N, 2)*toeplitz(col_acceleration, row_acceleration)
	
	#velocity derivatives
	row_velocity = hstack((array([[-1, 1]]), zeros((1,N-2))))
	col_velocity = vstack((array([[-1]]), zeros((N-2,1))))
	D_velocity = N*toeplitz(col_velocity, row_velocity)
	plt.subplot(2,2,1)
	plt.plot(n, position_profile)
	plt.title('Position')
	
	
	plt.subplot(2,2,2)
	plt.plot(n[0:N-1], dot(D_velocity, position_profile))
	plt.title('Velocity')
	
	plt.subplot(2,2,3)
	plt.plot(n[0:N-2], dot(D_acceleration, position_profile))
	plt.title('Acceleration')
	plt.xlabel('Time (s)')
	
	plt.subplot(2,2,4)
	plt.plot(n[0:N-3], dot(D_jerk, position_profile))
	plt.xlabel('Time (s)')
	plt.title('Jerk')
		

'''
Solve minimum effort control problem for a partical at an intial postion of 2.0
and a final position of 5.0. This is the "straight path" trajectory. 
'''
# set up jerk matrix D_{jerk}:
row_jerk = hstack((array([[-1, 3, -3, 1]]), zeros((1,N-4))))
col_jerk = vstack((array([[-1]]), zeros((N-4,1))))
D_jerk = power(N, 3)*toeplitz(col_jerk, row_jerk)

# set up constraint matrices:Aeq*x = beq
initial_position = hstack((array([[1]]), zeros((1,N-1))))
final_position = hstack((zeros((1,N-1)), array([[1]])))

initial_velocity = hstack((array([[-1, 1]]), zeros((1,N-2))))
final_velocity = hstack((zeros((1,N-2)), array([[-1, 1]])))

initial_acceleration = hstack((array([[1,-2,1]]), zeros((1,N-3))))
final_acceleration = hstack((zeros((1,N-3)), array([[1,-2,1]])))

Aeq = vstack((initial_position, final_position, initial_velocity, \
	final_velocity, initial_acceleration, final_acceleration))
beq = zeros((6,1))
beq[0] = 2 #initial position
beq[1] = 5 #final position 


x_straight = minimum_effort_control()
plt.figure()
plot_dynamics(x_straight.value)


'''
Solve minimum effort control problem for a partical at an intial postion of 2.0,
passing by a via point at t = N/2 at position 7.0 and having a final position
of 5.0. This is the "curved path" trajectory. The objective function remains 
the same as the straight path case. 
'''
#objective function remains the same
#via point
via_point_position = zeros((1,N))
via_point_position[0,int_(N/2)] = 1

Aeq = vstack((initial_position, final_position, via_point_position,  \
	initial_velocity, final_velocity, initial_acceleration, final_acceleration))
beq = zeros((7,1))
beq[0] = 2 #inital position
beq[1] = 5 #final position
beq[2] = 7 #via point position

x_curve = minimum_effort_control()
plt.figure()
plot_dynamics(x_curve.value)


plt.show()
