'''
Created on Apr 19, 2018

@author: Bruno Herisse (ONERA)

This Python script simulate vertical landing control based on visual information extracted from optical flow. Read the following paper for more information about the algorithm :
B. Herisse, T. Hamel, R. Mahony, F-X. Russotto, "Landing a VTOL Unmanned Aerial Vehicle on a moving platform using optical flow", IEEE Transactions on Robotics, Vol. 28, No. 1, pp. 77-89, February 2012.
'''
import numpy as np
import math
import scipy.integrate
import matplotlib.pyplot as plt

# Parameters
kp = 10     # proportional gain parameter
ki = 100    # integral gain parameter
delta = 0.2 # constant drift (m/s/s)
h0 = 1      # initial altitude (m)
wc = 0.5    # optical flow set point
oscill = np.array([0.1, 1])     # platform oscillations (magnitude, frequency)
t0, t1 = 0, 10                  # start and end time (s)

# Model definition
def model(t,x):
    H = x[0]-oscill[0]*math.sin(2*math.pi*oscill[1]*t)  				# current height
    dH = x[1]-oscill[0]*(2*math.pi*oscill[1])*math.cos(2*math.pi*oscill[1]*t) 		# current height speed
    GE = 1/(1-math.pow((0.15/(H+0.5)),2));  						# ground effect contribution
    xdot = np.zeros((1,3))
    xdot[0,0] = x[1]									# time derivative of height
    xdot[0,1] = - kp*GE*(dH/H+wc) - delta - x[2] + (GE-1)*9.81				# time derivative of vertical speed
    xdot[0,2] = ki*GE*(dH/H+wc)								# time derivative of the integral term of the control
    return xdot

# Simulation
t = np.linspace(t0, t1, 100)  		# the points of evaluation of solution
x0 = [h0, 0, 0]                   	# initial value for (altitude, speed)
x = np.zeros((len(t), len(x0)))   	# array for solution
x[0, :] = x0
h = np.zeros((len(t), 2))   		# array for height
h[0, 0] = h0				# true height
h[0, 1] = h0				# height of reference
w = np.zeros((len(t), 2))   		# array for optical flow
w[0, 0] = 0				# true optical flow
w[0, 1] = wc				# optical flow of reference 
r = scipy.integrate.ode(model).set_integrator("dopri5")  		# integration with dopri5 (adaptive step-size) = ode45 in Matlab
r.set_initial_value(x0, t0)   						# initial values
for i in range(1, t.size):
    x[i,:] = r.integrate(t[i]) 						# get one more value, add it to the array
    h[i,0] = x[i,0]-oscill[0]*math.sin(2*math.pi*oscill[1]*t[i])	# compute true height
    h[i,1] = h0*math.exp(-wc*t[i])					# compute reference height
    w[i,0] = -(x[i,1]-oscill[0]*(2*math.pi*oscill[1])*math.cos(2*math.pi*oscill[1]*t[i]))/(x[i,0]-oscill[0]*math.sin(2*math.pi*oscill[1]*t[i])) # compute true OF
    w[i,1] = wc																	# compute reference OF
    if not r.successful():
        raise RuntimeError("Could not integrate")

# Figures
plt.figure(1)
plt.plot(t, h)
plt.xlabel('time (s)')
plt.ylabel('height to platform (m)')
plt.legend(['true height', 'reference height'])
plt.figure(2)
plt.plot(t, w)
plt.xlabel('time (s)')
plt.ylabel('optical flow')
plt.legend(['true optical flow', 'reference optical flow'])
plt.show()
