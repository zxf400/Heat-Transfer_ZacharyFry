import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Initiaize array with correct BCs and on initial guess at T dis
N = 10
T_arr = np.ones((N,N))*((100*75*50*0)/4)
T_arr[:,0] = 0
T_arr[:,-1] = 100
T_arr[0,:] = 75
T_arr[-1,:] = 50
                
# Set variables for taolerance and max diff
tol = 1e-1
max_diff = tol + 1
lam = 1.5
iters = 0
max_iters = 10000

while max_diff > tol:
    max_diff = 0
    
##############################################################
# Dirichlet 
##############################################################
    
    # good idea for interation number, if hit max interations produce error
    for n in range(1,N-1): # loop over column (y) index within boundarys
        for m in range(1,N-1):
           T_new = (T_arr[m+1,n] + T_arr[m-1,n] +T_arr[m, n+1] + T_arr[m,n-1])/4
           T_new = lam*T_new + (1-lam)*T_arr[m,n]
           diff = abs(T_new - T_arr[m,n])
           max_diff = max(max_diff, diff)
           T_arr[m,n] = T_new
           
# Check if max iterations were reached
if iters == max_iters:
    print("Reached Max iterations")
           
print(T_new)

#############################################################
# Von Neumann
#############################################################

# Top Boundary 
T_arr[0, :] = T_arr[1, :]

# Bottom Boundary
T_arr[-1, :] = T_arr[-2, :]

# Left Boundary
T_arr[:, 0] = T_arr[:, 1]

# Right Boundary
T_arr[:, -1] = T_arr[:, -2]

print("Final Temperature Distribution:")
print(T_arr)

plt.imshow(T_arr, cmap='hot', interpolation='nearest')
plt.colorbar(label="Temperature")
plt.title("Temperature Distribution - Von Neumann")
plt.show()

############################################################
# Mixed Boundary Condition
############################################################

# Top Boundary: Dirichlet
T_arr[0, :] = 75

# Bottom Boundary: Neumann
T_arr[-1, :] = T_arr[-2, :]

# Left Boundary: Dirichlet
T_arr[:, 0] = 0

# Right Boundary: Neumann
T_arr[:, -1] = T_arr[:, -2]

print("Final Temperature Distribution (Mixed Boundary Conditions):")
print(T_arr)

plt.imshow(T_arr, cmap='hot', interpolation='nearest')
plt.colorbar(label="Temperature")
plt.title("Temperature Distribution - Mixed Boundary Condition")
plt.show()



           

           


        
        
