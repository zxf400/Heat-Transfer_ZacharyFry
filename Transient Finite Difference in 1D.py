import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# Parameters
#######################################################################

Fo = 0.5
N = 101
M = 15000
L = 1.0
dx = L / (N-1)
h = 50 # W/m^2K
k = 200 # W/mK
Bi = h*dx/k

# Tip boundary condition type: 'convective', 'neumann', or 'dirichlet'
bc_type = 'convective'  

def T_inf_func(t):
    return 0.5 + 0.5 * np.sin(2*np.pi*t/3000)

#######################################################################
# Set up Initial Condition, that meets the BCs
#######################################################################

T_arr = np.zeros((N,M))
T_arr[0] = 0
T_arr[-1] = 1

#######################################################################
# Calc coef. Matrix; A
#######################################################################

A = np.zeros((N,N))
np.fill_diagonal(A[1:-1,0:-2], Fo)
np.fill_diagonal(A[1:-1,1:-1],-2-2*Fo)
np.fill_diagonal(A[1:-1,2:], Fo)

# Dirichlet at left
A[0,0] = 1

# Right boundary condition
if bc_type == 'convective':
    A[-1,-2] = -2 * Fo
    A[-1,-1] = 2 + 2*Fo*(1+Bi)
elif bc_type == 'neumann':
    A[-1, -2] = -1
    A[-1, -1] = 1
elif bc_type == 'dirichlet':
    A[-1,-1] = 1

#######################################################################
# Calc const matrix; C
#######################################################################

C = np.zeros((N,N))
np.fill_diagonal(C[1:-1,0:-2], -Fo)
np.fill_diagonal(C[1:-1,1:-1], -2+2*Fo)
np.fill_diagonal(C[1:-1,2:], -Fo)

# Dirichlet at left 
C[0,0] = 1

# Right boundary condition
if bc_type == 'convective':
    C[-1,-2] = 2*Fo
    C[-1,-1] = -2*Fo+2-2*Fo*Bi
elif bc_type == 'neumann':
    C[-1, -2] = -1
    C[-1, -1] = 1
elif bc_type == 'dirichlet':
    C[-1, -1] = 1

#######################################################################
# Loop over time and apply FTCS
#######################################################################

for idx in range(1,M):
    b = C @ T_arr[:, idx-1]
    if bc_type == 'convective':
        T_inf = T_inf_func(idx)
        b[-1] += 4 * Fo * Bi * T_inf
    elif bc_type == 'dirichlet':
        b[-1] = 1  
    T_arr[:,idx] = np.linalg.solve(A,b)

#######################################################################
# Plot Results
#######################################################################

fig, ax = plt.subplots(figsize=(6,5))
x_arr = np.linspace(0,1,N)
ax.plot(x_arr, T_arr[:,::3000])
ax.set_ylabel('Temperature')
ax.set_xlabel('Distance')
ax.set_title(f'Temperature Evolution Over Time ({bc_type.capitalize()} Tip)')
ax.grid()
plt.show()
