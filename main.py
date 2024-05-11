"""
MATH3036 Coursework 2 main script

@author: Prof Kris van der Zee (Lecturer)

version: 8 Mar 2023
"""

#%% Question 3(a)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[2, -1,0],[-1, 2,-1],[0,-1,2]])
b = np.array([[1],[0],[5]])

# Solver settings
x0 = np.array([[1],[1],[1]])
kmax = 8

# Plot residuals
fig, ax, SD_res, CG_res = LST.PlotResidualsOfMethods(A,b,x0,kmax)
print("\n SD_res = \n",SD_res)
print("\n CG_res = \n",CG_res)


#%% Question 3(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST


# Set system
n = 100;
tau = .15
d = np.ones(n)
A = np.diag(d) - tau * ( \
      np.diag(np.ones(n-1), 1) \
    + np.diag(np.ones(n-1),-1) \
    + np.diag(np.ones(round(n/2)-1), round(n/2)+1) \
    + np.diag(np.ones(round(n/2)-1),-round(n/2)-1) \
    )
b = np.ones([n,1])

# Solver settings
x0 = np.zeros([n,1])
kmax = 32

# Plot residuals
fig, ax, SD_array, CG_array = LST.PlotResidualsOfMethods(A,b,x0,kmax)



#%% Question 4(a)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

# Set system
A = np.array([[2, -1,0],[-1, 2,-1],[0,-1,2]])
x0 = np.array([[0],[0],[0]])
b = np.array([[1],[0],[5]])
kmax = 3

x_array, z_array = LST.CGNEmethod(A,b,x0,kmax)
print("x_array =\n",x_array)
print("z_array =\n",z_array)


#%% Question 4(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST


# Set system (nonsymmetric)
n = 100;
tau = .4
d = np.ones(n)
A = np.diag(d) - tau * ( \
      np.diag(np.ones(n-1), 1) \
    - np.diag(np.ones(n-1),-1) \
    + np.diag(np.ones(round(n/2)-1), round(n/2)+1) \
    - np.diag(np.ones(round(n/2)-1),-round(n/2)-1) \
    )
x = np.ones([n,1])
b = A @ x

# Solver settings
x0 = np.zeros([n,1])
kmax = 25

# CG method
x_array_CG = LST.CGmethod_sol(A,b,x0,kmax)
print("Last x (CG) =\n",x_array_CG[:,-1])

# CGNE method
x_array_CGNE, z_array = LST.CGNEmethod(A,b,x0,kmax)
print("Last x (CGNE) =\n",x_array_CGNE[:,-1])




#%% Question 5(a)

print("5a")
import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST
print("---------------------------------------------")
A = np.array([[1, 3],[2, 2]])
print("shape of A",np.shape(A))
v0 = np.array([[1],[0]])
kmax = 6
print("shape of v",np.shape(v0))
v_array, eigval_array, r_array = LST.PowerIterationMethod(A,v0,kmax)
print("---------------------------------------------")
print("\nPower Method")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("---------------------------------------------")
print("v_array shape",np.shape(v_array))
print("eigval_array shape",np.shape(eigval_array))
print("r_array shape",np.shape(r_array))
print("The eigvalues actually", np.linalg.eigvals(A))


#%% Question 5(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST
print("---------------------------------------------")
A = np.array([[2, -1, 0,0],[-1, 2, -1,0],[0,-1, 2, -1],[0,0,-1,2]])
v0 = 1/np.sqrt(10)*np.array([[1],[2],[1],[2]])
kmax = 8
print("shape of A",np.shape(A))
print("shape of v",np.shape(v0))
print("---------------------------------------------")
v_array, eigval_array, r_array = LST.PowerIterationMethod(A,v0,kmax)
print("\nPower Method")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("---------------------------------------------")
print("v_array shape",np.shape(v_array))
print("eigval_array shape",np.shape(eigval_array))
print("r_array shape",np.shape(r_array))
print("The eigvalues actually", np.linalg.eigvals(A))


#%% Question 6(a)
print("6a")

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST
print("---------------------------------------------")
A = np.array([[1, 3],[2, 2]])
print("shape of A",np.shape(A))
v0 = np.array([[1],[0]])
kmax = 6
print("shape of v",np.shape(v0))
print("---------------------------------------------")
v_array, eigval_array, r_array = LST.RayleighQuotientIteration(A,v0,kmax)
print("\nRayleigh Quotient Iteration")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("---------------------------------------------")
print("v_array shape",np.shape(v_array))
print("eigval_array shape",np.shape(eigval_array))
print("r_array shape",np.shape(r_array))
print("The eigvalues actually", np.linalg.eigvals(A))

#%% Question 6(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST
print("---------------------------------------------")
A = np.array([[2, -1, 0,0],[-1, 2, -1,0],[0,-1, 2, -1],[0,0,-1,2]])
v0 = 1/np.sqrt(10)*np.array([[1],[2],[1],[2]])
kmax = 8
print("shape of A",np.shape(A))
print("shape of v",np.shape(v0))
print("---------------------------------------------")
v_array, eigval_array, r_array = LST.RayleighQuotientIteration(A,v0,kmax)
print("\nRayleigh Quotient Iteration")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("---------------------------------------------")
print("v_array shape",np.shape(v_array))
print("eigval_array shape",np.shape(eigval_array))
print("r_array shape",np.shape(r_array))
print("The eigvalues actually", np.linalg.eigvals(A))




#%% Question 7(a)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

# Set system
A = np.array([[2, -1,0],[-1, 2,-1],[0,-1,2]])
b = np.array([[1],[0],[5]])

# Solver settings
y0 = np.array([[0],[0],[0]])
kmax = 3

x_array, y_array = LST.CGAATmethod(A,b,y0,kmax)
print("x_array =\n",x_array)
print("y_array =\n",y_array)


#%% Question 7(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST


# Set system (nonsymmetric)
n = 100;
tau = .4
d = np.ones(n)
A = np.diag(d) - tau * ( \
      np.diag(np.ones(n-1), 1) \
    - np.diag(np.ones(n-1),-1) \
    + np.diag(np.ones(round(n/2)-1), round(n/2)+1) \
    - np.diag(np.ones(round(n/2)-1),-round(n/2)-1) \
    )
x = np.ones([n,1])
b = A @ x

# Solver settings
x0 = np.zeros([n,1])
y0 = np.zeros([n,1])
kmax = 25

# CG method
x_array_CG = LST.CGmethod_sol(A,b,x0,kmax)
print("Last x (CG) =\n",x_array_CG[:,-1])

# CGNE method
x_array_CGNE, z_array = LST.CGNEmethod(A,b,x0,kmax)
print("Last x (CGNE) =\n",x_array_CGNE[:,-1])

# CGAAT method
x_array_CGAAT, y_array = LST.CGAATmethod(A,b,y0,kmax)
print("Last x (CGAAT) =\n",x_array_CGAAT[:,-1])




#%% Question 8(a)


import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[1, 3],[2, 2]])
v0 = np.array([[1],[0]])
kmax = 12
p = 1
v_array, eigval_array, r_array, z_array = LST.EffPowerIterMethod(A,v0,kmax,p)
print("\nEfficient Power Method")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("z_array =\n",z_array)


#%% Question 8(b)

import numpy as np
import matplotlib.pyplot as plt
import LinSysTools as LST

A = np.array([[2, -1, 0,0],[-1, 2, -1,0],[0,-1, 2, -1],[0,0,-1,2]])
v0 = 1/np.sqrt(10)*np.array([[1],[2],[1],[2]])
kmax = 12
p = 2
v_array, eigval_array, r_array, z_array = LST.EffPowerIterMethod(A,v0,kmax,p)
print("\nEfficient Power Method")
print("v_array =\n",v_array)
print("eigval_array =\n",eigval_array)
print("r_array =\n",r_array)
print("z_array =\n",z_array)







