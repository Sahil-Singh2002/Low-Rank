"""
MATH3036 CW2 module

@author: Sahil
"""

import numpy as np
import matplotlib.pyplot as plt

def SDmethod_sol(A,b,x0,kmax):
    """
    Implements the Steepest Descent (SD) method for solving Ax = b with symmetric 
    positive definite (SPD) matrix A.

    Args:
        A (ndarray): SPD coefficient matrix (size n x n).
        b (ndarray): Right-hand side vector (size n).
        x0 (ndarray): Initial guess vector (size n).
        kmax (int): Maximum number of iterations.

    Returns:
        ndarray: Array of shape (n, kmax+1), where each column is an 
        approximation of x at each iteration.
    """
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    
    # Store initial approximation
    x = x0
    x_array[:,[0]] = x
    
    # Initial r
    r = b-A@x0
         
    # SD loop
    for k in np.arange(kmax):
        # Step length
        a = (r.T@r) / (r.T@A@r)
        
        # Update approx
        x =  x+ a*r
        
        # Update residual
        r = b -A@x
        
        # Store
        x_array[:,[k+1]] = x

    # Return
    return x_array

def CGmethod_sol(A,b,x0,kmax):
    """
    Implements the Conjugate Gradient (CG) method for solving Ax = b with 
    symmetric positive definite (SPD) matrix A.

    Args:
        A (ndarray): SPD coefficient matrix (size n x n).
        b (ndarray): Right-hand side vector (size n).
        x0 (ndarray): Initial guess vector (size n).
        kmax (int): Maximum number of iterations.

    Returns:
        ndarray: Array of shape (n, kmax+1), where each column is an 
        approximation of x at each iteration.
    """
    # Initialize
    x_array = np.zeros([np.shape(x0)[0],kmax+1])
    
    # Store initial approximation
    x = x0
    x_array[:,[0]] = x
    
    # Initial r and p
    r_old = b - A @ x0
    p = r_old
    
    # CG loop
    for k in np.arange(kmax):

        # Step length
        if np.linalg.norm(p) < 1e-15:
            a = 0.0
        else:
            a = (r_old.T @ r_old) / (p.T @ A @ p)
            
        # Update approximation 
        x =  x + a * p
        
        # Update residual 
        r = r_old - a * A @ p
        
        # Update search direction
        if np.linalg.norm(r_old) < 1e-15:
            b = 0.0
        else:
            b = r.T @ r / (r_old.T @ r_old)
        p = r + b*p
        
        # Update r_old
        r_old = r
        
        # Store
        x_array[:,[k+1]] = x

    # Return
    return x_array

def PlotResidualsOfMethods(A,b,x0,kmax):
    """
    Plots the convergence of the 2-norm residuals ||b - Ax_k|| 
    over iterations for Steepest Descent (SD) and Conjugate Gradient (CG) methods.

    Args:
        A (ndarray): Coefficient matrix (size n x n).
        b (ndarray): Right-hand side vector (size n).
        x0 (ndarray): Initial guess vector (size n).
        kmax (int): Maximum number of iterations.

    Returns:
        tuple: (fig, ax, SD_res, CG_res). The figure and axis objects for the plot,
        and the residual arrays for both SD and CG.
    """
    # Set range of k
    k_range = np.arange(kmax+1)
    #set the row size for array
    Row =np.shape(x0)[0]
    # Get approximations of methods
    SD_array = SDmethod_sol(A,b,x0,kmax)
    CG_array = CGmethod_sol(A,b,x0,kmax)
    
    # Initialize vectors for
    # 2-norms of residuals of methods
    SD_r = np.zeros([kmax+1,1])
    CG_r = np.zeros([kmax+1,1])
    
    # Compute 2-norms of residuals
    for j in range(0,kmax+1):
#
        SD_r[j] = np.linalg.norm(
                                b - A @ SD_array[:, j].reshape(Row,1)
                                )

        CG_r[j] = np.linalg.norm(
                                b - A @ CG_array[:, j].reshape(Row,1)
                                )
    
    # Preparing figure, using Object -Oriented (OO) style; see:
    # https://matplotlib.org/stable/tutorials/introductory/quick_start.html 
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("||b - A x_k||")
    ax.set_title("Convergence of norms residuals")
    ax.grid(True)
    
    # Plot using following:
    ax.plot(k_range , CG_r , marker="o", label="CG Method", linestyle=":")
    ax.plot(k_range , SD_r , marker="x", label="SD Method", linestyle="-")

    # Add legend
    ax.legend()   
    
    SD_res = SD_r
    CG_res = CG_r
    
    return fig, ax, SD_res, CG_res

def CGNEmethod(A, b, x0, kmax):
    """
    Implements the Conjugate Gradient on Normal Equations (CGNE) method to solve 
    non-SPD systems via AtAx = Atb.

    Args:
        A (ndarray): Coefficient matrix (size n x n, not necessarily SPD).
        b (ndarray): Right-hand side vector (size n).
        x0 (ndarray): Initial guess vector (size n).
        kmax (int): Maximum number of iterations.

    Returns:
        tuple: (x_array, z_array). Arrays containing approximations of x and z 
        at each iteration (both of shape (n, kmax+1)).
    """
    
    # Shape size
    x_array = np.zeros((x0.shape[0], kmax + 1))
    z_array = np.zeros((x0.shape[0], kmax + 1))
    p = np.zeros((x0.shape[0], kmax + 1))
    r = np.zeros((x0.shape[0], kmax + 1))
    
    # Initialize values
    x_array[:, 0] = x0[:, 0]
    r[:, 0] = b[:, 0]
    p[:, 0] = A.T @ r[:, 0]
    z_array[:, 0] = p[:, 0]
    
    for k in range(1, kmax + 1):
        # Step length update
        alpha = np.dot(z_array[:, k - 1], z_array[:, k - 1]) / np.dot(A @ p[:, k - 1], A @ p[:, k - 1])
        
        # Conjugate gradient descent
        x_array[:, k] = x_array[:, k - 1] + alpha * p[:, k - 1]
        
        # Residual update
        r[:, k] = r[:, k - 1] - alpha * (A @ p[:, k - 1])
        
        # to reduce computation error of rounding off. to find beta it is splite
        # into two lines so you can get better accuracy
        z_array[:, k] = A.T @ r[:, k]
        beta = np.dot(z_array[:, k], z_array[:, k]) / np.dot(z_array[:, k - 1], z_array[:, k - 1])
        
        # Search direction update
        p[:, k] = z_array[:, k] + beta * p[:, k - 1]
    
    return x_array, z_array

def PowerIterationMethod(A,v0,kmax):
    """
    Uses the Power Iteration method to approximate the dominant 
    eigenvalue/eigenvector pair of a given matrix A.

    Args:
        A (ndarray): Input matrix (size m x m).
        v0 (ndarray): Initial approximation vector (size m).
        kmax (int): Maximum number of iterations.

    Returns:
        tuple: (v_array, eigval_array, r_array). Arrays containing 
        approximations of eigenvectors, eigenvalues, and residuals.
    """
    v_array = np.zeros((v0.shape[0], kmax))
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    
    # Normalize the initial vector
    v = v0 / np.linalg.norm(v0)
    
    for k in range(kmax):
        
        #eigenvector approximation
        v_next = np.dot(A, v) / np.linalg.norm( np.dot(A, v) )
        v = v_next
        
        #ensuring v_next is treated as a 1D array
        v_array[:, k] = v_next.flatten() 
        
        #Rayleigh quotient
        eigval_array[k] = np.dot(v_array[:, k].T, np.dot(A, v_array[:, k]))
        r_array[k] = np.linalg.norm(np.dot(A, v_array[:, k]) - eigval_array[k] * v_array[:, k])
        
    return v_array, eigval_array, r_array
          

def RayleighQuotientIteration(A,v0,kmax):
    """
    Uses the Rayleigh Quotient Iteration method to approximate an 
    eigenvalue/eigenvector pair of a given matrix A.

    Args:
        A (ndarray): Input matrix (size m x m).
        v0 (ndarray): Initial approximation vector (size m).
        kmax (int): Maximum number of iterations.

    Returns:
        tuple: (v_array, eigval_array, r_array). Arrays containing 
        approximations of eigenvectors, eigenvalues, and residuals.
    """
    
    # Initialize
    m = np.shape(v0)[0]
    v_array = np.zeros([m,kmax])
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    
    # Initial eigenvector 
    v = v0
    
    # Initial eigenvalue
    eigval = v0.T @ A @ v0
    
    
    for k in np.arange(kmax):
        
        # A - eigval I
        B = A - eigval * np.eye(m)
    
        # Solve Bw = v
        w = np.linalg.solve(B,v)
        
        # Normalize
        v = w / np.linalg.norm(w,2)
        
        # Rayleigh quotient
        eigval = v.T @ A @ v

        # Residual
        r = np.linalg.norm(eigval*v - A@v,2)
        
        # Store
        v_array[:,[k]] = v
        eigval_array[k] = eigval
        r_array[k] = r
        
    return v_array, eigval_array, r_array


def CGAATmethod(A,b,y0,kmax):
    """
    Implements the Conjugate Gradient method for the system A(A)^T * y = b to 
    solve non-SPD systems.

    Args:
        A (ndarray): Coefficient matrix (size n x n, not necessarily SPD).
        b (ndarray): Right-hand side vector (size n).
        y0 (ndarray): Initial guess vector (size n).
        kmax (int): Maximum number of iterations.

    Returns:
        tuple: (x_array, y_array). Arrays containing approximations of x and y 
        at each iteration (both of shape (n, kmax+1)).
    """
    x_array = np.zeros((b.shape[0], kmax+1))
    y_array = np.zeros((b.shape[0], kmax+1))
    
    y = y0
    # Initial residual
    r = b - np.dot(A, np.dot(A.T, y))
    # Initial direction
    p = r
    
    y_array[:, 0] = y.flatten()
    
    for k in range(1, kmax+1):
        Ap = np.dot(A, np.dot(A.T, p))
        rTr = np.dot(r.T, r)
        alpha = rTr / np.dot(p.T, Ap)
        # Update y
        y_next = y + alpha * p
        # Update residual
        r_next = r - alpha * Ap
        beta = np.dot(r_next.T, r_next) / rTr
        # Update direction
        p_next = r_next + beta * p
        # Compute x from y
        x = np .dot(A.T, y_next)
        
        # Store
        x_array[:, k] = x.flatten()
        y_array[:, k] = y_next.flatten()
        
        # Prep. next iteration
        y = y_next
        r = r_next
        p = p_next
    
    return x_array, y_array


def EffPowerIterMethod(A, v0, kmax, p):
    """
    Uses an efficient variant of the Power Iteration method to approximate a 
    dominant eigenvalue/eigenvector pair of a given matrix A.

    Args:
        A (ndarray): Input matrix (size m x m).
        v0 (ndarray): Initial approximation vector (size m).
        kmax (int): Maximum number of iterations.
        p (int): Cut-off to determine the number of components to update.

    Returns:
        tuple: (v_array, eigval_array, r_array, z_array). Arrays containing 
        approximations of eigenvectors, eigenvalues, residuals, and z values.
    """
    # Initialize arrays
    v_array = np.zeros((A.shape[0], kmax + 1))
    eigval_array = np.zeros(kmax)
    r_array = np.zeros(kmax)
    z_array = np.zeros((A.shape[0], kmax))

    # Set the initial vector
    v_array[:, 0] = v0.flatten()

    for k in range(kmax):
        # Compute z^(k)
        z = A @ v_array[:, k]
        z_array[:, k] = z

        # Compute the e vector
        e = v_array[:, k] - (z / (v_array[:, k].T @ z))

        # Compute  and set initualy
        y = np.zeros(A.shape[0])
        
        #--------------------------------------------------
        # Condition for efficiency 
        #--------------------------------------------------
        # compute the largest indexes
        abs_e_j = np.abs(e)
        largest_indexes = np.argsort(abs_e_j)[-p:]

        for j in range(A.shape[0]):
            if j in largest_indexes:
                y[j] = z[j] / (v_array[:, k].T @ z)
            else:
                y[j] = v_array[:, k][j]
        
        #--------------------------------------------------
        # Compute v^(k)
        v_array[:, k + 1] = y / np.linalg.norm(y)

        # Compute lambda^(k)
        eigval_array[k] = (v_array[:, k + 1].T @ A @ v_array[:, k + 1])

        # Compute residual norm r^(k)
        r_array[k] = np.linalg.norm(A @ v_array[:, k + 1] - eigval_array[k] * v_array[:, k + 1])

    return v_array[:,1:], eigval_array, r_array, z_array