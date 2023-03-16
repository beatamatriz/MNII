from numpy import *
from numpy.linalg import *
from numpy import abs, sum, max, min

def mat_arange(k):
    return array([[k*i+j+1 for j in range(k)] for i in range(k)])

def hilbert(n):
    return array([[1/(i+j+1) for j in range(n)] for i in range(n)])

def vandermonde(n, alphas):
    return array([[alphas[i]**(j) for j in range(n)] for i in range(n)])

def conjugada(A):
    if len(A.shape) == 1 or A.shape[0] == 1:
        return conjugate(A.reshape(-1,1))
    elif A.shape[1] == 1:
        return conjugate(A.reshape(1,-1))
    else:
        return conjugate(transpose(A))
    
def norma_vec(X, p):
    inf_norm = max(abs((1.0 + 0j)*X))
    if p == inf:
        return inf_norm
    if p >= 1:
        if inf_norm > 1:
            return inf_norm*((sum((abs((1.0 + 0j)*X)/inf_norm)**p))**(1/p))
        else:
            return (sum(abs((1.0 + 0j)*X)**p))**(1/p)
    else:
        return "error"
def norma_mat(A, p):
    #arreglar esto
    return norm(A,p)

def descenso(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error descenso: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, :i]@X[:i, :]
        X[i, :] = X[i, :]/A[i, i]
    return True, X

def remonte(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error remonte: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error remonte: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n-1,-1,-1):
        X[i, :] = B[i, :]
        if i != n-1:
            X[i, :] -= A[i, i+1:]@X[i+1:, :]
        X[i, :] = X[i, :]/A[i, i]
    return True, X


def solve_diag(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error solve_diag: error en las dimensiones."
    if min(abs(diag(A))) < 1e-200:
        return False, "Error solve_diag: matriz singular."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]/A[i,i]
    return True, X


def gauss_pp(A, B=None, verbose=False):
    m, n = shape(A)
    if B is None:
        B = eye(m)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gauss_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n-1):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos+k
        if verbose:
            print("Procesando",str(k+1)+"-ésima iteración...")
            print("Pivote:", ik+1)
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(k+1, n):
                gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                gaussA[i, k+1:] -= gaussA[i, k]*gaussA[k, k+1:]
                gaussB[i, :] -= gaussA[i, k]*gaussB[k, :]
    if verbose:
        print("Nuevo sistema esquivalente A'X=B'")
        print("A' = ", triu(gaussA))
        print("B' = ", gaussB)
    exito, X = remonte(gaussA, gaussB)
    return exito, X

def gaussjordan_pp(A, B=None, verbose=False):
    m, n = shape(A)
    if B is None:
        B = eye(m)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error gaussjordan_pp: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        gaussA = array(A, dtype=complex)
        gaussB = array(B, dtype=complex)
    else:
        gaussA = array(A, dtype=float)
        gaussB = array(B, dtype=float)
    for k in range(n):
        pos = argmax(abs(gaussA[k:, k]))
        ik = pos+k
        if verbose:
            print("Procesando",str(k+1)+"-ésima iteración...")
            print("Pivote:", ik+1)
        if ik != k:
            gaussA[[ik, k], :] = gaussA[[k, ik], :]
            gaussB[[ik, k], :] = gaussB[[k, ik], :]
        if abs(gaussA[k, k]) >= 1e-200:
            for i in range(n):
                if i != k:
                    gaussA[i, k] = gaussA[i, k]/gaussA[k, k]
                    gaussA[i, k+1:] -= gaussA[i, k]*gaussA[k, k+1:]
                    gaussB[i, :] -= gaussA[i, k]*gaussB[k, :]
        
        else:
            return False, "Error gaussjordan_pp: matriz singular."
        
    if verbose:
        print("Nuevo sistema equivalente A'X=B'")
        print("A' = ", diag(gaussA))
        print("B' = ", gaussB)
    exito, X = solve_diag(gaussA, gaussB)
    return exito, X

def inverse_gauss(A):
    m, n = shape(A)
    if m!=n:
        return False, "Error inverse_gauss: error de dimensiones"
    exito, inv_A = gauss_pp(A, eye(m))
    if not exito:
        return exito, "Error inverse_gauss: matriz singular"
    else:
        return exito, inv_A
    
def inverse_gaussjordan(A):
    m, n = shape(A)
    if m!=n:
        return False, "Error inverse_gaussjordan: error de dimensiones"
    exito, inv_A = gaussjordan_pp(A, eye(m))
    if not exito:
        return exito, "Error inverse_gaussjordan: matriz singular"
    else:
        return exito, inv_A