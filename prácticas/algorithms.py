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


def descenso1(A, B):
    m, n = shape(A)
    p, q = shape(B)
    if m != n or n != p or q < 1:
        return False, "Error descenso: error en las dimensiones."
    if A.dtype == complex or B.dtype == complex:
        X = zeros((n, q), dtype=complex)
    else:
        X = zeros((n, q), dtype=float)
    for i in range(n):
        X[i, :] = B[i, :]
        if i != 0:
            X[i, :] -= A[i, :i]@X[:i, :]
    return True, X


def gauss_pp(A, B=None, verbose=False, getTriu=False):
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
    if getTriu:
        return True, triu(gaussA)
    else:
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

def facto_lu(A):
    m,n = shape(A)
    if m!=n:
        return False, "Error facto_lu: error de dimensiones"
    if A.dtype == complex:
        LU = array(A, dtype=complex)
    else:
        LU = array(A, dtype=float)
    
    for k in range(n-1):
        if abs(LU[k, k]) >= 1e-200:
            for i in range(k+1, n):
                LU[i, k] = LU[i, k]/LU[k, k]
                LU[i, k+1:] -= LU[i, k]*LU[k, k+1:]
        else:
            return False, "Error facto_lu: no existe factorización"

    return True, LU



def metodo_lu(A, B=None):
    m,n = shape(A)
    if B is None:
        B = eye(m)
    p,q = shape(B)
    if p == m and q >= 1:
        success, LU = facto_lu(A)
        if success and abs(LU[m-1,m-1]) >= 1e-200:
            _, Y = descenso1(LU, B)
            _, X = remonte(LU, Y)
            return success, X
        else:
            return success, LU
    else:
        return False, "Error metodo_lu: error de dimensiones"
    
def inverse_lu(A):
    return metodo_lu(A)


def householder(Z):
    dim = Z.shape
    if len(dim) == 1:
        Z=conjugada(Z)
    m,n = Z.shape
    I = eye(m)
    H = I - 2*Z@conjugada(Z)/(conjugada(Z)@Z)
    return H


def zetaholder(X):
    dim = X.shape
    if len(dim) == 1:
        X=conjugada(X)
    m,n = X.shape
    E = zeros((m,n))
    E[0][0]+=1
    Z = X + norm(X)*pow(math.e,angle(X[0][0]))*E
    H = householder(Z)
    return H, householder(H@X + X)

def cond(X,p=1):
    _,X_ = inverse_lu(X)
    
    return norm(X,p)*norm(X_,p) 
    
    
def jacobi(A, B, XOLD, itermax=500, tol=1e-10):
    m, n = shape(A)
    p, q = shape(B)
    r, s = shape(XOLD)
    if m != n or n != p or q != 1 or n != r or s != 1 or min(abs(diag(A))) < 1e-200:
        return False, 'ERROR jacobi: no se resuelve el sistema.'
    k = 0
    error = 1.
    while k < itermax and error >= tol:
        k = k+1
        XNEW = array(B)
        for i in range(n):
            if i != 0:
                XNEW[i, 0] -= A[i, :i]@XOLD[:i, 0]
            if i != n-1:
                XNEW[i, 0] -= A[i, i+1:]@XOLD[i+1:, 0]
            XNEW[i, 0] = XNEW[i, 0]/A[i, i]
        error = norma_vec(XNEW - XOLD, inf)
        XOLD = array(XNEW)
    print('Iteración: k = ', k)
    print('Error absoluto: error = ', error)
    if k == itermax and error >= tol:
        return False, 'ERROR jacobi: no se alcanza convergencia.'
    else:
        print('Convergencia numérica alcanzada: jacobi.')
        return True, XNEW


    
def gauss_seidel(A, B, XOLD, itermax, tol):
    ...
    
def relajacion(A, B, XOLD, omega, itermax, tol):
    ...
    
def potencia(A, X, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potencia: no se ejecuta el programa.', 0, 0
    k = 0
    error = 1.
    normaold = 0.
    if A.dtype == complex or X.dtype == complex:
        lambdas = zeros(n, dtype=complex)
    else:
        lambdas = zeros(n, dtype=float)
    while k < itermax and error >= tol:
        k = k+1
        Y = A@X
        normanew = norm(Y, ord=norma)
        error = abs(normanew - normaold)
        for i in range(n):
            if abs(X[i, 0]) >= 1.e-100:
                lambdas[i] = Y[i, 0]/X[i, 0]
            else:
                lambdas[i] = 0.
        X = Y/normanew
        print('Iteración: k = ', k)
        print('Norma: ||A*X_k|| = ', normanew)
#        print('Lambdas: lambdas_k = \n', lambdas)
#        print('Vectores: X_k = \n', X)
        normaold = normanew
    if k == itermax and error >= tol:
        return False, 'ERROR potencia: no se alcanza convergencia.', 0, 0
    else:
        print('Método de la potencia: convergencia numérica alcanzada.')
        return True, normanew, lambdas, X    

    
def potenciainv(A, X, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n != r or s != 1:
        return False, 'ERROR potenciainv: no se ejecuta el programa.', 0, 0
    exito, LU = facto_lu(A)
    if not exito:
        return False, 'ERROR potenciainv: sin factorización LU.', 0, 0
    k = 0
    error = 1.
    normaold = 0.
    if A.dtype == complex or X.dtype == complex:
        lambdas = zeros(n, dtype=complex)
    else:
        lambdas = zeros(n, dtype=float)
    while k < itermax and error >= tol:
        k = k+1
        exito, Y = descenso1(LU, X)
        exito, Y = remonte(LU, Y)
        if not exito:
            return False, 'ERROR potenciainv: sin factorización LU.', 0, 0
        normanew = norm(Y, ord=norma)
        error = abs(normanew - normaold)
        for i in range(n):
            if abs(X[i, 0]) >= 1e-100:
                lambdas[i] = Y[i, 0]/X[i, 0]
            else:
                lambdas[i] = 0.
        X = Y/normanew
        print('Iteración: k = ', k)
        print('Norma: ||A-1*X_k|| = ', normanew)
#        print('Lambdas: lambdas_k = ', lambdas)
#        print('Vectores: X_k = ', X)
        normaold = normanew
    if k == itermax and error >= tol:
        return False, 'ERROR potenciainv: no se alcanza convergencia.', 0, 0
    else:
        print('Método de la potencia inversa: convergencia numérica alcanzada.')
        return True, normanew, lambdas, X
    
def potenciades(A, X, des, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n!= r or s != 1:
        return False, 'ERROR potenciades: no se ejecuta el programa.', 0, 0
    B = A - des*eye(n)
    exito, normanew, lambdas, X = potencia(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X 

def potenciadesinv(A, X, des, norma, itermax, tol):
    m, n = shape(A)
    r, s = shape(X)
    if m != n or n!= r or s != 1:
        return False, 'ERROR potenciadesinv: no se ejecuta el programa.', 0, 0
    B = A - des*eye(n)
    exito, normanew, lambdas, X = potenciainv(B, X, norma, itermax, tol)
    return exito, normanew, lambdas, X

"""
Glosario:

1. Definición de Arrays
==========================================================================================================================
eye(N[, M, k, dtype, order, like])            ::      matriz k-diagonal de 1oss     |   ||
identity(n[, dtype, like])                    ::      matriz identidad              ||  | __
ones(shape[, dtype, order, like])             ::      matriz rellena de unos (no repitdo la broma, que no hizo risa)
zeros(shape[, dtype, order, like])            ::      matriz nula
full(shape, fill_value[, dtype, order, like]) ::      matriz rellena (del valor fill_value)     
[...]_like(a[, dtype, order, subok, shape])   ::      matriz [...] con las mismas dimensiones que A
diag(v[, k])                                  ::      matriz k-diagonal a partir de un vector !no necesariamente, pero tal
diagflat(v[, k])                              ::      matriz k-diagonal interpretando v como vector
tri(N[, M, k, dtype, like])                   ::      triangular inferior de unos 
tril(m[, k])                                  ::      extraer la matriz triangular inferior
triu(m[, k])                                  ::      extraer la matriz triangular superior
vander(x[, N, increasing])                    ::      matriz de vandermonde (mejor que mi método feucho, supongo)
arange([start,] stop[, step,][, dtype, like]) ::      numeritos de un rango distribuidos con un paso metidos en un array



2. Operaciones con Arrays (numpy.linalg)
==========================================================================================================================

2.1 Productos
--------------------------------------------------------------------------------------------------------------------------
dot(a, b[, out])                              ::      producto escalar o matricial (creo) sin conjugar vectores complejos
multi_dot(arrays, *[, out])                   ::      cadena de productos matriciales (primer vector fila, último columna)
vdot(a, b, /)                                 ::      producto escalar o matricial con conjugado complejo
inner(a, b, /)                                ::      producto escalar interno (vectores)
outer(a, b[, out])                            ::      producto externo, diádico, tensorial, mágico, fantástico (muy útil)
matmul(x1, x2, /[, out, casting, order, ...]) ::      producto matricial (equivalentemente operador @)
matrix_power(a, n)                            ::      potencia de una matriz cuadrada
kron(a, b)                                    ::      producto de kronecker

2.2 Factorizaciones
--------------------------------------------------------------------------------------------------------------------------    
cholesky(a)                                   ::      factorización de cholesky
qr(a[, mode])                                 ::      factorización qr
svd(a[, full_matrices, compute_uv, ...])      ::      descomposición en valores singulares

2.3 Valores Propios
--------------------------------------------------------------------------------------------------------------------------
eig(a)                                        ::      valores y vectores propios (matriz cuadrada)
eigh(a[, UPLO])                               ::      valores y vectores propios (matriz hermitiana)
eigvals(a)                                    ::      valores propios
eigvalsh(a[, UPLO])                           ::      valores propios (matriz hermitiana)

2.4 Normas
--------------------------------------------------------------------------------------------------------------------------
norm(x[, ord, axis, keepdims])                ::      norma (vectorial o matricial)
cond(x[, p])                                  ::      condicionamiento de una matriz
det(a)                                        ::      determinante
matrix_rank(A[, tol, hermitian])              ::      rango (método de decomposición en valores singulares)
trace(a[, offset, axis1, axis2, dtype, out])  ::      la traza, como era de esperar

2.5 Sistemas de Ecuaciones
--------------------------------------------------------------------------------------------------------------------------
solve(a, b)                                   ::      resuelve un sistema de ecuaciones en forma matricial Ax = B
lstsq(a, b[, rcond])                          ::      resuelve el sistema lineal Ax = B con el método de mínimos cuadrados
inv(a)                                        ::      matriz inversa

"""