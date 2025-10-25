# La idea es implementar un algoritmo de boosting simple desde cero en Python basado en el paper
# "A new perpective on Boosting un linear Refression via" en la carpeta de Papers.
# Para esto tenemos entonces: Una matriz X de datos en R^{nxp}, un vector de respuestas y en R^{n}
# Y un coeficiente de regresión b en R^{p}.

# Asumimos que los features han sido centrados y todos tienen norma 2 igual a 1. y que y tambien ha sido
# centrado para tener promedio 0.

# Buscamos entonces encontrar b tal que y = Xb, así que definimos el residuo r = y - Xb.

import numpy as np


def LS_Boost(X,y,numiter= 100, epsilon= 0.1):
    """
    Algoritmo de boosting para regresión lineal LS-Boost.
    Parámetros:
        X: Matriz de datos de tamaño (n,p).
        y: Vector de respuestas de tamaño (n,).
        numiter: Número de iteraciones del algoritmo, por defecto 100.
        epsilon: Tasa de aprendizaje, por defecto 0.1.
    Retorna:
        b: Vector de coeficientes de regresión de tamaño (p,).
    """
    if epsilon <= 0:
        raise ValueError("El parámetro epsilon debe ser positivo.")
    else:
        r,b = y, np.zeros(X.shape[1])
        for it in range(numiter):
            u_m = [np.dot(X[:,m],r)/np.linalg.norm(X[:,m])**2 for m in range(X.shape[1])]
            res = [np.sum((r - u_m[m]*X[:,m])**2) for m in range(X.shape[1])]
            j_k = np.argmin(res) 

            r = r - epsilon*X[:,j_k]*u_m[j_k]
            b[j_k] = b[j_k] + epsilon*u_m[j_k]

    return b


def FS_Boost(X,y,numiter = 100, epsilon = 0.1):
    """
    Algoritmo de boosting para regresión lineal FS-Boost.
    Parámetros:
        X: Matriz de datos de tamaño (n,p).
        y: Vector de respuestas de tamaño (n,).
        numiter: Número de iteraciones del algoritmo, por defecto 100.
        epsilon: Tasa de aprendizaje, por defecto 0.1.
    Retorna:
        b: Vector de coeficientes de regresión de tamaño (p,).
    """
    if epsilon <= 0:
        raise ValueError("El parámetro epsilon debe ser positivo.")
    else:
        r,b = y, np.zeros(X.shape[1])
        for it in range(numiter):
            corr = [abs(np.dot(r,X[:,m])) for m in range(X.shape[1])]
            j_k = np.argmax(corr)

            s = np.sign(np.dot(r,X[:,j_k]))
            r = r - epsilon*s*X[:,j_k]
            b[j_k] = b[j_k] + epsilon*s

    return b


def R_FS(X,y,numiter = 100, epsilon = 0.1, delta = 1):
    """
    Algoritmo de boosting para regresión lineal R-FS.
    Parámetros:
        X: Matriz de datos de tamaño (n,p).
        y: Vector de respuestas de tamaño (n,).
        numiter: Número de iteraciones del algoritmo, por defecto 100.
        epsilon: Tasa de aprendizaje, por defecto 0.1.
        delta: Parámetro de regularización, por defecto 1.
    Retorna:
        b: Vector de coeficientes de regresión de tamaño (p,).    
    """
    if epsilon <= 0:
        raise ValueError("El parámetro epsilon debe ser positivo.")
    elif delta <= 0:
        raise ValueError("El parámetro delta debe ser positivo.")
    elif epsilon >= delta:
        raise ValueError("El parámetro epsilon debe ser menor que delta.")
    else:
        r,b =  y,np.zeros(X.shape[1])
        for it in range(numiter+1):
            corr = [abs(np.dot(r,X[:,m])) for m in range(X.shape[1])]
            j_k = np.argmax(corr)

            s = np.sign(np.dot(r,X[:,j_k]))
            r = r - epsilon*(s*X[:,j_k] + (1/delta)*(r-y))
            b = (1 - epsilon/delta)*b
            b[j_k] += epsilon*s
            
    return b


def Path_R_FS(X,y,numiter = 100,epsilon = 0.1, delta_list = [0.001,0.01,0.1,1]):
    """
    Algoritmo de boosting para regresión lineal R-FS con ruta de regularización.
    Parámetros: 
        X: Matriz de datos de tamaño (n,p).
        y: Vector de respuestas de tamaño (n,).
        numiter: Número de iteraciones del algoritmo, por defecto 100.
        epsilon: Tasa de aprendizaje, por defecto 0.1.
        delta_list: Lista de valores de regularización para cada iteración, debe tener longitud numiter+1.
    Retorna:
        b: Vector de coeficientes de regresión de tamaño (p,).
    """
    if not isinstance(delta_list, list):
        raise ValueError("El parámetro delta_list debe ser una lista de valores positivos acotados.")
    if len(delta_list) != numiter+1:
        raise ValueError(f"delta_list debe contener {numiter+1} valores.")
    if any(d == 0 for d in delta_list):
        raise ValueError("Todos los valores en delta_list deben ser no nulos.")
    if epsilon <= 0:
        raise ValueError("El parámetro epsilon debe ser positivo.")
    if epsilon > np.min(delta_list):
        raise ValueError("El parámetro epsilon debe ser menor que el mínimo valor en delta_list.")
    else:
        delta_list = np.sort(np.abs(np.array(delta_list)))
        r,b =  y,np.zeros(X.shape[1])

        for it in range(numiter+1):
            corr = [abs(np.dot(r,X[:,m])) for m in range(X.shape[1])]
            j_k = np.argmax(corr)

            s = np.sign(np.dot(r,X[:,j_k]))
            r = r - epsilon*(s*X[:,j_k] + (1/delta_list[it])*(r-y))
            b = (1 - epsilon/delta_list[it])*b
            b[j_k] += epsilon*s

    return b

