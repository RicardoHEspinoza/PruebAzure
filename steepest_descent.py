
import numpy as np
from math import *
import re
import sympy as sp

phi = (1.0 + sqrt(5.0))/2.0

# searches for best learning_rate
# learning_rate is named as val here
# we want to optimize the f(X - L*d) function here where L is learning rate
def golden_section_searcher(f, X, d, prev_val, lower, upper, epsilon):
    
    x1 = upper - ((phi - 1)*(upper - lower))
    x2 = lower + ((phi - 1)*(upper - lower))
    val = x1
    
    param2 = X - np.dot(x2, d)
    param2 = param2.tolist()
    
    param1 = X - np.dot(x1, d)
    param1 = param1.tolist()
    
    if f(*param2) < f(*param1):
        if x1 > x2:
            upper = x1
        else:
            lower = x1

    else:
        if x2 > x1:
            upper = x2
        else:
            lower = x2

    if abs(prev_val - val) <= epsilon:
        return val
    else:
        return golden_section_searcher(f, X, d, val, lower, upper, epsilon)


# derivation idea is similar to limit derivation
# but h doesnt go 0 it is close to zero
# for high dimensions, we take delf
# which has all derivatives
def derivate(f, X, variables):
    h = 0.0000001
    delf = []
    
    for i in range(len(X)):
        vals = X.copy()
        vals[i] = vals[i] + h
        delf.append((f(*vals) - f(*X))/h)
            
    return delf


def difference(X, Y):
    total = 0
    
    for i in range(len(X)):
        total = total + abs(X[i] - Y[i])
    total = total / len(X)
    

    return total


def normalizar_funcion(func_str):
    func_str = func_str.replace(" ", "")
    func_str = func_str.replace("^", "**")
    func_str = re.sub(r"\bx12\b", "x1**2", func_str)
    func_str = re.sub(r"\bx22\b", "x2**2", func_str)
    func_str = re.sub(r"\bx32\b", "x3**2", func_str)
    return func_str


def leer_entrada(mensaje):
    return input(mensaje).strip().lstrip("\ufeff")


def steepest_descent(f, X, variables, epsilon, maximize=False):
    
    if maximize:
        g = lambda *x: -f(*x)
    else:
        g = f

    print("{:<10} {:<25} {:<15} {:<25}".format("Iterador", "xi (punto actual)", "distancia (r)", "gradiente"))
    
    iterator = 1
    while True:
        d = derivate(g, X, variables)
        x_prev = X
        
        learning_rate = golden_section_searcher(g, X, d, 1, -10, 10, 0.0001)
        X = X - np.dot(learning_rate, d)
        X = X.tolist()
        
        distancia = np.linalg.norm(np.array(X) - np.array(x_prev))
        
        print("{:<10} {:<25} {:<15.6f} {:<25}".format(iterator, str(x_prev), distancia, str(d)))
        
        if difference(x_prev, X) < epsilon:
            print("{:<10} {:<25} {:<15.6f} {:<25}".format(iterator+1, str(X), 0.0, str(d)))
            print("\nPunto óptimo encontrado:", X)
            return X
        
        iterator += 1
        
    return X

def main():
    x1, x2 = sp.symbols('x1 x2')
    variables = (x1, x2)
    expr = 2*x1**2 - x1*x2 + x2**2 - 3*x1 + sp.exp(2*x1 + x2)
    f = sp.lambdify(variables, expr, 'numpy')

    punto_inicial = [0.0, 0.0]
    maximize = False

    steepest_descent(f, punto_inicial, variables, 0.00001, maximize)

if __name__ == "__main__":
    main()



