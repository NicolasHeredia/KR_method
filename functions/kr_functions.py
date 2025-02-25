import numpy as np
from collections import defaultdict
 
def log_points(n, min_val, max_val):
    """
    Genera `n` valores enteros distribuidos logarítmicamente entre `min_val` y `max_val`, 
    reduciendo la densidad excesiva en valores altos.
    
    Parámetros:
        n (int): Cantidad de valores a generar.
        min_val (int): Valor mínimo.
        max_val (int): Valor máximo.
    
    Retorna:
        np.ndarray: Array de `n` valores enteros distribuidos logarítmicamente.
    """
    log_values = np.geomspace(min_val, max_val, n)  # Usa geomspace para mejor distribución
    int_values = np.unique(np.round(log_values).astype(int))
    
    # Asegurar que sean exactamente `n` valores únicos
    while len(int_values) < n:
        k = n - len(int_values)
        extra_values = np.random.randint(min_val, max_val + 1, k)
        int_values = np.unique(np.concatenate((int_values, extra_values)))

    return int_values # Asegurar que la longitud sea exactamente `n`


def transformar_dict(a):
    a_dict = defaultdict(list)
    # Agrupar los datos, ignorando filas con el último valor = 0 y asegurando enteros
    for row in a:
        idx = int(row[0])  # Índice principal
        if row[3] != 0:  # Si el último valor NO es 0, lo agregamos
            a_dict[idx].append([int(row[1]), int(row[2]), row[3]])  # Asegurar enteros en las dos primeras columnas

    return a_dict


def factor_delta(a, b):
    # Para la matriz ai:
    mask = a[:, 1] == a[:, 2]  # Crea una máscara booleana donde p == l
    a[mask, 3] *= 0.5            # Multiplica la columna de valores por 0.5 en las filas filtradas

    # Para la matriz bi:
    mask = b[:, 1] == b[:, 2]
    b[mask, 3] *= 0.5

    return a, b

def ab_fraction(n):
    """
    Calcula la matriz de resultados con los índices, pares (p, l) y el valor de `ai`.
    Tambien calcula la misma matriz pero con bi.

    Args:
        n (array-like): Array de valores a procesar.

    Returns:
        numpy.ndarray: Matriz con las columnas [idx, p, l, ai].
        numpy.ndarray: Matriz con las columnas [idx, p, l, bi].
    """
    n = np.array(n)  # Asegurarse de que 'n' sea un array de numpy
    ai = []  # Lista para almacenar los resultados

    for p in range(len(n)):
        for l in range(p, len(n)):
            nj, nk = n[p], n[l]
            njk = nj + nk
            idx = np.searchsorted(n, njk, side="right") - 1  # Encontrar índice del intervalo
            if idx >= 0 and idx < len(n) - 1:
                if njk == n[idx]:
                    results = 0.0
                else:
                    results = (n[idx + 1] - (n[p] + n[l])) / (n[idx + 1] - n[idx])
                ai.append((idx, p, l, results))  # Agregar resultado a la lista

    ai = np.array(ai, dtype=object)  # Convertir la lista en un array NumPy

    ### FRACCION B ###

    # Filtrar las filas donde el primer elemento no sea 0
    filtered_results = ai[ai[:, 0] != 0]

    # Crear una copia para aplicar las transformaciones
    bi = np.copy(filtered_results)

    bi[:, 0] = bi[:, 0] + 1 
    bi[:, 3] = 1.0 - bi[:, 3]

    ### cuando k == i entonces hay que multiplicar por 0.5 ###
    ai, bi = factor_delta(ai, bi)

    ai_dicc = transformar_dict(ai)
    bi_dicc = transformar_dict(bi)

    return ai_dicc, bi_dicc

