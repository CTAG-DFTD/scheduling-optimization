import time
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

def pulp_solver(maxcut_matrix):
    """
    Solver MaxCut usando PuLP con penalización de balance.
    Genera un gráfico por cada llamada y lo guarda como JPG en la carpeta results.
    El nombre del archivo y el título incluyen el número de día.
    """
    global call_counter
    call_counter += 1  # Incrementa el contador para identificar el día actual
    day = call_counter

    start_time = time.time()
    n = len(maxcut_matrix)

    # Crear modelo
    prob = LpProblem("MaxCut", LpMaximize)
    x = {i: LpVariable(f"x[{i}]", 0, 1, LpBinary) for i in range(n)}
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = LpVariable(f"y[{i},{j}]", 0, 1, LpBinary)
            prob += y[i, j] <= x[i]
            prob += y[i, j] <= x[j]
            prob += y[i, j] >= x[i] + x[j] - 1

    # Penalización por balance
    lambda_penalty = 0.15
    s_pos = LpVariable("s_pos", 0)
    s_neg = LpVariable("s_neg", 0)
    prob += s_pos - s_neg == lpSum(x[i] for i in range(n)) - (n / 2)
    s_abs = LpVariable("s_abs", 0)
    prob += s_abs == s_pos + s_neg

    # Función objetivo
    prob += lpSum(maxcut_matrix[i][j] * (x[i] + x[j] - 2 * y[i, j]) for i, j in y) - lambda_penalty * s_abs

    # Resolver
    prob.solve(PULP_CBC_CMD(msg=False))
    elapsed_time = time.time() - start_time

    # Construir bitstring
    bitstring = [int(round(x[i].value())) for i in range(n)]

    return bitstring, elapsed_time
