import pulp

# Definir los datos del problema
valores = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
pesos = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
K = 50
n = len(valores)

# Crear el problema de optimización
problema = pulp.LpProblem("Knapsack_Problem", pulp.LpMaximize)

# Variables binarias para representar si un objeto es seleccionado o no
x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

# Función objetivo: maximizar el valor total
problema += pulp.lpSum([valores[i] * x[i] for i in range(n)])

# Restricción: el peso total no debe exceder la capacidad de la mochila
problema += pulp.lpSum([pesos[i] * x[i] for i in range(n)]) <= K

# Resolver el problema
problema.solve()

# Obtener los resultados
objetos_seleccionados = [i for i in range(n) if x[i].value() == 1]
valor_total = sum(valores[i] for i in objetos_seleccionados)
peso_total = sum(pesos[i] for i in objetos_seleccionados)

print(f"Objetos seleccionados: {objetos_seleccionados}")
print(f"Valor total: {valor_total}")
print(f"Peso total: {peso_total}")
