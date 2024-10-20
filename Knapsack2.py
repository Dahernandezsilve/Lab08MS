from Knapsack import Knapsack

valores = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
pesos = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
maxWeight = 50
popSize = 100
mutationRate = 0.01
generations = 10
s = 0.5
c = 0.8

knapsack = Knapsack(valores, pesos, maxWeight, popSize, mutationRate, generations, s, c)
best, bestValue, bestWeight = knapsack.geneticAlgorithm(valores, pesos, maxWeight, popSize, mutationRate, generations, s, c)

print(f"Mejor solución: {best}")
print(f"Valor total: {bestValue}")
print(f"Peso total: {bestWeight}")