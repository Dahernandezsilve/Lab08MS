import random

class Knapsack:
    def __init__(self, values, weights, maxWeight, pop_size, mutationRate, generations):
        self.values = values
        self.weights = weights
        self.maxWeight = maxWeight
        self.pop_size = pop_size
        self.mutationRate = mutationRate
        self.generations = generations

    def genRandomSolution(self, size):
        solution = [random.randint(0, 1) for _ in range(size)]
        print(f"🔧 Generando solución aleatoria: {solution}")
        return solution

    def genInitialPopulation(self, popSize, size):
        population = [self.genRandomSolution(size) for _ in range(popSize)]
        print(f"👥 Población inicial generada con {len(population)} individuos.")
        return population

    def fitness(self, solution, values, weights, maxWeight):
        weight = 0
        value = 0
        for i in range(len(solution)):
            if solution[i] == 1:
                weight += weights[i]
                value += values[i]
        if weight > maxWeight:
            print(f"⚖️ Exceso de peso: {weight} > {maxWeight}. Fitness = 0.")
            return 0
        print(f"🧮 Calculando fitness: valor={value}, peso={weight}")
        return value

    def basicSelection(self, population, values, weights, maxWeight):
        best = None
        bestFitness = 0
        for solution in population:
            fit = self.fitness(solution, values, weights, maxWeight)
            if fit > bestFitness:
                best = solution
                bestFitness = fit
        print(f"🏅 Mejor solución seleccionada: {best} con fitness {bestFitness}")
        return best

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 2)
        child = parent1[:point] + parent2[point:]
        print(f"🔀 Cruce entre {parent1} y {parent2}, punto: {point}. Hijo: {child}")
        return child

    def mutate(self, solution, mutationRate):
        print(f"🧬 Mutando solución original: {solution}")
        for i in range(len(solution)):
            if random.random() < mutationRate:
                solution[i] = 1 - solution[i]
                print(f"⚡️ Mutación en el índice {i}: {solution}")
        return solution

    def geneticAlgorithm(self, values, weights, maxWeight, popSize, mutationRate, generations):
        population = self.genInitialPopulation(popSize, len(values))
        

        for gen in range(generations):
            print(f"\n🌱 Generación {gen+1} iniciada...")
            newPopulation = []

            for _ in range(popSize):
                parent1 = self.basicSelection(population, values, weights, maxWeight) # Selección
                parent2 = self.basicSelection(population, values, weights, maxWeight) # Selección

                child = self.crossover(parent1, parent2) # Cruce

                child = self.mutate(child, mutationRate) # Mutacion

                newPopulation.append(child)

            population = newPopulation
            print(f"📈 Población evolucionada a la generación {gen+1}.")

        best = self.basicSelection(population, values, weights, maxWeight)
        bestValue = self.fitness(best, values, weights, maxWeight)
        bestWeight = sum([weights[i] for i in range(len(best)) if best[i] == 1])

        print(f"\n🎉 Mejor solución final: {best}")
        print(f"🏆 Valor total: {bestValue}")
        print(f"⚖️  Peso total: {bestWeight}")
        
        return best, bestValue, bestWeight

# Ejemplo de uso
values = [10, 20, 30, 40, 50]
weights = [1, 2, 3, 4, 5]
maxWeight = 6
popSize = 100
mutationRate = 0.01
generations = 10

knapsack = Knapsack(values, weights, maxWeight, popSize, mutationRate, generations)
knapsack.geneticAlgorithm(values, weights, maxWeight, popSize, mutationRate, generations)
