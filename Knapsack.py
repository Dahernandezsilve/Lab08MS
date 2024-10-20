import random

class Knapsack:
    def __init__(self, values, weights, maxWeight, pop_size, mutationRate, generations, s, c):
        self.values = values
        self.weights = weights
        self.maxWeight = maxWeight
        self.pop_size = pop_size
        self.mutationRate = mutationRate
        self.generations = generations
        self.s = s
        self.c = c

    def genRandomSolution(self, size):
        solution = [random.randint(0, 1) for _ in range(size)]
        print(f"🔧 Generando solución aleatoria: {solution}")
        return solution

    def genInitialPopulation(self, popSize, size):
        population = [self.genRandomSolution(size) for _ in range(popSize)]
        print(f"👥 Población inicial generada con {len(population)} individuos.")
        return population

    def fitness(self, solution, values, weights, maxWeight):
        weight = sum([weights[i] for i in range(len(solution)) if solution[i] == 1])
        value = sum([values[i] for i in range(len(solution)) if solution[i] == 1])
        if weight > maxWeight:
            print(f"⚖️ Exceso de peso: {weight} > {maxWeight}. Fitness = 0.")
            return 0
        print(f"🧮 Calculando fitness: valor={value}, peso={weight}")
        return value

    def basicSelection(self, population, values, weights, maxWeight):
        sorted_population = sorted(population, key=lambda sol: self.fitness(sol, values, weights, maxWeight), reverse=True)
        num_selected = int(self.s * len(population))
        print(f"🔝 Seleccionando el {self.s * 100}% de la mejor población: {num_selected} individuos.")
        return sorted_population[:num_selected]

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

    def geneticAlgorithm(self, values, weights, maxWeight, popSize, mutationRate, generations, s, c):
        population = self.genInitialPopulation(popSize, len(values))

        for gen in range(generations):
            print(f"\n🌱 Generación {gen+1} iniciada...")
            
            selected_population = self.basicSelection(population, values, weights, maxWeight)

            newPopulation = []

            num_crossovers = int(c * popSize)

            for _ in range(num_crossovers):
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, mutationRate)
                newPopulation.append(child)

            while len(newPopulation) < popSize:
                newPopulation.append(random.choice(selected_population))

            population = newPopulation
            print(f"📈 Población evolucionada a la generación {gen+1}. Tamaño: {len(population)}")

        best = max(population, key=lambda sol: self.fitness(sol, values, weights, maxWeight))
        bestValue = self.fitness(best, values, weights, maxWeight)
        bestWeight = sum([weights[i] for i in range(len(best)) if best[i] == 1])

        print(f"\n🎉 Mejor solución final: {best}")
        print(f"🏆 Valor total: {bestValue}")
        print(f"⚖️  Peso total: {bestWeight}")
        
        return best, bestValue, bestWeight

values = [10, 20, 30, 40, 50]
weights = [1, 2, 3, 4, 5]
maxWeight = 6
popSize = 100
mutationRate = 0.01
generations = 10
s = 0.5
c = 0.8

knapsack = Knapsack(values, weights, maxWeight, popSize, mutationRate, generations, s, c)
knapsack.geneticAlgorithm(values, weights, maxWeight, popSize, mutationRate, generations, s, c)
