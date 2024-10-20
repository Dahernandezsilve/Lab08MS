import numpy as np
import random
import os
import gzip
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('seaborn-v0_8-deep')

if not os.path.exists('ch150.tsp'):
    print("📦 Descomprimiendo archivo .gz...")
    with gzip.open('tspFiles/ch150.tsp.gz', 'rb') as f_in:
        with open('ch150.tsp', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("✅ Archivo descomprimido con éxito.")
else:
    print("ℹ️ El archivo ya está descomprimido, no es necesario descomprimir nuevamente.")

def readFile(filePath):
    cities = []
    with open(filePath, 'r') as f:
        lines = f.readlines()
        isCoordSection = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                isCoordSection = True
                continue
            if line.startswith("EOF"):
                break
            if isCoordSection:
                parts = line.strip().split()
                cityNumber = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                cities.append((cityNumber, x, y))
    return np.array(cities)

cities = readFile('ch150.tsp')

def calculateDistanceMatrix(cities):
    numCities = len(cities)
    distanceMatrix = np.zeros((numCities, numCities))
    for i in range(numCities):
        for j in range(i, numCities):
            distance = np.linalg.norm(cities[i, 1:] - cities[j, 1:])
            distanceMatrix[i, j] = distance
            distanceMatrix[j, i] = distance
    return distanceMatrix

distanceMatrix = calculateDistanceMatrix(cities)

class GeneticAlgorithmTSP:
    def __init__(self, distanceMatrix, cities, popSize, mutationRate, generations):
        self.distanceMatrix = distanceMatrix
        self.cities = cities
        self.popSize = popSize
        self.mutationRate = mutationRate
        self.generations = generations
        self.numCities = len(cities)
        self.population = self.createPopulation()

    def createPopulation(self):
        return [random.sample(range(self.numCities), self.numCities) for _ in range(self.popSize)]

    def fitness(self, route):
        distance = sum([self.distanceMatrix[route[i], route[i+1]] for i in range(len(route) - 1)])
        distance += self.distanceMatrix[route[-1], route[0]]
        return distance

    def selection(self):
        fitnessScores = np.array([self.fitness(route) for route in self.population])
        probabilities = fitnessScores.max() - fitnessScores + 1e-6
        probabilities /= probabilities.sum()
        selectedIndices = np.random.choice(np.arange(self.popSize), size=self.popSize // 2, replace=False, p=probabilities)
        return [self.population[i] for i in selectedIndices]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = city
        return child

    def mutate(self, route):
        for i in range(len(route)):
            if random.random() < self.mutationRate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def evolve(self):
        bestRoute = self.bestRoute()
        selected = self.selection()
        newPopulation = selected[:]
        while len(newPopulation) < self.popSize - 1:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            newPopulation.append(child)
        newPopulation.append(bestRoute)
        self.population = newPopulation

    def bestRoute(self):
        return min(self.population, key=lambda route: self.fitness(route))

def update(frame, ga, scat, line):
    ga.evolve()
    bestRoute = ga.bestRoute()
    x = [cities[city][1] for city in bestRoute] + [cities[bestRoute[0]][1]]
    y = [cities[city][2] for city in bestRoute] + [cities[bestRoute[0]][2]]
    scat.set_offsets(np.c_[x, y])
    line.set_data(x, y)
    ax.set_title(f"Generación: {frame+1} - Mejor Distancia: {ga.fitness(bestRoute):.2f}")
    return scat, line

popSize = 100
mutationRate = 0.01
generations = 10000
ga = GeneticAlgorithmTSP(distanceMatrix, cities, popSize, mutationRate, generations)

fig, ax = plt.subplots(figsize=(10, 6))
x = cities[:, 1]
y = cities[:, 2]
scat = ax.scatter(x, y, color='blue', s=40, edgecolor='k', zorder=2)
line, = ax.plot([], [], 'r-', lw=2, zorder=1)

ax.set_xlim(min(x)-10, max(x)+10)
ax.set_ylim(min(y)-10, max(y)+10)

ax.set_title('Optimización del TSP usando Algoritmo Genético', fontsize=14)
ax.set_xlabel('Coordenada X')
ax.set_ylabel('Coordenada Y')
ax.grid(True)

ani = animation.FuncAnimation(fig, update, frames=generations, fargs=(ga, scat, line), interval=50, repeat=False)
plt.show()
