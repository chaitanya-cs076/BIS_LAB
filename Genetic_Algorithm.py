f(x)=xsin(x)


import numpy as np
import random

# Define the problem: The function to optimize
def fitness_function(x):
    return x * np.sin(x)

# Genetic Algorithm parameters
population_size = 10
mutation_rate = 0.1
crossover_rate = 0.8
num_generations = 50
x_min, x_max = 0, 10  # Range of x

# Generate the initial population
def create_population(size, x_min, x_max):
    return np.random.uniform(x_min, x_max, size)

# Evaluate fitness for the entire population
def evaluate_fitness(population):
    return np.array([fitness_function(ind) for ind in population])

# Selection: Roulette wheel selection
def select_parents(population, fitness):
    # Ensure all fitness values are non-negative by adding a small constant if necessary
    fitness = fitness - np.min(fitness) + 1e-6  # Shift the fitness values to be positive
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness  # Normalize to sum to 1
    return population[np.random.choice(len(population), size=2, p=probabilities)]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(0, 1)  # Single-point crossover for simplicity
        return (parent1, parent2) if point == 0 else (parent2, parent1)
    return parent1, parent2

# Mutation: Apply random changes
def mutate(individual, mutation_rate, x_min, x_max):
    if random.random() < mutation_rate:
        mutation_value = np.random.uniform(-1, 1)
        individual += mutation_value
        individual = np.clip(individual, x_min, x_max)  # Ensure within bounds
    return individual

# Main Genetic Algorithm
def genetic_algorithm():
    population = create_population(population_size, x_min, x_max)
    best_solution = None
    best_fitness = -np.inf

    for generation in range(num_generations):
        fitness = evaluate_fitness(population)
        
        # Track the best solution
        max_fitness_index = np.argmax(fitness)
        if fitness[max_fitness_index] > best_fitness:
            best_fitness = fitness[max_fitness_index]
            best_solution = population[max_fitness_index]

        new_population = []
        for _ in range(population_size // 2):  # Produce new population
            parent1, parent2 = select_parents(population, fitness)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, mutation_rate, x_min, x_max)
            offspring2 = mutate(offspring2, mutation_rate, x_min, x_max)
            new_population.extend([offspring1, offspring2])

        population = np.array(new_population)

    return best_solution, best_fitness

# Run the Genetic Algorithm
best_solution, best_fitness = genetic_algorithm()
print(f"Best Solution: x = {best_solution}")
print(f"Best Fitness: f(x) = {best_fitness}")
