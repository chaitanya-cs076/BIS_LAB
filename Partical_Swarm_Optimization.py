#Travelling Salesman Problem

import numpy as np

# Function to calculate the total distance of a route (path)
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Return to start
    return total_distance

# Particle Swarm Optimization (PSO) for TSP
class PSO_TSP:
    def __init__(self, distance_matrix, num_particles=30, num_iterations=100, w=0.5, c1=1, c2=1):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient

        # Initialize particles' positions (routes) and velocities
        self.particles = np.array([np.random.permutation(self.num_cities) for _ in range(num_particles)])
        self.velocities = np.array([np.zeros(self.num_cities) for _ in range(num_particles)])

        # Evaluate fitness of each particle (route)
        self.fitness = np.array([calculate_total_distance(route, distance_matrix) for route in self.particles])

        # Initialize personal best positions and fitness
        self.p_best = np.copy(self.particles)
        self.p_best_fitness = np.copy(self.fitness)

        # Initialize global best position and fitness
        self.g_best = self.p_best[np.argmin(self.p_best_fitness)]
        self.g_best_fitness = np.min(self.p_best_fitness)

    # Update velocities and positions
    def update_particles(self):
        for i in range(self.num_particles):
            # Update velocity: w * velocity + c1 * random() * (personal best - current position) + c2 * random() * (global best - current position)
            r1 = np.random.rand(self.num_cities)
            r2 = np.random.rand(self.num_cities)
            cognitive_velocity = self.c1 * r1 * (self.p_best[i] - self.particles[i])
            social_velocity = self.c2 * r2 * (self.g_best - self.particles[i])
            inertia_velocity = self.w * self.velocities[i]
            self.velocities[i] = inertia_velocity + cognitive_velocity + social_velocity

            # To ensure we move to a new route, modify the velocity to shuffle positions
            velocity_order = np.argsort(self.velocities[i])  # Sort based on the velocity magnitude
            new_particle = np.array([self.particles[i][j] for j in velocity_order])

            # Ensure the new particle is a valid permutation
            self.particles[i] = new_particle
            self.fitness[i] = calculate_total_distance(new_particle, self.distance_matrix)

            # Update personal best
            if self.fitness[i] < self.p_best_fitness[i]:
                self.p_best[i] = self.particles[i]
                self.p_best_fitness[i] = self.fitness[i]

            # Update global best
            if self.fitness[i] < self.g_best_fitness:
                self.g_best = self.particles[i]
                self.g_best_fitness = self.fitness[i]

    # Run the PSO algorithm
    def run(self):
        for iteration in range(self.num_iterations):
            self.update_particles()
            print(f"Iteration {iteration + 1}: Best Distance = {self.g_best_fitness}")
        return self.g_best, self.g_best_fitness

# Function to take user input for distance matrix
def input_distance_matrix():
    num_cities = int(input("Enter the number of cities: "))
    print("Enter the distance matrix row by row (space-separated):")
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        row = list(map(int, input(f"Row {i + 1}: ").split()))
        distance_matrix[i] = row
    return distance_matrix

# Get user input for the distance matrix
distance_matrix = input_distance_matrix()

# Initialize PSO with the distance matrix provided by the user
pso_tsp = PSO_TSP(distance_matrix)

# Run PSO to find the shortest path
best_route, best_distance = pso_tsp.run()

print("\nBest route found:", best_route)
print("Best route distance:", best_distance)
