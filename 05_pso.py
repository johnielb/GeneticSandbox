"""
Optimise a particle swarm based on two objective functions, Rosenbrock's function and Griewanks' function

Inspired from https://github.com/DEAP/deap/blob/master/examples/pso/basic.py
"""
import math
import random
import operator

import numpy as np
from deap import base, creator, tools, benchmarks

verbose = False


def rosenbrock(individual):
    fitness = 0
    if any(x < -30 or x > 30 for x in individual):
        return np.inf,

    for d in range(0, len(individual) - 1):
        fitness += 100 * (individual[d] ** 2 - individual[d + 1] ** 2) ** 2 + (individual[d] - 1) ** 2
    return fitness,


def griewank(individual):
    summation = 0
    product = 1
    if any(x < individual.xmin or x > individual.xmax for x in individual):
        return np.inf,

    for d in range(0, len(individual)):
        summation += individual[d] ** 2
        product *= np.cos(individual[d] / np.sqrt(d+1))

    return summation / 4000 - product + 1,


def generate_particle(length):
    xmin = -30
    xmax = 30
    particle = creator.Particle(random.uniform(xmin, xmax) for _ in range(length))
    particle.xmin = xmin
    particle.xmax = xmax
    particle.vmin = -15
    particle.vmax = 15
    particle.speed = [random.uniform(particle.vmin, particle.vmax) for _ in range(length)]
    return particle


def update_particle(particle, gbest):
    # Set scalar parameters
    inertia_weight = 0.7298
    phi = 1.49618
    # Calculate vector variables
    r1 = (random.uniform(0, phi) for _ in range(len(particle)))
    r2 = (random.uniform(0, phi) for _ in range(len(particle)))
    cognitive = map(operator.mul, r1, map(operator.sub, particle.best, particle))
    social = map(operator.mul, r2, map(operator.sub, gbest, particle))
    # Calculate velocities adding it all together
    particle.speed = list(map(operator.add,
                                 [v * inertia_weight for v in particle.speed],
                                 map(operator.add, cognitive, social)))
    # Limit extreme velocities
    for i, v in enumerate(particle.speed):
        if v < particle.vmin:
            particle.speed[i] = particle.vmin
        elif v > particle.vmax:
            particle.speed[i] = particle.vmax
    # Apply velocity on position
    particle[:] = list(map(operator.add, particle, particle.speed))


def create_toolbox(ind_length, evaluate):
    toolbox = base.Toolbox()
    # Generate individual of variable length with random position and velocity
    toolbox.register("particle", generate_particle, ind_length)
    # Generate population as list of particles
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle)
    # Associate variable objective
    toolbox.register("evaluate", evaluate)
    return toolbox


def evolve_swarm(pop, stats, toolbox):
    epochs = 100
    # One global best, no local best, fully connected topology
    gbest = None

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    e = 0
    # Stop if we've reached a good enough fitness (20) or a cutoff of epochs
    while e < epochs and (gbest is None or gbest.fitness.values[0] > 20):
        for particle in pop:
            particle.fitness.values = toolbox.evaluate(particle)
            if not particle.best or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
            if not gbest or gbest.fitness < particle.fitness:
                gbest = creator.Particle(particle)
                gbest.fitness.values = particle.fitness.values
        for particle in pop:
            toolbox.update(particle, gbest)

        if verbose:
            logbook.record(gen=e, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

        e += 1

    return gbest


def repeat_experiment(toolbox, repeats):
    mu = 1000
    best_fitnesses = []
    for r in range(repeats):
        pop = toolbox.population(n=mu)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)

        best = evolve_swarm(pop, stats, toolbox)
        print("Repeat %2d: %0.2f" % (r, best.fitness.values[0]))
        best_fitnesses.append(best.fitness.values[0])

    print("Mean best fitness: " + str(np.mean(best_fitnesses)))
    print("SD best fitness: " + str(np.std(best_fitnesses)))


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
               xmin=None, xmax=None, vmin=None, vmax=None, best=None)


def main(seed=None):
    repeats = 30

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Rosenbrock function at D = 20
    print("# Rosenbrock function")
    toolbox = create_toolbox(20, rosenbrock)
    repeat_experiment(toolbox, repeats)

    print("# Griewank function")
    # Griewank function at D = 20 and D = 50
    for d in [20, 50]:
        print("## D = " + str(d))
        toolbox = create_toolbox(d, griewank)
        repeat_experiment(toolbox, repeats)


if __name__ == "__main__":
    main(seed=0)
