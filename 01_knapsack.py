import math
import sys
import random

from deap import creator, base, tools

verbose = True


def parse_data_file(fname):
    if verbose:
        print("===", fname, "===")
    file = open(fname)
    header = file.readline().split(" ")
    n = int(header[0])
    capacity = int(header[1])
    if verbose:
        print(n, "items, capacity", capacity)

    candidates = {}
    for i in range(n):
        line = file.readline().split(" ")
        # Append tuple to candidates, giving it a randomly accessible name i
        candidates[i] = (int(line[0]), int(line[1]))

    assert len(candidates) == n
    return candidates, capacity


def main():
    for i in range(1, len(sys.argv)):  # start from 1, skip 0th argument - filename
        candidates, capacity = parse_data_file(sys.argv[i])
        n = len(candidates)
        if verbose:
            print(candidates)

        # Below inspired from https://github.com/DEAP/deap/blob/b8513fc16fa05b2fe6b740488114a7f0c5a1dd06/examples/ga/knapsack.py

        # Set up the creator factory with the Fitness and Individual we want
        # Fitness: tuples are value (maximise +), weight (minimise -)
        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", set, fitness=creator.Fitness)

        # Set up toolbox to generate population
        toolbox = base.Toolbox()
        # Generate item randomly
        toolbox.register("attr_item", random.randrange, len(candidates))
        # Generate individual
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
        # Generate population of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(candidate):
            value = 0.0
            weight = 0.0
            for item in candidate:
                value += candidates[item][0]
                weight += candidates[item][0]

            if weight > capacity:  # override evaluation if overweight
                value = 0.0
                weight = math.inf

            return value, weight

        def crossover(c1, c2):
            temp = set(c1)
            c1 &= c2
            c2 ^= temp
            return c1, c2

        def mutate(candidate):
            if random.random() < 0.5:
                if len(candidate) > 0:  # If we can remove an item in set, remove
                    candidate.remove(random.choice(sorted(tuple(candidates))))
            else:
                candidate.add(random.randrange(n))
            return candidate,

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)



if __name__ == '__main__':
    main()
