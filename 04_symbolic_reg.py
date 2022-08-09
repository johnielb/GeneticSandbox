"""
Build a GP system to automatically evolve a number of genetic programs for the following regression problem:
f(x) = { 1/x+sinx, x>0; 2x+x^2+3.0, x≤0 }

Inspired from https://github.com/DEAP/deap/blob/12ed30e0b8fc95c155f15dfce468b79e72153d74/examples/gp/symbreg.py
"""
import math
import operator
import random

import numpy
import numpy as np
from deap import creator, base, tools, algorithms, gp

verbose = True
mu = 100
p_cross = 0.85
p_mutate = 0.15
n_elite = 2
epochs = 100
init_min_depth = 2
init_max_depth = 6
max_depth = 15
mutate_min_depth = 2
mutate_max_depth = 6


def createPrimitiveSet():
    primitives = gp.PrimitiveSetTyped("main", [float], float, "in")
    # Numeric operators
    primitives.addPrimitive(operator.add, [float, float], float)
    primitives.addPrimitive(operator.sub, [float, float], float)
    primitives.addPrimitive(operator.mul, [float, float], float)
    # Add extra operators necessary to the problem
    primitives.addPrimitive(math.sin, [float], float)

    def protectedDiv(x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 1

    primitives.addPrimitive(protectedDiv, [float, float], float)

    def square(x):
        return x**2

    # Adding a power operator gave it too much power it couldn't handle (negative operands => complex)
    primitives.addPrimitive(square, [float], float)
    primitives.addEphemeralConstant("rand1", lambda: random.random() * 100, float)
    # Boolean operators, only 1 comparator (no need for equals) and an if-else node
    primitives.addPrimitive(operator.gt, [float, float], bool)

    def if_else(condition, opt1, opt2):
        return opt1 if condition else opt2

    primitives.addPrimitive(if_else, [bool, float, float], float)
    primitives.addTerminal(False, bool)
    primitives.addTerminal(True, bool)

    return primitives


def createToolbox():
    toolbox = base.Toolbox()
    # Generate individual using half (full) and half (grow) method
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitives, min_=init_min_depth, max_=init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # Generate population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitives)

    def evaluate(individual, points):
        f = toolbox.compile(expr=individual)

        def trueF(x):
            if x > 0:
                return 1 / x + math.sin(x)
            return 2 * x + x ** 2 + 3.0

        try:
            sse = math.fsum([(f(x) - trueF(x)) ** 2 for x in points])
        except ValueError:
            print(individual)
        # Return as a tuple for DEAP
        return sse / len(points),

    # Evaluate over x = {-10,-9,-8,...,8,9,10}
    toolbox.register("evaluate", evaluate, points=[x / 10. for x in range(-10, 10)])
    # Tournament selection, 3 participants
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    # Mutate new subtrees
    toolbox.register("expr_mut", gp.genFull, min_=mutate_min_depth, max_=mutate_max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    return toolbox


# Start with primitive set to feed into toolbox
primitives = createPrimitiveSet()
# Minimise the fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
# Set up toolbox to generate population
toolbox = createToolbox()


def main(seed=None):
    random.seed(seed)

    pop = toolbox.population(n=mu)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, p_cross, p_mutate, epochs, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof)

    # print log
    return pop, log, hof


if __name__ == '__main__':
    main(0)
