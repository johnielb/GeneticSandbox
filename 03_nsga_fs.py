"""
Build a NSGA-II system to automatically evolve a feature subset that minimises error and the number of features selected.

Inspired from https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
"""
import random
import sys

import numpy
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from deap.tools._hypervolume import hv
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# HYPERPARAMETERS
mu = 50
p_cross = 0.75
p_mutate = 0.2
n_elite = 2
epochs = 200

global X, y
verbose = True


def load_data(fname, sep):
    if verbose:
        print("===", fname, "===")

    df = pd.read_csv(fname, header=None, sep=sep).dropna(axis=1)
    global X, y
    # Drop columns that aren't numeric
    X = df.iloc[:, 0:-1]\
        .apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna(axis=1)
    y_raw = df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y_raw)


def evaluate(individual):
    X_sub = X.loc[:, individual]
    if sum(individual) == 0:
        return 1, 1  # invalid solution

    clf = KNeighborsClassifier().fit(X_sub, y)
    y_pred = clf.predict(X_sub)
    error = 1 - (y == y_pred).sum() / len(y)
    sel_ratio = sum(individual) / len(X.columns)

    return error, sel_ratio


def one_point_crossover(i1, i2):
    ind_len = len(i1)
    idx = random.choice(range(1, ind_len - 1))
    fragment1 = i2[idx:ind_len]
    fragment2 = i1[idx:ind_len]
    i1[idx:ind_len] = fragment1
    i2[idx:ind_len] = fragment2
    return i1, i2


def mutate(individual):
    idx = random.choice(range(len(individual)))
    individual[idx] = not individual[idx]
    return individual,


# Minimise the two fitness metrics
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
# Individuals are represented as lists (of Booleans)
creator.create("Individual", list, fitness=creator.FitnessMin)


def createToolbox(ind_length):
    toolbox = base.Toolbox()
    # Generate attributes of the individual: pick a random candidate index to add to the set
    toolbox.register("attr_item", random.choice, [True, False])
    # Generate individual by repeatedly generating random as many Booleans there are features
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, ind_length)
    # Generate population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluate multi-objective, returning two metrics, accuracy and subset ratio
    toolbox.register("evaluate", evaluate)
    toolbox.register("mutate", mutate)
    toolbox.register("mate", one_point_crossover)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def control_fitness():
    control_ind = [True for _ in X.columns]
    return evaluate(control_ind)


def plot_pareto_front(hofs, fnames, seeds):
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    fig.suptitle("Pareto fronts")
    subfigs = fig.subfigures(nrows=len(fnames))

    for j, fname in enumerate(fnames):
        subfig = subfigs[j]
        axs = subfig.subplots(ncols=len(seeds), sharex=True, sharey=True)
        subfig.suptitle(fname)
        subfig.supxlabel("Classification error")
        subfig.supylabel("Proportion of features selected")

        for i, seed in enumerate(seeds):
            hof = hofs[i][j]
            plot = axs[i]

            weighted_fitness = numpy.array([ind.fitness.wvalues for ind in hof]) * -1
            ref = (1.1, 1.1)
            total_hv = hv.hypervolume(weighted_fitness, ref)

            points = list(map(lambda x: x.values, hof.keys))
            plot.plot(*zip(*points))
            plot.scatter(*zip(*points))
            plot.set(title="Seed=" + str(seed) + ", hv=" + str(round(total_hv, 5)))

    fig.show()


def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    hofs = []

    for f in range(1, len(sys.argv)):
        # start from 1, skip 0th argument - filename
        path = sys.argv[f]
        file = path.split("/")[-2]
        if path.endswith(".dat"):
            sep = " "
        else:
            sep = ","
        load_data(path, sep=sep)
        ind_length = len(X.columns)

        # Set up toolbox to generate population
        toolbox = createToolbox(ind_length)

        pop = toolbox.population(n=mu)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)

        hof = tools.ParetoFront()

        algorithms.eaMuPlusLambda(pop, toolbox, mu, mu, p_cross, p_mutate, epochs,
                                  stats, halloffame=hof)

        print("Control fitness: " + str(control_fitness()))

        hofs.append(hof)

    return hofs


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("usage: python 03_nsga_fs.py")
        print("e.g.:  python 03_nsga_fs.py data/vehicle/vehicle.dat data/musk/clean1.data")
        sys.exit(0)

    hofs = []

    seeds = range(3)
    for i in seeds:
        print("======= Seed =", i, "=======")
        hofs.append(main(i))

    plot_pareto_front(hofs, sys.argv[1:], seeds)