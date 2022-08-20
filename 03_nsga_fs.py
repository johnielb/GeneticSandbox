"""
Build a NSGA-II system to automatically evolve a feature subset that minimises error and the number of features selected.

Inspired from https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
"""
import sys
import time
import random
from array import array

import numpy
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms

# HYPERPARAMETERS
mu = 50
p_cross = 0.75
p_mutate = 0.2
n_elite = 2
epochs = 100

global X, y
verbose = True


def loadData(fname):
    if verbose:
        print("===", fname, "===")

    df = pd.read_csv(fname, header=None)
    global X, y
    # Drop columns that aren't numeric
    X = df.iloc[:, 0:-1]\
        .apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna(axis=1)
    y_raw = df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y_raw)


def evaluate(individual):
    X_sub = X.loc[:, individual]
    clf = KNeighborsClassifier().fit(X_sub, y)
    y_pred = clf.predict(X_sub)
    accuracy = (y == y_pred).sum() / len(y)
    sel_ratio = sum(individual) / len(X.columns)

    return accuracy, sel_ratio


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


def createToolbox(ind_length):
    # Minimise the two fitness metrics
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    # Individuals are represented as lists (of Booleans)
    creator.create("Individual", list, fitness=creator.FitnessMin)

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


def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    for f in range(1, len(sys.argv)):  # start from 1, skip 0th argument - filename
        start = time.time()
        loadData(sys.argv[f])
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

        return pop, stats, hof


if __name__ == '__main__':
    for i in range(5):
        print("======= Seed =", i, "=======")
        main(i)
