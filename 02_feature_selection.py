import time

import numpy as np
import pandas as pd
from numpy.random import choice
import sys
import random

from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# HYPERPARAMETERS
mu = 50
p_cross = 1
p_mutate = 0.25
n_elite = 2
epochs = 100

global X, y
verbose = True


def parse_data_file(fname):
    if verbose:
        print("===", fname, "===")

    df = pd.read_csv(fname, header=None)
    return df.iloc[:, 0:-1], df.iloc[:, -1]


def generate_population(length):
    population = []
    for i in range(mu):
        ind = []
        for j in range(length):
            ind.append(random.choice([True, False]))
        population.append(ind)

    return population


def filterGA(individual):
    X_sub = X.loc[:, individual]
    scores = mutual_info_classif(X_sub, y, discrete_features=False)
    return sum(scores)


def wrapperGA(individual):
    X_sub = X.loc[:, individual]
    clf = KNeighborsClassifier().fit(X_sub, y)
    y_pred = clf.predict(X_sub)
    return (y == y_pred).sum() / len(y)


def sort_population_filter(i1, i2):
    return filterGA(i2) - filterGA(i1)


def roulette_probabilities(population, evaluate):
    values = [evaluate(i) for i in population]
    vmin, vmax = min(values) - 1, max(values)  # -1 to allow the minimum value to occupy a non-zero slice
    norm = [(v - vmin) / (vmax - vmin) for v in values]
    return [x / sum(norm) for x in norm]


def one_point_crossover(i1, i2):
    ind_len = len(i1)
    idx = random.choice(range(1, ind_len - 1))
    c1 = i1[0:idx] + i2[idx:ind_len]
    c2 = i2[0:idx] + i1[idx:ind_len]
    return c1, c2


def mutate(individual):
    idx = random.choice(range(len(individual)))
    individual[idx] = not individual[idx]
    return individual


def breed_generation(population, evaluate):
    # Fitness evaluation of each individual
    population_scores = [evaluate(ind) for ind in population]
    population_sort = np.argsort(population_scores)[::-1]  # descending sort
    population = [population[i] for i in population_sort]

    # Do elitism (copy top individuals to the new generation)
    children = population[0:n_elite]

    # Generate probabilities for roulette wheel selection
    roulette = roulette_probabilities(population, evaluate)

    # Repeat until the new population is full:
    while len(children) < mu:
        parents = choice(range(len(roulette)), p=roulette, size=2, replace=False)
        child1, child2 = population[parents[0]], population[parents[1]]
        if random.random() < p_cross:
            child1, child2 = one_point_crossover(child1, child2)
        if random.random() < p_mutate:
            child1 = mutate(child1)
        if random.random() < p_mutate:
            child2 = mutate(child2)
        children += [child1, child2]

    return population, children


def update_solution(population, previous_solution, evaluate):
    candidate_solution = population[0]
    candidate_score = evaluate(candidate_solution)
    # Only if it's better than the current solution
    if previous_solution is None or candidate_score > filterGA(previous_solution):
        return candidate_solution
    return previous_solution


def test_feature_set(individual):
    X_sub = X.loc[:, individual]
    nb = GaussianNB().fit(X_sub, y)
    y_pred = nb.predict(X_sub)
    print("Accuracy:", (y == y_pred).sum() / len(y))


def main(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    for f in range(1, len(sys.argv)):  # start from 1, skip 0th argument - filename
        start = time.time()
        global X, y
        X, y = parse_data_file(sys.argv[f])
        ind_length = len(X.columns)

        for evaluate in [wrapperGA, filterGA]:
            print("=", evaluate.__name__, "=")
            # Randomly initialise a population of individuals (bit string, each bit has
            # 50% probability to be 1, and 50% to be 0)
            population = generate_population(ind_length)

            solution = None
            score_rolling_window = []

            for epoch in range(epochs):
                population, children = breed_generation(population, evaluate)

                # Update the best solution
                solution = update_solution(population, solution, evaluate)
                candidate_score = evaluate(solution)
                if verbose:
                    print("Epoch", epoch, "solution:", candidate_score)

                # Extra stopping criteria: if score is no better than last 10 epochs, stop
                if score_no_better_than_last(5, candidate_score, score_rolling_window):
                    break

                population = children

            print("Solution:", solution)
            print("Value:", filterGA(solution))
            print("Elapsed time (s):", time.time() - start)

            test_feature_set(solution)


def score_no_better_than_last(n, candidate_score, score_rolling_window):
    value = len(score_rolling_window) >= n and all(candidate_score <= score for score in score_rolling_window)
    score_rolling_window.append(candidate_score)
    if len(score_rolling_window) > n:
        score_rolling_window.pop(0)
    return value


if __name__ == '__main__':
    for i in range(5):
        print("======= Seed =", i, "=======")
        main(i)
