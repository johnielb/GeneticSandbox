---
title: AIML426 Project 1

author: Johniel Bocacao, 300490028

date: 25 July 2022

---

# Part 1

I designed this genetic algorithm from scratch. We start with a randomly initialised population, with **100** individuals represented by a list of Booleans. This representation encodes what an individual actually is, a series of Boolean choices whether to include an item in the knapsack, rather than say a string of ASCII integers. This representation also enables the use of Boolean operations, such as negation during mutation. 

Given this randomly initialised population, a generation process is repeated for **500** epochs. This high number was picked to accommodate the large size of the third file.
* The first step sorts the population by fitness. Fitness is determined by summing the value of the candidate, and penalising by how much it overshoots capacity (**alpha = 2**, constant). This penalty, instead of rejecting such an individual, promotes diversity in the generation giving the GA more material to work with. 
* Of course, this choice meant I had to distinguish between feasible individuals and the entire population. The stored solution can only be feasible individuals, unless there are none, in which case the fittest individual is stored.
* The next generation is populated with **2** of the most elite (most feasible, else most fit) individuals, before generating the rest of the children. This is 2% of the next generation, which promotes further diversity in the 98% which will possess different material, as we cross over with new children 100% of the time.
* Until **100** children in the population are generated, select two parents randomly, weighted by their fitness (i.e. roulette wheel selection). Crossover happens **100%** of the time, where a random index in the individual is picked as the crossover point where genetic material is swapped between the parents from that point onwards. Mutation happens **25%** of the time, which introduces novel genetic material by flipping a bit in the individual randomly. This rate is higher than suggested to help the GA converge on the optimal value quicker.

## Results
| Seed | 10_269 | 23_10000 | 100_995 |
| --- | --- | --- | --- | 
| 0 | 295 | 9767 | 1514 |
| 1 | 295 | 9767 | 1514 |
| 2 | 295 | 9767 | 1497 |
| 3 | 295 | 9767 | 1478 |
| 4 | 295 | 9767 | 1514 |
| Mean | 295 | 9767 | 1503 |
| Optima | 295 | 9767 | 1514 |
| Deviation from optima | 0 | 0 | 11 |
| Standard deviation | 0 | 0 | 14 |

## Convergence curves
### 10_269
![img.png](out/part1/img.png)

All top 5 solutions in the run with seed=0 converged to the optimal solution 295 by the 6th epoch. This is not unexpected as the parameters have been tuned aggressively (e.g. high 25% mutation), largely to accommodate the bigger datasets below. This GA also has a high number of individuals (100) in each population compared to how long each individual is (10), increasing the chances of finding an optimal solution relatively quickly. This success is reflected in the results table, with each 5 seeds resulting in the same optimal solution = 295. 

### 23_10000
![img_1.png](out/part1/img_1.png)

All top 5 solutions in the run with seed=0 converged to the optimal solution 9767 by the 67th epoch. This graph shows a greater amount of variance in fitness than the previous graph, with massive drops in fitness at the beginning and fluctuations around the optima towards the end, likely due to the higher candidate values (in the high hundreds) in the data. This was not a problem in converging onto the optimal value, with the results table showing each 5 seeds resulted in the same optimal solution = 9767.

### 100_995
![img_2.png](out/part1/img_2.png)

Early populations begin in the negative fitness territory as the randomly generated individuals massively overshoot in capacity. Around the 40th epoch, all seeds had top 5 solutions with positive fitness. By the 146th epoch, all top 5 solutions in the run with seed=4 converged to the optimal solution 1514. There is less variance than in the 2nd file, possibly due to a variety in item sizes in this 3rd file. 500 generations may not be enough for some seeds to converge to the optimal value, with two runs (seed 2, seed 3) failing to meet the optimal value.


# Part 2

This genetic algorithm is largely the same as in Part 1, with parameters readjusted for the specific problem. We start with a randomly initialised population, with **50** individuals represented by a list of Booleans. This representation encodes what an individual actually is, a series of Boolean choices whether to include a feature in the set, rather than say a string of ASCII integers. This representation also enables the use of Boolean operations, such as negation during mutation. 

Given this randomly initialised population, a generation process is repeated for **100** epochs, or when the score hasn't improved in 10 epochs. This is a smaller number than in Part 1 to reflect the smaller variability in this problem.
* The first step sorts the population by fitness. Fitness is determined either by filter feature selection, where the mutual information criterion of each selected feature is summed together as the fitness value; or wrapper feature selection, where a classifier is trained on the selected features, using its accuracy as the fitness value. 
* The next generation is populated with **2** of the most fit individuals, before generating the rest of the children. This is 2% of the next generation, which promotes further diversity in the 98% which will possess different material, as we cross over with new children 100% of the time.
* Until **100** children in the population are generated, select two parents randomly, weighted by their fitness (i.e. roulette wheel selection). Crossover happens **100%** of the time, where a random index in the individual is picked as the crossover point where genetic material is swapped between the parents from that point onwards. Mutation happens **25%** of the time, which introduces novel genetic material by flipping a bit in the individual randomly. This rate is higher than suggested to help the GA converge on the optimal value quicker.

## Results
### wbcd
| Seed | Wrapper time (s) | Filter time (s) | Wrapper accuracy (%) | Filter accuracy (%) |
| --- | --- | --- | --- |  --- | 
| 0 | 8 | 125 | 0.928 | 0.942 |
| 1 | 8 | 153 | 0.931 | 0.942 |
| 2 | 8 | 129 | 0.931 | 0.942 |
| 3 | 8 | 167 | 0.935 | 0.942 |
| 4 | 10 | 226 | 0.951 | 0.942 |
| Mean | 8.4 | 160 | 0.935 | 0.942 |
| SD | 0.8 | 36.4 | 0.008 | 0 |

### sonar
| Seed | Wrapper time (s) | Filter time (s) | Wrapper accuracy (%) | Filter accuracy (%) |
| --- | --- | --- | --- |  --- | 
| 0 | 4 | 118 | 0.635 | 0.692 |
| 1 | 4 | 52 | 0.682 | 0.731 |
| 2 | 4 | 131 | 0.726 | 0.706 |
| 3 | 4 | 93 | 0.673 | 0.712 |
| 4 | 12 | 131 | 0.678 | 0.702 |
| Mean | 5.6 | 104.8 | 0.679 | 0.709 |
| SD | 3.2 | 30 | 0.029 | 0.013 |

The average computational time for wrapper FS was 8.4 seconds for wbcd, and 5.6 seconds for sonar. These averages are significantly smaller than the average computational time for filter FS, 160 seconds for wbcd, and 105 seconds for sonar. This result is surprising, as typically one expects wrapper FS to take longer than filter FS, where you need to create an entire model to evaluate a feature subset for the former, while the latter simply evaluates the feature subset mathematically. 

This result may be explained by the simple model used for wrapper FS, a KNeighbours classifier, and may be significantly optimised by the package. On the other hand, filter FS used the mutual information score, which may have been computationally expensive due to the discretisation of all the continuous features. One factor not considered in these results is the number of epochs before stopping. Typically, wrapper FS stopped quicker, while filter FS could continue to make incremental gains. Perhaps a more aggressive stopping criteria, maybe one that considers the size of the gains, could be used to shorten filter FS' elapsed time.

The average accuracy for wbcd was 93.5% for wrapper FS, and 94.2% for filter FS. For sonar, accuracy was 67.9% for wrapper FS, and 70.9% for filter FS. Filter FS tended to have a marginally better accuracy for both datasets. This result is in line with what I expected, as wrapper FS has poorer generalisability to other classifier models (here, moving from KNeighbours to GaussianNB), while filter FS is more generalisable due to its direct relationship with the data rather than a model's output. Had we used a GaussianNB as the wrapper classifier, we might have seen a different result with a better accuracy more suited to the final GaussianNB model.

# Part 3

Once again, individuals are represented as a list of Booleans to semantically encode what an individual actually does, a series of Boolean choices to include a feature in or not, and allows an individual to directly subset features. The fitness function is in two parts, 

Given this randomly initialised population, a generation process is repeated for **200** epochs. This is a smaller number than in Part 1 to reflect the smaller variability in this problem.
* The first step sorts the population by fitness. Fitness is determined by two objectives: classification error based on a wrapper-based evaluation of the feature subset as in Part 2 using a KNeighborsClassifier, and selection ratio based on how many features an individual selected. Both are minimised. 
* The second step generates **100** offspring by either crossover (75% of the time), mutation (20% of the time) or reproduction (rest of the time, 5%).
* The third step selects the best **100** individuals from the previous and this new generation. Selection is performed in two sub-steps, first sorting by the different non-dominated ranks, then assigning a crowding distance to each individual based on how many other individuals are close to it.

![img.png](out/part3/img.png)

The complete set of features for the vehicle dataset achieved a classification error rate of 21%. Most feature subsets on the three Pareto fronts achieved a lower classification error than that, improving by a maximum of 5 percentage points down. The complete set of features for the musk dataset achieved a classification error rate of 6%. All but one feature subset on three Pareto fronts achieved a lower classification error than that, improving by a maximum of 3 percentage points down.

This finding suggests that as the selection ratio increases, classification error begins to increase as adding more dimensions introduces more noise, making it harder to find a signal to aid classification. Removing features that don't provide useful information can improve accuracy, but only by 3-5%. 

Without using the selection ratio as an evolutionary objective, evolution may only stop at the top of the Pareto front with the lowest classification error. However, it may be preferable to continue removing features to decrease the model's complexity without re-increasing classification error, particularly for generalising to unseen data. To do this, one would select an individual before the elbow of the Pareto front starts.

# Part 4
The terminal set used in this problem consisted of a random float, or a random bool (True or False). This set captures all the possible types in the regression problem. The function set in this problem added all the functions sufficient to capture the regression problem:
- Add, subtract, and multiply
- Divide with protection if a zero is used as the denominator.
- Sine only. Cosine is just a phase-shifted sine so a problem with cosine (which this problem doesn't use) can be conveyed with sine.
- Square. I initially tried a power function, but became too complex to handle when negative bases were used.
- Greater than only, the comparator this problem uses. Less than is just the inverse swapping the two options.
- If else, accepting a boolean condition, and two float options, returning either float.

Programs are evaluated by finding the mean squared error between the true regression and the program output. The fitness cases used are in the range of [-5, 15). The graph below shows this range, which clearly captures the shape of the first piecewise function and the second piecewise function as it transforms from the asymptote to the sine. The cases step by 0.2, which is granular enough to capture the steep descent in [0, 1].

![img.png](out/part4/img.png)

The population size of each generation is 1000, with 5% = 50 of the most elite individuals being carried over. Each individual begins 2-6 nodes deep, with mutated subtrees being 2-6 nodes deep, limited to a maximum depth of 15. The probability of crossing over material at the reproductive stage is 75%, and 25% for mutation. 200 generations are evaluated.

| Seed | Best fitness (MSE, minimised) | Corresponding node length |
| --- | --- |  --- | 
| 0 (red) | 0.31498 | 429 |
| 1 (orange) | 0.38310 | 463 | 
| 2 (green) | 0.05091 | 407 | 
| Mean | 0.24966 | 433 |
| SD | 0.14326 | 23.036 |

![img2.png](out/part4/img2.png)

With 200 generations, the individual with seed 2 (green line) fit the true blue regression the best, and had the lowest error as a result. The other two individuals fit the regression well, and had low errors absolutely, but not as well as the best one did in the region -3 < x < 2. All individuals seemed to use irregular piecewise linear functions to approximate the strange region -1 < x < 1, but the best individual actually fit the parabolic curve. 

The best individual is examinable at a high level. The top level node is an if else decision. If the condition is true, the output is sin(x) when simplified (and under x < 91). This is the major component of the function at x>0, as 1/x becomes small as x increases. Otherwise, several terms are added together, as is the case for the true function at x <=0. 

It is interesting that the best individual has the least nodes out of all of them. This result demonstrates that a more complex model does not lead to better results.

# Part 5
$\phi$_1 and $\phi$_2 are set to the same value empirically found by Eberhart and Shi, $\phi = 1.49618$. Likewise, w was set to 0.7298 from the same source. The population size is set to produce 1000 individuals each generation.

The fitness function follows the two given objective functions, the Rosenbrock and Griewank functions. Particles are encoded as a position vector (list of floats) to denote where they are in the search space, and stores its previous velocity to enable acceleration/inertia. The topology type is fully connected, as only one global best is stored and shared between all particles. One experiment stops if a good enough fitness is reached (20) or a maximum number of generations has been reached.

## Griewank function
| D | Mean | SD |
| --- | --- | --- |
| 20 | 1.5992 | 0.0861 |
| 50 | 3.2374 | 0.1904 |

## Rosenbrock function
| D | Mean | SD |
| --- | --- | --- |
| 20 | 2132.82 | 10315.49 |