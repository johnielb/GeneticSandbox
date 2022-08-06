---
title: AIML426 Project 1

author: Johniel Bocacao, 300490028

date: 25 July 2022

---

# Part 1
> Describe the details of your designed GA (including the overall outline, representation, fitness function, crossover and mutation, selection, and parameters). You should also show the results (mean and standard deviation, and convergence curves), and make discussions and conclusions in your report.

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