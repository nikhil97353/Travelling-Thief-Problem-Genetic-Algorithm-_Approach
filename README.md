# Genetic-Algorithm-approach-to-Travelling Thief Problem
The GECCO19 Traveling Thief Problem (TTP) combines two NP-hard problems - the Traveling Salesman Problem (TSP) and the Knapsack Problem (KNP). TSP focuses on finding the shortest distance for a traveling salesman to visit each city and return to the original city, while KNP focuses on maximizing the value of the thief's backpack while minimizing the backpack weight below the backpack weight capacity. This project will combine the two problems Optimize in order to obtain an optimal solution.

Key Features:
Dynamic Fitness Calculation: Computes the traveling time based on the thief's speed, which decreases as more items are added to the knapsack.
2-Opt Algorithm: Implements the 2-opt algorithm to improve the route by swapping city pairs for shorter travel time.
Profit Maximization: Ensures optimal profit by selecting the most valuable items without exceeding the knapsack capacity.

Algorithm Overview:
The core of this solution is a Multi-objective Evolutionary Algorithm (MOEA). It includes:

Initial Population Generation: Random generation of individuals (TSP routes and selected items for the knapsack).
Selection Operators: Tournament selection and Roulette Wheel selection to choose parents.
Crossover Operators: Two-point crossover and ordered crossover for TSP and single-point crossover for the knapsack problem.
Mutation Operators: Inversion mutation for TSP routes and flipping mutation for knapsack items.
Pareto Front: A set of non-dominated solutions representing the best trade-offs between objectives.
Crowding Distance: Used for sorting solutions in the Pareto front based on how spread out they are.

## How to Use the Project

### Code Execution:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TTP-Optimization.git
   cd TTP-Optimization
2. Run the provided Python script:
   ```bash
   evolutionary_algorithm.py
 
Conclusion:
The project utilized a Genetic Algorithm to solve the TSP with additional constraints. Several iterations and improvements were made to enhance the algorithm's performance in handling constraints while maximizing profits. The final implementation offers a trade-off between exploration and exploitation for an optimal solution.

References:
Research papers and articles on Genetic Algorithms, Travelling Salesman Problem, and optimization with constraints.
Python documentation for necessary libraries.


