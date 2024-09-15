# Genetic-Algorithm-approach-to-Travelling Thief Problem
Brief description
The GECCO19 Traveling Thief Problem (TTP) combines two NP-hard problems - the Traveling Salesman Problem (TSP) and the Knapsack Problem (KNP). TSP focuses on finding the shortest distance for a traveling salesman to visit each city and return to the original city, while KNP focuses on maximizing the value of the thief's backpack while minimizing the backpack weight below the backpack weight capacity. This project will combine the two problems Optimize in order to obtain an optimal solution.
How to install the project
Dependencies: Ensure Python (with required libraries - matplotlib) is installed.

How to use the project
Code Execution:
Run the provided Python script (python your_script.py).
Adjust parameters (population size, mutation rate, etc.) as needed.
Expected Output:
Visualization of city coordinates.
Console output showing each generation's best fitness.
Final best fitness and the corresponding optimal route visualization.
tests
Open main.ipynb
1.	Run block : Import packages and libraries
2.	Go to Main Function and do following changes
change the 'test_name' to the dataset you want to solve
ea = MYTEAM(test_name='a280-n279')
change the 'size p' as the size of population following the the 'LIMI_SOLUTION' in evolutionary_algorithm.py
ea.generate_initial(size_p=100)
LIMIT_SOLUTION = {
    'a280-n279': 100,
    'a280-n1395': 100,
    'a280-n2790': 100,
    'fnl4461-n4460': 50,
    'fnl4461-n22300': 50,
    'fnl4461-n44600': 50,
    'pla33810-n33809': 20,
    'pla33810-n169045': 20,
    'pla33810-n338090': 20,
}
change the 'generations' to how many generations you want to run
ea.optimize(generations=100, tournament_size=30, crossover='OX')

Run block
ea = MYTEAM(test_name='a280-n279')
ea.generate_initial(size_p=100)

5.Run block (The following is a demo only, the output of the code you are running may be different)
ea.optimize(generations=100, tournament_size=30, crossover='OX')
Get Result:
Generation:  1
Generation:  2
Generation:  3
Generation:  4
Generation:  5
Generation:  6
Generation:  7
Generation:  8
Generation:  9
Generation:  10
Generation:  11
Generation:  12
Generation:  13
Generation:  14
Generation:  15
Generation:  16
Generation:  17
Generation:  18
 Generation:  19
 Generation:  20
 Generation:  21
 Generation:  22
 Generation:  23
 Generation:  24
 Generation:  25
 ...
 Generation:  100
6.Run block
ea.show()


 
Conclusion:
The project utilized a Genetic Algorithm to solve the TSP with additional constraints. Several iterations and improvements were made to enhance the algorithm's performance in handling constraints while maximizing profits. The final implementation offers a trade-off between exploration and exploitation for an optimal solution.

References:
Research papers and articles on Genetic Algorithms, Travelling Salesman Problem, and optimization with constraints.
Python documentation for necessary libraries.
Future Directions:
Experimentation with different crossover and mutation techniques.
Utilization of other metaheuristic approaches for comparison.
Performance tuning for larger problem instances or different constraint settings.

