# Durandal
Convex NLP Solver based on a succsessive linear programming. Requires that the constraints form a bounded space.


## How does it work?
To initalize the algorithm any feasible point in the space is sufficent. We generate the supporting hyperplane of the of the objective function at that point then add this as a constraint. In addition we take the value of the objective function generates an upper bound on the optimal solution.
