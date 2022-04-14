# Durandal
Convex NLP Solver based on a successive linear programming. Uses the principle of supporting hyperplanes of convex functions to solve the problem.


## Features

- [x] - Solve NLPs via a series of converging LP
- [ ] - Dynamic removal of cuts as they become uneeded
- [ ] - A more generic LP interface for other LP solvers

# Requriments (Restrictions)

* Only Affine constraints are supported
* Feasible space must be bounded
* The objective function must be bounded above inside the feasible space
* The objective function must be convex (nonconvex are allowed but the solution might be quite bad)

## How does it work?

This algorrithm forms a converging sequence of upper and lower bounds on the objective function by introducing cuts in the form of supporting hyperplanes of the objective function. Additionally, the candidate solution is always feasible, so it can be interupted at at any point in the method. The core kernel of this routine is generation of supporting hyperplanes and reoptimizing the central LP problem. This is effectivley warm started as the dual of the central LP is feasible after the introduction of the cut.


This can basically be viewed as making approximations over the epigraph of the objective function over the feasible region and refining the approximation per iteration.

## Should I use it?

### Is it free?

Yes

### Does it work?

Yes
