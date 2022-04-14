# Durandal
Durandal is a convex NLP solver based on successive linear programming. Uses the principle of supporting hyperplanes of convex functions to solve the problem. This cutting procedure is where we get the solver's name from, the legendary sword Durandal.  

Currently, Durandal is very experimental and is a personal project.

## Features

- [x] - Solve NLPs via a series of converging LPs
- [ ] - Dynamic removal of cuts as they become unneeded
- [ ] - A more generic LP interface for other LP solvers

## Requriments (Restrictions)

* Only affine constraints are supported
* Feasible space must be bounded
* The objective function must be bounded above inside the feasible space
* The objective function must be convex (nonconvex are allowed, but no convergence guarantees are given)

## How does it work?

This algorithm forms a converging sequence of upper and lower bounds on the objective function by introducing cuts in the form of supporting hyperplanes of the objective function. Additionally, the candidate solution is always feasible so that it can be interrupted at any point in the method. The core kernel of this routine is the generation of supporting hyperplanes and re-optimizing the central LP problem. This re-optimization warm started as the dual of the central LP is feasible after introducing the cut.


This procedure can basically be viewed as making approximations over the epigraph of the objective function over the feasible region and refining the approximation per iteration.

## Should I use it?

### Is it free?

Yes

### Does it work?

Yes
