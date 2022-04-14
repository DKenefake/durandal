from typing import List, Callable

import numpy
import gurobipy as gp
from gurobipy import GRB


def initialize_durandal(A, b) -> numpy.ndarray:
    """
    This function is used to initialize the durandal solve routine. This generates an initial point in the interior of
    the feasible space. This is done via a chebychev ball LP.

    .. math::
        \min_{x,r} -r
    .. math::
        \begin{align*}
        \text{s.t. } Ax + ||A_i||_2r &\leq b\\
        A_{eq} x &= b_{eq}\\
        r &\geq 0
        \end{align*}

    :param A: The LHS of the constraint matrix
    :param b: The RHS of the constraint matrix
    :return: x, a feasible point in the interior of the feasible space
    """

    c = numpy.zeros((A.shape[1] + 1, 1))
    c[A.shape[1]][0] = -1

    const_norm = numpy.linalg.norm(A, axis=1).reshape(-1, 1)

    A_ball = numpy.block([[A, const_norm], [c.T]])

    b_ball = numpy.concatenate((b, numpy.zeros((1, 1))))
    model = gp.Model()

    model.Params.OutputFlag = 0

    x = model.addMVar(numpy.size(c), vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY)

    model.setObjective(c.flatten()@x)
    model.addConstr(A_ball@x <= b_ball.flatten())

    model.optimize()

    return x.X[:-1]


class SupportingPlane:
    """
    Main helper class for the solver utilizes
    """
    c: numpy.ndarray
    d: float

    def __init__(self, f, grad_f, x):
        """
        Generates the supporting hyperplane of the function given a point, x


        :param f: R^N -> R function that is the objective of the objective
        :param grad_f: R^N -> R^N function that evaluates to the gradient of the objective
        :param x: A point to generate the supporting hyperplane of the function
        """
        num_x = numpy.size(grad_f)
        self.d = grad_f.T @ x - f
        self.c = numpy.zeros(num_x + 1)
        self.c[:num_x] = grad_f.flatten()
        self.c[-1] = -1


class NLP:
    ub: float
    lb: float
    best_sol: numpy.ndarray
    planes: List[SupportingPlane]
    f: Callable[[numpy.ndarray], float]
    grad_f: Callable[[numpy.ndarray], numpy.ndarray]
    A: numpy.ndarray
    b: numpy.ndarray

    def __init__(self, f, grad_f, A, b):
        self.f = f
        self.grad_f = grad_f

        # calculate an initial point to start of the calculation
        initial_point = initialize_durandal(A, b).reshape(-1,1)

        self.best_sol = initial_point

        # find f and grad f at this point
        f_init = self.f(initial_point)
        grad_f_init = self.grad_f(initial_point)

        # add the initial plane to the set
        self.planes = [SupportingPlane(f_init, grad_f_init, initial_point), ]

        # set upper and lower bounds
        self.ub = f_init
        self.lb = -float('inf')

        # expand to include the y term
        self.A = numpy.block([A, numpy.zeros((A.shape[0], 1))])
        self.b = b

    def solve(self, max_cuts: int = 10, output: bool = False):
        num_cuts = 0

        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.Method = 1
        num_vars = self.A.shape[-1]

        x = model.addMVar(num_vars, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

        # set objective
        model.setObjective(x[-1], GRB.MINIMIZE)
        model.addConstrs(sp.c.flatten() @ x <= sp.d for sp in self.planes)
        model.addConstr(self.A @ x <= self.b.flatten())
        model.optimize()

        while num_cuts <= max_cuts:

            # add cutting plane

            x_it = x[:-1].X.reshape(-1, 1)

            f_x = self.f(x_it)
            grad_f_x = self.grad_f(x_it)

            sp = SupportingPlane(f_x, grad_f_x, x_it)

            self.planes.append(sp)

            model.addConstr(sp.c.flatten() @ x <= sp.d, name=f'cut_{num_cuts}')
            model.optimize()

            # update tracking data
            lb = x[-1].X
            num_cuts += 1

            self.lb = max(self.lb, lb)

            if f_x <= self.ub:
                self.best_sol = x_it
                self.ub = f_x

            if output:
                print(f'Lower bound {self.lb} & Upper bound {self.ub}')

        return self.best_sol
