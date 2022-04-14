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

    x = model.addMVar(numpy.size(c), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)

    model.setObjective(c.flatten() @ x)
    model.addConstr(A_ball @ x <= b_ball.flatten())

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
    model: gp.Model
    x: gp.MVar

    def __init__(self, f, grad_f, A, b):
        """
        Initializes the NLP routine, finds an initial feasible point and generates an initial supporting hyperplane
        """

        # store the objective function
        self.f = f
        # wrap the grad f function in a lambda to reshape it to a column vector (just in case)
        self.grad_f = lambda v: grad_f(v).reshape(-1, 1)

        # initilize the planes as empty
        self.planes = []

        # find an initial feasible point
        self.init_x_point(A, b)

        # generate the initial LP model
        self.init_lp_model()

    def init_x_point(self, A, b) -> None:

        # calculate an initial point to start of the calculation
        initial_point = initialize_durandal(A, b).reshape(-1, 1)

        self.best_sol = initial_point

        # find f and grad f at this point
        f_init = self.f(initial_point)
        grad_f_init = self.grad_f(initial_point)
        init_plane = SupportingPlane(f_init, grad_f_init, initial_point)
        # add the initial plane to the set
        self.planes.append(init_plane)

        # set upper and lower bounds
        self.ub = f_init
        self.lb = -float('inf')

        # expand to include the y term
        self.A = numpy.block([A, numpy.zeros((A.shape[0], 1))])
        self.b = b

    def init_lp_model(self) -> None:
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.model.Params.Method = 1
        num_vars = self.A.shape[-1]

        self.x = self.model.addMVar(num_vars, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

        # set objective
        self.model.setObjective(self.x[-1], GRB.MINIMIZE)

        sp = self.planes[0]
        self.model.addConstrs(sp.c.flatten() @ self.x <= sp.d for sp in self.planes)
        self.model.addConstr(self.A @ self.x <= self.b.flatten())

    def solve(self, max_cuts: int = 100, output: bool = False, term_callback=None, gen_callback=None):

        if term_callback is None:
            term_callback = nlp_termination_callback

        if gen_callback is None:
            gen_callback = nlp_general_callback

        self.model.optimize()

        while len(self.planes) <= max_cuts:

            # Generate the cutting plane

            x_it = self.x[:-1].X.reshape(-1, 1)
            f_x = self.f(x_it)
            grad_f_x = self.grad_f(x_it)
            sp = SupportingPlane(f_x, grad_f_x, x_it)

            self.add_cut(sp)

            self.model.optimize()

            # add updating for the
            self.update_bounds(x_it, f_x, lb=self.x[-1].X)

            # call the generic call back
            gen_callback(self)
            # call the termination call back
            if term_callback(self):
                break

            if output:
                print(f'Lower bound {self.lb} & Upper bound {self.ub}')

        return self.best_sol

    def update_bounds(self, x_0, f_0, lb):
        # update tracking data

        self.lb = max(self.lb, lb)

        if f_0 <= self.ub:
            self.best_sol = x_0
            self.ub = f_0

    def add_cut(self, sp: SupportingPlane) -> None:
        """
        Adds the supporting hyper plane to the plane set and applied the cut to the model
        """
        # add the supporting hyperplane to the plane list
        self.planes.append(sp)

        # apply the hyperplane to the model
        self.model.addConstr(sp.c.flatten() @ self.x <= sp.d, name=f'user_cut')


def nlp_termination_callback(obj: NLP) -> bool:
    return False


def nlp_general_callback(obj: NLP) -> None:
    return None
