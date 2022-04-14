import numpy

from src.durandal.nlp import NLP


def test_solve_1():
    def f(x):
        return numpy.exp(x) + x ** 2

    def grad_f(x):
        return numpy.exp(x) + 2 * x

    A = numpy.array([[1], [-1]])
    b = numpy.array([[2], [2]])

    nlp = NLP(f, grad_f, A, b)

    x_sol = nlp.solve(max_cuts=20, output=True)

    print(x_sol)


def test_solve_2():
    def f(x):
        return sum(xi**2 for xi in x)

    def grad_f(x):
        return 2*x

    A = numpy.block([[numpy.eye(10)], [-numpy.eye(10)]])
    b = numpy.ones((20, 1)).reshape(-1, 1)

    nlp = NLP(f, grad_f, A, b)

    x_sol = nlp.solve(max_cuts=40, output=True)

    print(x_sol)

def test_solve_3():

    Q = numpy.array([[100, 0, 0], [0.1, 1, 0.1], [0, .3, 1]])
    c = numpy.array([2, 5, 3]).reshape(-1, 1)

    A = numpy.block([[numpy.eye(3)], [-numpy.eye(3)]])
    b = numpy.ones((6, 1)).reshape(-1, 1)

    def f(x):
        return 0.5 * x.T @ Q @ x + c.T @ x

    def grad_f(x):
        return Q @ x + c

    nlp = NLP(f, grad_f, A, b)

    print(nlp.solve(max_cuts=20, output=True))
    print(nlp.solve(max_cuts=30, output=True))


def test_solve_callback():

    def term_callback(nlp:NLP):
        return abs(nlp.ub - nlp.lb) <= .01

    lower_bounds = []

    def gen_callback(nlp:NLP):
        lower_bounds.append(nlp.lb)

    Q = numpy.array([[100, 0, 0], [0.1, 1, 0.1], [0, .3, 1]])
    c = numpy.array([2, 5, 3]).reshape(-1, 1)

    A = numpy.block([[numpy.eye(3)], [-numpy.eye(3)]])
    b = numpy.ones((6, 1)).reshape(-1, 1)

    def f(x):
        return 0.5 * x.T @ Q @ x + c.T @ x

    def grad_f(x):
        return Q @ x + c

    nlp = NLP(f, grad_f, A, b)

    nlp.solve(max_cuts=20, output=True, gen_callback=gen_callback)
    print(lower_bounds)