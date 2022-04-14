import numpy
from src.durandal.nlp import SupportingPlane


def test_hyperplane_1():
    def f(x):
        return float(sum(x_i ** 2 for x_i in x))

    def grad_f(x):
        return 2.0 * x

    x = numpy.array([1 for _ in range(1)]).reshape(-1, 1)

    sp1 = SupportingPlane(f(x), grad_f(x), x)

    assert sp1.d == 1
    assert numpy.allclose(sp1.c, numpy.array([[2, -1]]))
