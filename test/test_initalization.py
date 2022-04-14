import numpy

from src.durandal.nlp import initialize_durandal

def test_init_1():
    A = numpy.block([[numpy.eye(10)], [-numpy.eye(10)]])
    b = numpy.ones((20, 1)).reshape(-1, 1)

    point = initialize_durandal(A, b)
    assert numpy.allclose(point, numpy.zeros_like(point))
