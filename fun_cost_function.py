from py_libs import *


def cost_function(J, z, t):
    return norm(J@z - t)**2

def gradient(J, z, t):
    return 2*J.T@(J@z - t)