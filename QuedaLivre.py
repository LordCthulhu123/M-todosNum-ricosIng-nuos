import numpy as np
import matplotlib.pyplot as plt


v0, lambda_, g, t0 = 0, 0, 9.81, 0 
v = lambda t: v0 * np.exp(lambda_*(t0 - t))
