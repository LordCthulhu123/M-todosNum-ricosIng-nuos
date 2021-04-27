import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
import pandas as pd
import matplotlib.pyplot as plt

def NumericalMethod(t_values: ArrayLike, function: Callable[..., float], initial_value=10) -> ArrayLike:
    t_values = list(t_values)
    y = []
    y.append(initial_value)
    for i in range(0, len(t_values) - 1):
        out = y[i] + function(t_values[i], y[i]) * (t_values[i + 1] - t_values[i])
        y.append(out)
    return np.array(y)

v0, lambda_, g, t0 = 10, 2, 9.81, 0 
v = lambda t: v0 * np.exp(lambda_*(t0 - t)) + (g/lambda_) * (np.exp(lambda_*(t0 - t)) - 1)
func = lambda t, y: -g - lambda_ * y

fig, axs = plt.subplots(1)
data = {'Analytical': v(np.linspace(0, 6, 1000))}
axs.plot(np.linspace(0, 6, 1000), data['Analytical'], label="Solução analítica")

for n in [10, 100, 1000]:
    this_t = np.linspace(0, 6, n)
    data[str(n)] = NumericalMethod(this_t, func)
    axs.plot(this_t, data[str(n)], label="n = {}".format(str(n)))

axs.set_ylim(-10, 15)
axs.legend(loc="upper right")
plt.show()
print(pd.DataFrame(data))

