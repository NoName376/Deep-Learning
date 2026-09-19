import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.sin(x)

def df_true(x):
    return np.cos(x)

def deriv_forward(f, x, h):
    return (f(x + h) - f(x)) / h

def deriv_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


x0 = 1.0
print("true :", df_true(x0))
print("forward h=1e-5:", deriv_forward(f, x0, 1e-5))
print("central h=1e-5:", deriv_central(f, x0, 1e-5))
hs = np.logspace(-1, -12, 45)
err_fwd = np.array([
    abs(deriv_forward(f, x0, h) - df_true(x0))
    for h in hs
])


err_ctr = np.array([abs(deriv_central(f, x0, h) - df_true(x0)) for h in hs])


plt.figure(figsize=(6, 4))
plt.loglog(hs, err_fwd, "o-", label="forward")
plt.loglog(hs, err_ctr, "s-", label="central")
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("absolute error")
plt.legend()
plt.title("Finite-Difference Error vs Step Size")
plt.tight_layout()
plt.show()
print("best h, forward:", hs[np.argmin(err_fwd)])
print("best h, central:", hs[np.argmin(err_ctr)])

assert abs(
    deriv_central(f, x0, 1e-5) - df_true(x0)
) < 1e-7

print("Task 1 OK")