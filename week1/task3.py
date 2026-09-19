import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(w, b, x, y):
    z = w * x + b
    a = sigmoid(z)
    L = (a - y) ** 2
    return z, a, L

def grad_analytic(w, b, x, y):
    z, a, L = forward(w, b, x, y)
    dL_da = 2.0 * (a - y)
    da_dz = a * (1.0 - a)
    dz_dw = x
    dz_db = 1.0
    dL_dw = dL_da * da_dz * dz_dw
    dL_db = dL_da * da_dz * dz_db
    return dL_dw, dL_db

def numeric_gradient_wb(w, b, x, y, h=1e-5):
    Lw_plus  = forward(w + h, b, x, y)[2]
    Lw_minus = forward(w - h, b, x, y)[2]
    dL_dw = (Lw_plus - Lw_minus) / (2.0 * h)

    Lb_plus  = forward(w, b + h, x, y)[2]
    Lb_minus = forward(w, b - h, x, y)[2]
    dL_db = (Lb_plus - Lb_minus) / (2.0 * h)

    return dL_dw, dL_db

x, y = 2.0, 1.0
w, b = 0.3, -0.1

z, a, L = forward(w, b, x, y)
print("z, a, L =", z, a, L)

dL_dw, dL_db = grad_analytic(w, b, x, y)
print("analytic dL/dw, dL/db =", dL_dw, dL_db)

ndw, ndb = numeric_gradient_wb(w, b, x, y)
print("numeric dL/dw, dL/db =", ndw, ndb)

assert abs(dL_dw - ndw) < 1e-6
assert abs(dL_db - ndb) < 1e-6
print("Task 3 OK")