import numpy as np

X = np.array([1.0, 2.0, 3.0])
Y = np.array([2.0, 4.0, 5.0])
N = len(X)

w, b = 0.0, 0.0
lr = 0.1

def predict(w, b, X):
    return w * X + b

def loss(w, b, X, Y):
    yhat = predict(w, b, X)
    return np.mean((yhat - Y) ** 2)

def grad_analytic(w, b, X, Y):
    r = predict(w, b, X) - Y
    dL_dw = (2.0 / N) * np.sum(r * X)
    dL_db = (2.0 / N) * np.sum(r)
    return dL_dw, dL_db

def numeric_gradient(w, b, X, Y, h=1e-5):
    dw = (loss(w + h, b, X, Y) - loss(w - h, b, X, Y)) / (2 * h)
    db = (loss(w, b + h, X, Y) - loss(w, b - h, X, Y)) / (2 * h)
    return dw, db

L0 = loss(w, b, X, Y)
print("residuals :", predict(w, b, X) - Y)
print("initial loss L0 =", L0)

gw, gb = grad_analytic(w, b, X, Y)
nw, nb = numeric_gradient(w, b, X, Y)
print("analytic dL/dw, dL/db =", gw, gb)
print("numeric  dL/dw, dL/db =", nw, nb)

assert abs(gw - nw) < 1e-6
assert abs(gb - nb) < 1e-6

w_new = w - lr * gw
b_new = b - lr * gb
print("w: %.6f -> %.6f" % (w, w_new))
print("b: %.6f -> %.6f" % (b, b_new))

L1 = loss(w_new, b_new, X, Y)
print("new loss L1 =", L1)
print("decrease    =", L0 - L1)

assert L1 < L0
print("Task 6 OK")