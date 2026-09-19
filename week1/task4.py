import numpy as np

X = np.array([
    [1.0, 2.0],
    [2.0, 0.5],
    [3.0, 1.0],
    [4.0, 3.0],
])

w = np.array([2.0, -1.0])
b = 0.5

def predict(X, w, b):
    return X @ w + b

yhat = predict(X, w, b)
print("yhat shape:", yhat.shape)
print("yhat      :", yhat)

w_bad = w.reshape(2, 1)
yhat_bad = X @ w_bad + b
print("yhat_bad shape:", yhat_bad.shape)
print("yhat_bad      :", yhat_bad)

assert yhat.shape == (4,)
assert yhat_bad.shape == (4, 1)
print("Task 4 OK")