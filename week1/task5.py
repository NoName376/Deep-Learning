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

yhat     = predict(X, w, b) 
w_bad    = w.reshape(2, 1)
yhat_bad = X @ w_bad + b 

y = np.array([1.0, 4.0, 5.0, 6.0])

def mse_loop(yhat, y):
    total = 0.0
    N = len(y)
    for i in range(N):
        r = yhat[i] - y[i]
        total += r * r
    return total / N

def mse_vec(yhat, y):
    return np.mean((yhat - y) ** 2)

print("residuals:", yhat - y)

L_loop = mse_loop(yhat, y)
L_vec  = mse_vec(yhat, y)
print("MSE loop       :", L_loop)
print("MSE vectorised :", L_vec)

assert abs(L_loop - L_vec) < 1e-12
assert (yhat - y).shape == (4,)

L_bad = mse_vec(yhat_bad, y)
print("(yhat_bad - y).shape:", (yhat_bad - y).shape)
print("MSE with (4,1)      :", L_bad)
print("ratio wrong/right   :", L_bad / L_vec)
print(yhat_bad - y)

print("fixed with ravel():", mse_vec(yhat_bad.ravel(), y))
print("Task 5 OK")