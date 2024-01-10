import numpy as np
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred / y_true - 1)) * 100

def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

def rmse(y_true, y_pred):
    return mse(y_true, y_pred)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def evaluate(y_true, y_pred):
    print('count:', len(y_true))
    print('mape:', mape(y_true, y_pred))
    print('rmse:', rmse(y_true, y_pred))
    print('mae:', mae(y_true, y_pred))

def evaluate_by_range(y_true, y_pred):
    ranges = [(0, np.inf), (0, 15), (15, 30), (30, 45), (45, np.Inf)]

    for r in ranges:
        mask = (y_true >= r[0]) & (y_true < r[1])
        y_true_range = y_true[mask]
        y_pred_range = y_pred[mask]

        print(f'Range {r}:')
        evaluate(y_true_range, y_pred_range)
        print('-------------------')

def plot(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.title('VIX Futures Price')
    plt.xlabel('true')
    plt.ylabel('pred')
    plt.show()