import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime

class GMM:
    def __init__(self, dt=1/252):
        self.dt = dt

    def mu(self, k):
        return 2**(k / 2) * math.gamma((k + 1) / 2) / math.gamma(1 / 2)

    def calculate_epsilon(self, V, a, b):
        epsilon = np.zeros(len(V))

        for t in range(1, len(V)):
            drift = (a - b * V[t-1]) * self.dt
            epsilon[t] = V[t] - V[t-1] - drift

        return epsilon

    def generate_moments(self, V, a, b, sigma, gamma, mu_y, lambda_):
        epsilon = self.calculate_epsilon(V, a, b)

        f = np.zeros((12, len(V)))
        f[0] = np.roll(epsilon, -1) - mu_y * lambda_ * self.dt
        f[1] = np.roll(epsilon, -1)**2 - sigma**2 * V**(2*gamma) * self.dt - 2 * mu_y**2 * lambda_ * self.dt
        f[2] = np.roll(epsilon, -1)**3 - 6 * mu_y**3 * lambda_ * self.dt
        f[3] = np.pi / 2 * np.abs(np.roll(epsilon, -1) * epsilon) - \
            (np.roll(V, -1) * V)**gamma * sigma**2 * self.dt
        f[4] = np.abs(np.roll(epsilon, -1) * epsilon * np.roll(epsilon, 1))**(4/3) - \
            (np.roll(V, -1) * V * np.roll(V, 1))**(4 * gamma / 3) * sigma**4 * self.dt**2 * self.mu(4 / 3)**3
        f[5] = np.abs(np.roll(epsilon, -1) * epsilon * np.roll(epsilon, 1) * np.roll(epsilon, 2)) - \
            (np.roll(V, -1) * V * np.roll(V, 1) * np.roll(V, 2))**gamma * sigma**4 * (2 / np.pi * self.dt)**2

        for i in range(6, 12):
            f[i] = f[i - 6] * V

        return f[:, 3:-1]

    def objective(self, theta, V):
        a, b, sigma, gamma, mu_y, lambda_ = theta
        f = self.generate_moments(V, a, b, sigma, gamma, mu_y, lambda_)

        W = np.cov(f)
        g = np.mean(f, axis=1)
        J = g.T @ W @ g

        return np.log(J)

    def fit(self,
            time_series,
            initial_guess=None,
            bounds=None):

        result = minimize(self.objective,
                          initial_guess,
                          bounds=bounds,
                          args=(time_series,),
                          method='L-BFGS-B')
        return result.x

    def rolling_fit(self,
                    time_series,
                    initial_guess=None,
                    bounds=None,
                    start_date=None,
                    end_date=None,
                    window=2*365):

        filtered_series = time_series
        if start_date:
            filtered_series = filtered_series.loc[filtered_series.index >= start_date - datetime.timedelta(days=window)]
        if end_date:
            filtered_series = filtered_series.loc[filtered_series.index <= end_date]

        dates = filtered_series.loc[filtered_series.index >= start_date].index

        params_list = []

        for date in dates:
            start = date - datetime.timedelta(days=window)
            end = date
            partial_time_series = time_series.loc[
                (time_series.index >= start) &
                (time_series.index <= end)].values.flatten()
            params = self.fit(partial_time_series, initial_guess, bounds)
            params_list.append(params)

        params_df = pd.DataFrame(params_list,
                                 columns=['a', 'b', 'sigma', 'gamma', 'mu_y', 'lambda'],
                                 index=dates)
        return params_df
