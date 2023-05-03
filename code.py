import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data


def calculate_MSE(x, y, theta):
    m = y.size
    return (1 / m) * np.sum([(theta[0] + theta[1] * x[i] - y[i]) ** 2 for i in range(m)])


data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
x0 = np.ones(x_train.shape)                                     # macierz 1-dynek o rozmiarze x_train
X = np.matrix(np.column_stack((x0, x_train)))                  # złóż jednowymiarowe, aby były kolumnami
Y = np.matrix(y_train).T                                        # .T - transpose

XT = X.T
theta_best = (XT * X).I * XT * Y                                # theta_best[1] = a, theta_best[0] = b
print('closed-form solution - theta best: ', theta_best)

# TODO: calculate error
mse = calculate_MSE(x_test, y_test, theta_best)
print('closed-form error MSE: ', mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# średnia arytm
x_train_mean = np.mean(x_train)
y_train_mean = np.mean(y_train)

# odchylenie standardowe
x_train_std = np.std(x_train)
y_train_std = np.std(y_train)

# standardization
x_train_z = (x_train - x_train_mean) / x_train_std
y_train_z = (y_train - y_train_mean) / y_train_std
x_test_z = (x_test - x_train_mean) / x_train_std
y_test_z = (y_test - y_train_mean) / y_train_std

# TODO: calculate theta using Batch Gradient Descent
learnRate = 0.001
theta_best = np.matrix(np.random.rand(2, 1))

XZ = np.matrix(np.column_stack((x0, x_train_z)))
YZ = np.matrix(y_train_z).T

m_y = y_test_z.size
for _ in range(1000):
    gradientOfCostFunction = (2 / m_y) * XZ.T * (XZ * theta_best - YZ)
    theta_best -= learnRate * gradientOfCostFunction


# TODO: calculate error
mse = calculate_MSE(x_test_z, y_test_z, theta_best)
print('gradient error MSE: ', mse)

# plot the regression line
x = np.linspace(min(x_test_z), max(x_test_z), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test_z, y_test_z)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()