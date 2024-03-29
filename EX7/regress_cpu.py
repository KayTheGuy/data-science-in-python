import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression


columns = ['temperature', 'cpu_percent', 'fan_rpm', 'sys_load_1']
training_data_file = sys.argv[1]
testing_data_file = sys.argv[2]


def get_data(filename):
    """
    Read the given CSV file. Return (sysinfo DataFrame, array of X (input) values, array of y (known output) values).
    """
    sysinfo = pd.read_csv(filename, parse_dates=[0])
    # the to-be-predicted temperature
    # fill the last row's temp value with 30 
    sysinfo['next_temp'] = sysinfo['temperature'].shift(-1).fillna(30).astype(int) 
    return sysinfo, sysinfo[columns].values, sysinfo['next_temp'].values


def get_trained_coefficients():
    """
    Create and train a model based on the training_data_file data.

    Return the model, and the list of coefficients for the 'columns' variables in the regression.
    """
    _, X_train, y_train = get_data(training_data_file)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    coefficients = model.coef_
    return model, coefficients


def output_regression(coefficients):
    """
    Print a human-readable summary of the regression results.
    """
    regress = ' + '.join('%.3g*%s' % (coef, col)
                         for col, coef in zip(columns, coefficients))
    print('next_temp = ' + regress)


def plot_errors(model):
    _, X_test, y_test = get_data(testing_data_file)
    plt.hist(model.predict(X_test) - y_test, bins=100)
    plt.savefig('test_errors.png')


def smooth_test(coef):
    sysinfo, X_test, _ = get_data(testing_data_file)
    # feel free to tweak these if you think it helps.
    transition_stddev = 1.5
    observation_stddev = 2.0

    dims = X_test.shape[-1]
    kalman_data = X_test
    initial = X_test[0]
    observation_covariance = np.diag([observation_stddev, 2, 10, 1]) ** 2
    transition_covariance = np.diag([transition_stddev, 80, 100, 10]) ** 2
    transition = np.zeros((4, 4))
    # add additional information about data
    transition[0] = [coef[0], coef[1], coef[2], coef[3]] 

    kf = KalmanFilter(
        initial_state_mean=initial,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition,
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)

    plt.figure(figsize=(12, 4))
    plt.plot(sysinfo['timestamp'], sysinfo['temperature'], 'b.', alpha=0.5)
    plt.plot(sysinfo['timestamp'], kalman_smoothed[:, 0], 'g-')
    plt.savefig('smoothed.png')


def main():
    model, coefficients = get_trained_coefficients()
    output_regression(coefficients)
    plot_errors(model)
    smooth_test(coefficients)


if __name__ == '__main__':
    main()
