import sys
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


def read_file():
    """ Return the data from csv file """
    # get csv file name
    if len(sys.argv) >= 2:
        data = pd.read_csv(sys.argv[1], parse_dates=[2])
        return data
    else:
        return None


def plot(x, y1, y2, y3):
    """ Plot raw data vs. smoothed data"""
    plt.figure(figsize=(12, 4))
    plt.plot(x, y1, 'b.', alpha=0.5)
    plt.plot(x, y2, 'r-', alpha=0.5)
    plt.plot(x, y3, 'g-', alpha=0.5)
    plt.legend(['RAW', 'LOESS', 'KALMAN'])
    # plt.show()  # easier for testing
    plt.savefig('cpu.svg') # for final submission


def main():
    """ Main function """
    cpu_data = read_file()

    if cpu_data is None:
        print('Unable to read the csv file')
        return -1

    # apply LOESS smoothing to data
    loess_data = sm.nonparametric.lowess(
        cpu_data['temperature'], cpu_data['timestamp'], frac=0.05)

    # apply Kalman smoothing to data
    # remove timestamp column
    kalman_data = cpu_data[['temperature', 'cpu_percent']]
    # get the first data point
    initial_state = kalman_data.iloc[0]
    observation_covariance = [[4 ** 2, 0],
                              [0, 50 ** 2]]    # 2 degree temperature and 50 % cpu usage deviation
    transition_covariance = [[0.5 ** 2, 0],
                             [0, 50 ** 2]]     # 2 degree temperature and 50 % cpu usage deviation
    transition_matrix = [[1, 0.125], [0, 1]]
    # create Kalman filter
    kf = KalmanFilter(initial_state_mean=initial_state,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance,
                      transition_matrices=transition_matrix
                      )
    kalman_data, _ = kf.smooth(kalman_data)

    # plot raw data vs LOESS and Kalman smoothed data
    x = cpu_data['timestamp']
    plot(x, cpu_data['temperature'], loess_data[:, 1], kalman_data[:, 0])


if __name__ == '__main__':
    main()
