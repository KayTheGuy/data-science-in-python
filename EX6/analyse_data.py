import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    """ Analyze the data from data.csv
    """

    data = pd.read_csv('data.csv')
    posthoc = pairwise_tukeyhsd(
        data['run_time'], data['sort_type'],
        alpha=0.05
    )
    print(posthoc)


if __name__ == '__main__':
    main()
