import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def main():
    """ Analyze the data from data.csv
    """

    data = pd.read_csv('data.csv')
    posthoc = pairwise_tukeyhsd(
        data['run_time'], 
        data['sort_type'],
        alpha=0.05
    )
    fig = posthoc.plot_simultaneous()
    fig.set_size_inches((5, 3))
    plt.show(fig)
    print(posthoc)


if __name__ == '__main__':
    main()
