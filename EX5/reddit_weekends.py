import sys
import gzip
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

global raw_counts
global weekday_counts
global weekend_counts

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


def read_data():
    """ Return the data from the gz file """
    if len(sys.argv) >= 2:
        global raw_counts
        counts_fh = gzip.open(sys.argv[1], 'rt', encoding='utf-8')
        raw_counts = pd.read_json(counts_fh, lines=True)
    else:
        print('Unable to read the gz file: provide a valid gz file name in the same directory!')
        sys.exit()


def clean_data():
    """ Clean data: 
            1) 'date' must be in 2012 and 2013
            2) 'subreddit' must be in the /r/canada 
    """
    global raw_counts
    global weekday_counts
    global weekend_counts
    # extract year and add to a new column
    raw_counts['year'] = pd.DatetimeIndex(raw_counts['date']).year
    raw_counts['day_num'] = pd.DatetimeIndex(raw_counts['date']).weekday

    # filter data
    raw_counts = raw_counts[((raw_counts['year'] == 2012) | (raw_counts['year'] == 2013))
                            &
                            (raw_counts['subreddit'] == 'canada')]
    # separate the weekdays from the weekends
    weekday_counts = raw_counts[raw_counts['day_num'] < 5]
    weekend_counts = raw_counts[raw_counts['day_num'] >= 5]


def analyze_data():
    """ 
        1) run Ttest on original weekday comment counts vs. weekend comment counts
        2) check normality for both original weekday and weekend comment counts
        3) check equality of covariances of original weekday and weekend comment counts
    """

    week = weekday_counts['comment_count']
    wknd = weekend_counts['comment_count']
    
    # Fix 1: apply different (np.log, np.exp, np.sqrt, counts**2) transformation on data
    # np.exp results in overflow and counts**2 had poor results
    # between log and sqrt, sqrt had the better general result

    week_sqrt = np.sqrt(week)
    wknd_sqrt = np.sqrt(wknd)

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=stats.ttest_ind(week, wknd).pvalue,
        initial_weekday_normality_p=stats.normaltest(week).pvalue,
        initial_weekend_normality_p=stats.normaltest(wknd).pvalue,
        initial_levene_p=stats.levene(week, wknd).pvalue,
        transformed_weekday_normality_p=stats.normaltest(week_sqrt).pvalue,
        transformed_weekend_normality_p=stats.normaltest(wknd_sqrt).pvalue,
        transformed_levene_p=stats.levene(week_sqrt, wknd_sqrt).pvalue,
        weekly_weekday_normality_p=0,
        weekly_weekend_normality_p=0,
        weekly_levene_p=0,
        weekly_ttest_p=0,
        utest_p=0,
    ))

    # plt.hist(week)
    # plt.title('weekdays')
    # plt.show()
    # plt.hist(wknd)
    # plt.title('weekends')
    # plt.show()

    # answer to the last question: When are more Reddit comments posted in /r/canada, weekdays or weekends?
    print (week.mean(axis=0))
    print (wknd.mean(axis=0))
    

def main():
    """ Read the Reddit comments from gz file and answer the following question:
        Are there a different number of Reddit comments posted on weekdays than on weekends?
    """
    read_data()
    clean_data()
    analyze_data()

if __name__ == '__main__':
    main()
