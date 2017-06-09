import sys
import gzip
import pandas as pd

global raw_counts
global clean_counts

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
    # extract year and add to a new column
    raw_counts['year'] = pd.DatetimeIndex(raw_counts['date']).year
    raw_counts['weekday'] = pd.DatetimeIndex(raw_counts['date']).weekday
    global clean_counts

    # filter data
    clean_counts = raw_counts[
                               ((raw_counts['year'] == 2012) | (raw_counts['year'] == 2013))
                                & 
                                (raw_counts['subreddit'] == 'canada')
                             ]


def main():
    """ Read the Reddit comments from gz file and answer the following question:
        Are there a different number of Reddit comments posted on weekdays than on weekends?
    """

    read_data()
    clean_data()
    # print (raw_counts)
    print(clean_counts)

    # print(OUTPUT_TEMPLATE.format(
    #     initial_ttest_p=0,d
    #     initial_weekday_normality_p=0,
    #     initial_weekend_normality_p=0,
    #     initial_levene_p=0,
    #     transformed_weekday_normality_p=0,
    #     transformed_weekend_normality_p=0,
    #     transformed_levene_p=0,
    #     weekly_weekday_normality_p=0,
    #     weekly_weekend_normality_p=0,
    #     weekly_levene_p=0,
    #     weekly_ttest_p=0,
    #     utest_p=0,
    # ))


if __name__ == '__main__':
    main()
