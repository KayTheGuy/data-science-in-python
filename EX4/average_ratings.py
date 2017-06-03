import sys
import numpy as np
import pandas as pd
import difflib as dl

# movie titles
possible_titles = []

# dataframe containing ratings
df_ratings = pd.DataFrame()

# output file name
out_csv = ''


def read_files():
    """ Return the data from text and csv files """
    if len(sys.argv) >= 4:
        # read output file name
        global out_csv
        out_csv = sys.argv[3]
        # read text file
        valid_titles = pd.read_csv(sys.argv[1], header=None, names=["title"], delimiter = '\n')
        # read csv file
        ratings = pd.read_csv(sys.argv[2])
        return valid_titles, ratings
    else:
        print("There must be three arguments: 2 input file names (txt and csv) and 1 output file name (csv)")
        return None, None


def make_avg_ratings():
    """ Return the average rating for movies """
    def get_title_close_match(title):
        """ Return the closest match for movie title """
        matches = dl.get_close_matches(title, possible_titles)
        if len(matches) > 0:
            # get the best match (the valid title from possible_titles)
            return matches[0]
        else:
            return ''

    get_title_close_match = np.vectorize(
        get_title_close_match)
    global df_ratings
    df_ratings['title'] = get_title_close_match(
        df_ratings['title'])  # get valid titles
    df_ratings = df_ratings[df_ratings['title'] != '']  # remove invalid titles
    df_ratings = df_ratings.sort_values(
        by=['title']).groupby(df_ratings['title']).mean().round(2)


def output_csv():
    """ Write the average rating for movies into a CSV file """
    df_ratings.to_csv(out_csv, sep='\t')


def main():
    """ Main function """
    valid_titles, ratings = read_files()

    if valid_titles is None or ratings is None:
        print('Unable to read input files')
        return -1

    global possible_titles
    global df_ratings

    df_ratings = ratings
    possible_titles = valid_titles['title'].values.tolist()

    make_avg_ratings()  # calculate the average rating for valid titles
    output_csv()        # save the result to a csv file


if __name__ == '__main__':
    main()
