import sys
import gzip
import numpy as np
import pandas as pd
import difflib as dl

# movie titles
possible_titles = []

# dataframes containing weather stations and cities
df_stations = pd.DataFrame()
df_cities = pd.DataFrame()

# output file name
out_svg = ''


def read_data():
    """ Return the data from text and csv files """
    if len(sys.argv) >= 4:
        # read output file name
        global out_svg
        out_svg = sys.argv[3]
        # read gzip file
        station_fh = gzip.open(sys.argv[1], 'rt', encoding='utf-8')
        stations = pd.read_json(station_fh, lines=True)
        # read csv file
        cities = pd.read_csv(sys.argv[2])
        return stations, cities
    else:
        print("There must be three arguments: 2 input file names (gzip and csv) and 1 output file name (svg)")
        return None, None


def main():
    """ Main function """
    stations, cities = read_data()

    if stations is None or cities is None:
        print('Unable to read input files')
        return -1
    df_cities = cities
    df_stations = stations

if __name__ == '__main__':
    main()
