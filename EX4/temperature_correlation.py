import sys
import gzip
import numpy as np
import pandas as pd
import difflib as dl
import matplotlib.pyplot as plt
from math import cos, asin, sqrt

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
        global df_cities
        global df_stations
        out_svg = sys.argv[3]
        # read gzip file
        station_fh = gzip.open(sys.argv[1], 'rt', encoding='utf-8')
        df_stations = pd.read_json(station_fh, lines=True)
        # read csv file
        df_cities = pd.read_csv(sys.argv[2])
    else:
        print('Unable to read input files:')
        print("\tThere must be three arguments: 2 input file names (gzip and csv) and 1 output file name (svg)")
        sys.exit()


def clean_data():
    """ Clean station and cities data """
    # change 'avg_tmax' to celsius by dividing by 10
    global df_stations
    global df_cities
    df_stations['avg_tmax'] = df_stations['avg_tmax'] / 10

    # dropping NaN values in 'population' or 'area'
    df_cities = df_cities.dropna(how='any')

    # convert 'area' from m^2 to km^2
    df_cities['area'] = df_cities['area'] / 10**6

    # exclude cities with area greater than 10000 km^2
    df_cities = df_cities[df_cities['area'] <= 10000]


def distance(lat1, lon1, lat2, lon2):
        """ Return the distance between two points based on their latitudes and longitudes (Haversine formula)"""
        # taken from:
        # https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
        p = 0.017453292519943295  # Pi/180
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * \
            cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))  # 2*R*asin... (in kilometers)


def get_avg_tmax_from_closest_station(city):
    """  Get 'avg_tmax' from the closest weather station to the city """
    vectorized_distance = np.vectorize(distance, otypes=[np.float])
    distance_list = vectorized_distance(
        city['latitude'], city['longitude'], df_stations['latitude'], df_stations['longitude'])
    # find the index of the closest station to the city
    min_idx = np.argmin(distance_list)
    return df_stations.iloc[min_idx]['avg_tmax']


def plot_data(x, y):
    """ draw plots to an svg file"""
    plt.figure(figsize=(16, 8))  # change the size to something sensible
    plt.subplot(1, 2, 1)  # subplots in 1 row, 2 columns, select the first
    # plt.plot(data1['views'].values)
    plt.plot(x, y, 'b .')
    plt.title('Temperature vs Population Density')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.savefig(out_svg)
    # plt.show()


def main():
    """ Main function """
    read_data()
    clean_data()

    # add 'avg_tmax' for each city to the dataframe
    df_cities['avg_tmax'] = df_cities.apply(
        get_avg_tmax_from_closest_station, axis=1)
    # calculate density and add to the dataframe
    df_cities['density'] = df_cities['population'] / df_cities['area']
    plot_data(df_cities['avg_tmax'], df_cities['density'])


if __name__ == '__main__':
    main()
