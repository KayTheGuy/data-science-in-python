import sys
import pandas as pd


def read_files():
    """ Return the data from text and csv files """
    if len(sys.argv) >= 2:
        titles = pd.read_csv(sys.argv[1], header=None, names=["title"])
        ratings = pd.read_csv(sys.argv[2])
        return titles, ratings
    else:
        return None, None


def main():
    """ Main function """
    titles, ratings = read_files()

    if titles is None:
        print('Unable to read the text file')
        return -1
    if ratings is None:
        print('Unable to read the csv file')
        return -1
    
    print (titles)


if __name__ == '__main__':
    main()
