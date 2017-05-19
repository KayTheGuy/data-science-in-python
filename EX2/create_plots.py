
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    # get file names
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    # read data from files
    data1 = pd.read_table(filename1, sep=' ', header=None, index_col=1, names=[
        'lang', 'page', 'views', 'bytes'])
    data2 = pd.read_table(filename2, sep=' ', header=None, index_col=1, names=[
        'lang', 'page', 'views', 'bytes'])

    return data1, data2

def plot(data1, joined_data):
    # draw plots
    plt.figure(figsize=(10, 5)) # change the size to something sensible
    plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
    plt.plot(data1['views'].values)
    plt.title('Popularity Distribuation')
    plt.ylabel('Views')
    plt.xlabel('Rank')
    plt.subplot(1, 2, 2) # ... and then select the second
    plt.plot(joined_data['views_data1'].values, joined_data['views_data2'].values, 'b .')
    plt.title('Daily Correlation')
    plt.ylabel('Day 2 Views')
    plt.xlabel('Day 1 Views')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('wikipedia.png') # plt.show()

def main():

    data1, data2 = load_data()

    # sort the data1 and data2 by the number of views (decreasing)   
    data1 = data1.sort_values(['views'], ascending=False)
    data2 = data2.sort_values(['views'], ascending=False)

    # joind data1 and data2 on their index 'page'
    joined_data = data1.join(data2, lsuffix='_data1', rsuffix='_data2')

    plot(data1, joined_data)

if __name__ == '__main__':
    main()
