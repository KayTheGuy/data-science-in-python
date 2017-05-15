import pandas as pd

def main():
    # load data
    totals = pd.read_csv('totals.csv').set_index(keys=['name'])
    counts = pd.read_csv('counts.csv').set_index(keys=['name'])

    # part A
    # sums over rows of total precipitation
    row_sums = totals.sum(axis=1)
    # index of the minimum element
    minIdx  = row_sums.idxmin()
    print('City with lowest total preciptiation:')
    print(minIdx)

    # part B
    # sums over columns of total precpitation (for each month)
    colum_sums = totals.sum(axis=0)
    # sums over columns of total observation (for each month)
    column_sums_obs = counts.sum(axis=0)
    # average precipitation for each month
    avg_precip_month = colum_sums / column_sums_obs
    print('Average precipitation in each month:')
    print(avg_precip_month)

    # part C
    # sums over rows of total observation (for each city)
    rows_sums_obs = counts.sum(axis=1)
    # average precipitation (daily) for each city
    avg_precip_city = row_sums / rows_sums_obs
    print('Average precipitation in each city:')
    print(avg_precip_city)


if __name__ == '__main__':
    main()