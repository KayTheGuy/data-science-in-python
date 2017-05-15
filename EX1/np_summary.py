import numpy as np


def main():
    data = np.load('monthdata.npz')
    totals = data['totals']
    counts = data['counts']

    # part A
    # sums over rows of total precipitation
    row_sums = totals.sum(axis=1)

    # index of the minimum element
    minIdx = np.argmin(row_sums)
    print('Row with lowest total preciptiation:')
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

    # part D
    # number of rows
    num_rows = totals.shape[0]
    # reshaped_totals
    reshaped_totals = totals.reshape((num_rows,4,3))
    # Total precipitation for each quarter in each city
    quart_precip_city = reshaped_totals.sum(axis=-1)
    print('Quarterly precipitation totals:')
    print(quart_precip_city.reshape(num_rows,4))

if __name__ == '__main__':
    main()