import time
import numpy as np
import pandas as pd
from scipy import stats
from implementations import all_implementations

def main():
    """ Comparing the performance of the seven sorting implementations given in the all_implementations array: 
        qs1, qs2, qs3, qs4, qs5, merge1, partition_sort.

    """
    data = pd.DataFrame(columns=['sort_type', 'run_time'])

    for i in range(0,40):
        random_array = np.random.randint(1, 1000, dtype='int', size=100)
        for sort in all_implementations:
            st = time.time()
            res = sort(random_array)
            en = time.time()
            data = data.append({'sort_type': sort.__name__, 'run_time': (en-st)}, ignore_index=True)

    
    # save data
    data.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()
