import sys
import pandas as pd
from scipy import stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def load_data():
    """ Return the search data from the json file """
    if len(sys.argv) >= 2:
        search_data = pd.read_json(sys.argv[1], orient='records', lines=True)
        return search_data
    else:
        print('Unable to read the json file: provide a valid json file name in the same directory!')
        sys.exit()


def group_users(data):
    """ Seperate the search data based for users with odd/even user ids """
    odd_users = data[data['uid'] % 2 == 1]
    even_users = data[data['uid'] % 2 == 0]
    return odd_users, even_users


def main():
    """ The provided searches.json has information about users' usage of the “search” feature, 
    which is where the A/B test happened. Users with an odd-numbered uid were shown a 
    new-and-improved search box. Others were shown the original design.

    Did more users use the search feature? 
                        (More precisely: did a different fraction of users have search count > 0?)
    Did users search more often? 
                        (More precisely: is the number of searches per user different?)
    """

    data = load_data()
    odd_users, even_users = group_users(data)
    # users that used new search
    used_A = odd_users[odd_users['search_count'] > 0]
    # users that used old search
    used_B = even_users[even_users['search_count'] > 0]

    # instructors that used new search
    instr_used_A = used_A[used_A['is_instructor'] == True]
    # instructors that used old search
    instr_used_B = used_B[used_B['is_instructor'] == True]

    # # Output
    # print(OUTPUT_TEMPLATE.format(
    #     more_users_p=stats.chi2_contingency(,
    #     more_searches_p=stats.mannwhitneyu(
    #         used_A['search_count'], used_B['search_count']).pvalue,
    #     more_instr_p=0,
    #     more_instr_searches_p=stats.mannwhitneyu(
    #         instr_used_A['search_count'], instr_used_B['search_count']).pvalue,
    # ))


if __name__ == '__main__':
    main()
