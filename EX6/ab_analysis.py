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
    Questions:
    1) Did more users use the search feature? 
                        (More precisely: did a different fraction of users have search count > 0?)
    2) Did users search more often? 
                        (More precisely: is the number of searches per user different?)
    """

    data = load_data()
    odd_users, even_users = group_users(data)
    # users that used new search
    used_A = odd_users[odd_users['search_count'] > 0]
    # users that used old search
    used_B = even_users[even_users['search_count'] > 0]

    # construct the contigency table for question 1:
    #
    #                                       new search(A)         old search(B)
    #  number of users using feature:
    #  total number of users:

    used_A_count = used_A.shape[0]
    used_B_count = used_B.shape[0]

    A_total = odd_users.shape[0]
    B_total = even_users.shape[0]

    contigency_table = [[used_A_count, used_B_count],
                        [A_total , B_total]]
    _, p1, _, _ = stats.chi2_contingency(contigency_table)

    # instructors with odd uid
    instr_odd_users = odd_users[odd_users['is_instructor'] == True]
    # instructors that used new search
    instr_used_A = used_A[used_A['is_instructor'] == True]
    # instructors with even uid
    instr_even_users = even_users[even_users['is_instructor'] == True]
    # instructors that used old search
    instr_used_B = used_B[used_B['is_instructor'] == True]

    instr_used_A_count = instr_used_A.shape[0]
    instr_used_B_count = instr_used_B.shape[0]

    instr_A_total = instr_odd_users.shape[0]
    instr_B_total = instr_even_users.shape[0]

    contigency_table = [[instr_used_A_count, instr_used_B_count],
                        [instr_A_total, instr_B_total]]
    _, p2, _, _ = stats.chi2_contingency(contigency_table)

    # answering question 2:
    p3 = stats.mannwhitneyu(
        even_users['search_count'], odd_users['search_count']).pvalue
    p4 = stats.mannwhitneyu(
        instr_even_users['search_count'], instr_odd_users['search_count']).pvalue

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p1,
        more_searches_p=p3,
        more_instr_p=p2,
        more_instr_searches_p=p4,
    ))


if __name__ == '__main__':
    main()
