import numpy as np

# Variable group stores a list of tuples with each tuple holding the groups of an item
with open('food_groups_raw.txt', mode="r") as var:
    text = var.read().splitlines()
print(text[0])
group = []
for i in text:
    group.append(tuple(i.split(", ")))
print(group[0])


def count_groups(groups, group_titles, junk_serv = 1):
    ''' Count the servings of food groups in the items of groups based on group_titles and accounting for junk food

        Parameters
        ----------
        groups : List[Tuple(group, group...)]
            X items with a Tuple of applicable food groups

        group_titles : List of shape M
            Available food groups

        junk_serv : int
            Amount of servings that a junk food should be given

        Returns
        -------
        numpy.ndarray, shape= M,
            Total number of servings with each column corresponding to the group_title columns
        '''
    total = np.zeros(len(group_titles))
    for tup in groups:
        serving = 2
        if 'junk' in tup:
            serving = junk_serv
        for grp in tup:
            total[group_titles.index(grp)] += serving
    return total


group_titles = ['vegetable', 'fruit', 'grain', 'dairy', 'meat', 'seafood', 'nonmeat protein', 'junk']
guide_serv = [5, 4, 6, 3, 1, .5, 1]