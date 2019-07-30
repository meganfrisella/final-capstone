import numpy as np

def one_hot(labels, item_labels):
    ''' Create one-hot encodings for each item in item_labels according to labels

        Parameters
        ----------
        labels : list shape (M,)
            list of distinct labels of which to categorize item_labels

        item_labels : list shape (X,)
            list of item labels with each element corresponding to the label of one item

        Returns
        -------
        numpy.ndarray, shape=(X, M)
            one hot encodings for each item
        '''
    onehot_encoded = list()
    for lbl in item_labels:
        item_encoded = [0 for _ in range(len(labels))]
        item_encoded[labels.index(lbl)] = 1
        onehot_encoded.append(item_encoded)
    return np.array(onehot_encoded)


with open('food_labels_raw.txt', mode="r") as var:
    all_labels = var.read().splitlines()
    distinct_labels = list(set(all_labels))
encodings = one_hot(distinct_labels, all_labels)

print(distinct_labels)
print(encodings.shape)
print(encodings[0])