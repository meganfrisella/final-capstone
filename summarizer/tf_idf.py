import nltk
import numpy as np
from collections import Counter
import re, string


def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub('', corpus)


def to_counter(doc):
    """
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.

    Parameters
    ----------
    doc : str

    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    doc = sorted(strip_punc(doc).lower().split())
    return Counter(doc)


def to_vocab(counters, k=None, stop_words=None):
    """
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`

    Parameters
    ----------
    counters : Iterable[Iterable[str]]

    k : Optional[int]
        If specified, only the top-k words are returned

    stop_words : Optional[Collection[str]]
        A collection of words to be ignored when populating the vocabulary
    """
    vocab = Counter()
    for counter in counters:
        vocab.update(counter)

    if stop_words is not None:
        for word in set(stop_words):
            vocab.pop(word, None)  # if word not in bag, return None
    return sorted(i for i, j in vocab.most_common(k))


def to_tf(counter, vocab):
    """
    Parameters
    ----------
    counter : collections.Counter
        The word -> count mapping for a document.
    vocab : Sequence[str]
        Ordered list of words that we care about.

    Returns
    -------
    numpy.ndarray
        The TF descriptor for the document, whose components represent
        the frequency with which each term in the vocab occurs
        in the given document."""
    x = np.array([counter[word] for word in vocab], dtype=float)
    return x / x.sum()


def to_idf(vocab, counters):
    """
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.

    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.

    Returns
    -------
    numpy.ndarray
        An array whose entries correspond to those in `vocab`, storing
        the IDF for each term `t`:
                           log10(N / nt)
        Where `N` is the number of documents, and `nt` is the number of
        documents in which the term `t` occurs.
    """
    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in vocab]
    nt = np.array(nt, dtype=float)
    return np.log10(N / nt)