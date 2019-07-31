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
    unique = Counter()
    for c in counters:
        unique.update(c)
    if stop_words is not None:
        for word in stop_words:
            del unique[word]
    if k is not None:
        unique = set(unique.most_common(k))
        return sorted(list(unique))
    return sorted(list(unique))


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
    total = 0.0
    for key in counter:
        if key in vocab:
            total += counter[key]
    tf = []
    for word in vocab:
        if counter[word] is None:
            tf.append(0)
        else:
            tf.append(1.0 * counter[word] / total)
    return np.array(tf)


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
    N = 1.0 * len(counters)
    idf = []
    for i in range(len(vocab)):
        word = vocab[i]
        docs = 0.0
        for countmap in counters:
            if word in countmap:
                docs += 1.0
        idf.append(N / docs)
    return np.array(np.log10(idf))