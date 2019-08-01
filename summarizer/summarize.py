from tf_idf import *


def create_tfidf(docs):
    """

    :param docs: List[String]
        List of the documents
    :return: Tuple(List[floats], List[String])
        List of tf_idf that correspond to List of vocabulary
    """

    stops = nltk.corpus.stopwords.words('english')
    word_counts = [to_counter(doc) for doc in docs]
    vocab = to_vocab(word_counts, stop_words=stops)
    tfs = np.vstack([to_tf(counter, vocab) for counter in word_counts])
    print(tfs)
    idf = to_idf(vocab, word_counts)
    tf_idfs = tfs * idf
    return tf_idfs, vocab


def summarize_mult(docs, tf_idf, vocab):
    """

    :param docs: List[String]
        List of the documents
    :param tf_idf: List[floats] of shape (M,N)
        List of the tf_idf where M corresponds to the number of documents
        and N corresponds to the number of vocab
    :param vocab: List[String] of shape N
    :return: List[M], which is the most common sentence of the document
    """
    # Splits the documents into sentences
    split_docs = []
    for doc in docs:
        split_docs.append(" ".join(doc.split("\n")).split(". "))

    summ_docs = []  # Will store the most defining sentence
    for doc in split_docs:
        doc_stats = []
        for sentence in doc:
            sentence = sentence.split(" ")
            sentence_tfidf = 0
            for word in sentence:
                if word in vocab:
                    word_stat = tf_idf[split_docs.index(doc), vocab.index(word)]
                    sentence_tfidf += word_stat/len(sentence)
            doc_stats.append(sentence_tfidf)
        print(doc_stats)
        summ_docs.append(doc[doc_stats.index(max(doc_stats))])
    return summ_docs

def summarize_one(doc, tf_idf, vocab):
    """

    :param doc: String
        Document as a string.
    :param tf_idf: List[floats] of shape (M,N)
        List of the tf_idf where M corresponds to the number of sentences
        and N corresponds to the number of vocab
    :param vocab: List[String] of shape N
    :return: List[M], which is the most common sentence of the document
    """
    # Splits the document into sentences
    doc = " ".join(doc.split("\n")).split(". ")
    doc_stat = [] # Tracks total tf_idf per sentence
    for sentence in doc:
        doc_stat.append(0.0)
        for word in sentence.split(" "):
            if word in vocab:
                doc_stat[doc.index(sentence)] += tf_idf[doc.index(sentence), vocab.index(word)]/len(sentence.split(" "))
        print(doc_stat)
    return doc[doc_stat.index(max(doc_stat))]


with open("six_abstracts.txt", 'r') as f:
    abstracts = f.read().split("&")
doc = abstracts[0]
tf_idfs, vocab = create_tfidf(doc.split(" "))
print(len(tf_idfs), len(vocab))
print(summarize_one(doc, tf_idfs, vocab))
# summ = summarize_mult(abstracts, *create_tfidf(abstracts))
# print(summ)
