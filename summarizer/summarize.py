import tf_idf as tf


def create_tfidf(docs):
    """

    :param docs: List[String]
        List of the documents
    :return: Tuple(List[floats], List[String])
        List of tf_idf that correspond to List of vocabulary
    """

    counter = [tf.to_counter(i) for i in docs]
    stops = nltk.corpus.stopwords.words('english')
    vocab = tf.to_vocab(counter, stop_words=stops)
    tf = tuple(tf.to_tf(to_counter(i), vocab) for i in docs)
    idf = tf.to_idf(vocab, counter)
    tf_idf = tf * idf
    return tf_idf, vocab


def summarize(docs, tf_idf, vocab):
    """

    :param docs: List[String]
        List of the documents
    :param tf_idf: List[floats] of shape (M,N)
        List of the tf_idf where M corresponds to the number of documents
        and N corresponds to the number of vocab
    :param vocab: List[String] of shape N
    :return: List[M], which is the most unique sentence of the document
    """
    # Splits the documents into sentences
    doc_1_split = " ".join(doc_1.split("\n")).split(". ")
    doc_2_split = " ".join(doc_2.split("\n")).split(". ")
    doc_3_split = " ".join(doc_3.split("\n")).split(". ")
    doc_4_split = " ".join(doc_4.split("\n")).split(". ")
    split_docs = [doc_1_split, doc_2_split, doc_3_split, doc_4_split]

    summ_docs = []  # Will store the most defining sentence
    for doc in split_docs:
        doc_stats = []
        for sentence in doc:
            sentence = sentence.split(" ")
            sentence_tfidf = 0
            for word in sentence:
                if word in vocab:
                    word_stat = tf_idf[split_docs.index(doc), vocab.index(word)]
                    sentence_tfidf += word_stat
            doc_stats.append(sentence_tfidf)
    summ_docs.append(doc[doc_stats.index(min(doc_stats))])
    return summ_docs