from tf_idf import *
import textwrap


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
    summ_docs = []  # Will store the most defining sentences
    for doc in split_docs:
        doc_stats = []
        for sentence in doc:
            sent_split = sentence.split(" ")
            sentence_tfidf = 0
            for word in sent_split:
                if word in vocab:
                    word_stat = tf_idf[split_docs.index(doc), vocab.index(word)]
                    sentence_tfidf += word_stat/len(sent_split)
            doc_stats.append(sentence_tfidf)
        # arr = np.array(doc_stats[:-1])
        # print(arr)
        # sort_ind = np.argsort(arr)
        # print(sort_ind)
        # sort_arr = np.array(sorted(arr))
        # print(sort_arr)
        # std = np.std(np.array(arr))
        # print(std)
        # diff = sort_arr[1:] - sort_arr[:-1]
        # print(diff)
        # where = np.where(diff > std)
        # print(where)
        # want_sent = []
        # if len(where[0]) != 0:
        #     ind = sort_ind[where[0][0] + 1]
        #     print(ind)
        #     indices = np.where(arr >= arr[ind])
        #     print(indices)
        #     for i in indices[0]:
        #         want_sent.append(doc[i])
        # else: want_sent.append(doc[sort_ind[-1]])
        summ_docs.append(doc[doc_stats.index(max(doc_stats))])
    return summ_docs


def summarize_one(doc, tf_idf, vocab):
    """

    :param doc: List[String]
        List of the sentences
    :param tf_idf: List[floats] of shape (M,N)
        List of the tf_idf where M corresponds to the number of sentences
        and N corresponds to the number of vocab
    :param vocab: List[String] of shape N
    :return: List[M], which is the most common sentence of the document
    """
    doc_stats = []
    for sentence in doc:
        sent_split = sentence.split(" ")
        sent_split = sent_split[:-1]
        sentence_tfidf = 0
        for word in sent_split:
            if word in vocab:
                word_stat = tf_idf[doc.index(sentence), vocab.index(word)]
                sentence_tfidf += word_stat
        doc_stats.append(sentence_tfidf)
    summ_docs = doc[doc_stats.index(max(doc_stats))]
    return summ_docs


def main():
    with open("six_abstracts.txt", 'r') as f:
        abstracts = f.read().split("&")
    summ = summarize_mult(abstracts, *create_tfidf(abstracts))
    print("Welcome to Summarizer! We summarize your text documents into a comprehensible sentence.")
    while True:
        k = input("What would you like to do?\n1. Read the entirety of one document.\n2. Summarize one document.")
        if k == '1':
            doc = input("Great! We currently have " + str(len(abstracts)) + " documents. Please input the number that you "
                                                                       "would like to read.\n")
            print(textwrap.fill(abstracts[int(doc) - 1], width = 150))
            print()
            again = input("Would you like to go back to the beginning? (Y/N)")
            if again.lower() == 'y': continue
            else: return
        if k == '2':
            doc = input("Which document would you like to summarize? We currently have  " + str(len(abstracts)) +
                        " documents. Please input the number that you would like to summarize.")
            string = summ[int(doc) - 1]
            print(textwrap.fill(string, width = 150))
        if k == '3':
            doc = input("Which document would you like to summarize? We currently have  " + str(len(abstracts)) +
                        " documents. Please input the number that you would like to summarize.")
            string = ""
            sent_split = " ".join(abstracts[int(doc) - 1].split("\n")).split(". ")
            sent_split = sent_split[:-1]
            tf_idf,vocab = create_tfidf(sent_split)
            summ_doc = summarize_one(sent_split, tf_idf, vocab)
            print(textwrap.fill(summ_doc, width=150))
        if k == '4':
            doc = input("Which document would you like to summarize? We currently have  " + str(len(abstracts)) +
                        " documents. Please input the number that you would like to summarize.")
            with open(doc, 'r') as f:
                text = f.read()
            sent_split = " ".join(text.split("\n")).split(". ")
            sent_split = sent_split[:-1]
            tf_idf, vocab = create_tfidf(sent_split)
            summ_doc = summarize_one(sent_split, tf_idf, vocab)
            print(textwrap.fill(summ_doc, width=150))

    # doc = abstracts[0]
    # tf_idfs, vocab = create_tfidf(doc.split(" "))
    # print(len(tf_idfs), len(vocab))
    # print(summarize_one(doc, tf_idfs, vocab))
    # # summ = summarize_mult(abstracts, *create_tfidf(abstracts))
    # # print(summ)


if __name__ == "__main__":
    main()
