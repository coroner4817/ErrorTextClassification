import nltk


# def get_word_features(dataset, word_feature_count):
#     all_tokens = []
#     for (r, ac) in dataset:
#         tokens = nltk.word_tokenize(r)
#         all_tokens += tokens
#
#     fdist = nltk.FreqDist(all_tokens)
#     word_features = []
#     for iWord in fdist.items():
#         if iWord[1] > word_feature_count:
#             word_features.append(iWord[0])
#
#     for (word, tag) in nltk.pos_tag(word_features):
#         if tag == 'DT' or tag == 'IN' or tag == 'TO' or tag == 'MD' or tag == 'CC' or tag == 'CD':
#             word_features.remove(word)
#
#     print '[Info]: Word feature count: ', len(word_features)
#     print '[Info]: Word features are: '
#     for (word, tag) in nltk.pos_tag(word_features):
#         print (word, tag)
#
#     return word_features
#
#
# def get_word_vec(dataset, split, word_feature_count):
#     # random.shuffle(dataset)
#
#     split_idx = int(len(dataset) * (1 - split))
#     tr_data = dataset[:split_idx]
#     tx_data = dataset[split_idx:]
#
#     tr_word_features = get_word_features(tr_data, word_feature_count)
#
#     tr_feature_set = [(document_features(tr_word_features, d), c) for (d,c) in tr_data]
#     tx_feature_set = [(document_features(tr_word_features, d), c) for (d,c) in tx_data]
#
#     print '[Status]: Finish get word vec'
#
#     return tr_feature_set, tx_feature_set


def document_features(word_features, document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


def train_NaiveBayes(dataset):

    tr_data = dataset.getTrainData()
    tx_data = dataset.getTestData()
    tokens = dataset.getTokens()
    tr_word_features = dataset.tokens_list
    tr_feature_set = [(document_features(tr_word_features, d), c) for (d,c) in tr_data]
    tx_feature_set = [(document_features(tr_word_features, d), c) for (d,c) in tx_data]

    print '[Info]: Start training using NaiveBayes...'

    classifier = nltk.NaiveBayesClassifier.train(tr_feature_set)
    accuracy = nltk.classify.accuracy(classifier, tx_feature_set)

    print '[Info]: Test Accuracy: ', accuracy
    return accuracy