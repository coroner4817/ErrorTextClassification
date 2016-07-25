import nltk


def document_features(word_features, document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


def train_NaiveBayes(dataset):

    tr_data = dataset.getTrainData()
    va_data = dataset.getValidData()
    tr_data += va_data
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