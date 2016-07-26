import pickle, datetime, glob
from data_util import *
from activating_utils import softmax
from learning_utils import sgd


def getSentenceFeature(tokens, wordVectors, sentence):
    sentVector = np.zeros((wordVectors.shape[1],))

    indices = [tokens[word] for word in sentence]
    sentVector = np.mean(wordVectors[indices, :], axis=0)

    return sentVector


def softmaxRegression(features, labels, weights, regularization = 0.0, nopredictions = False):
    prob = softmax(features.dot(weights))

    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N
    cost += 0.5 * regularization * np.sum(weights ** 2)

    grad = np.array(prob)

    # compute the grad base on fomula
    grad[range(N), labels] -= 1.0
    grad = features.T.dot(grad) / N
    grad += regularization * weights

    if N > 1:
        pred = np.argmax(prob, axis=1)
    else:
        pred = np.argmax(prob)

    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred


def accuracy(y, yhat):
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def softmax_wrapper(features, labels, weights, regularization = 0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights,
        regularization)
    return cost, grad


def save_weights(weights, cost):
    now_suffix = cost + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    with open("./output/saved_weights_"+now_suffix+".npy", "w") as f:
        pickle.dump(weights, f)
    with open("./cache/saved_weights_"+now_suffix+".npy", "w") as f:
        pickle.dump(weights, f)


def prepare_data(sub_dataset, dataset):
    dimVec=dataset.getWordVec().shape[1]
    tokens=dataset.getTokens()
    wordVectors=dataset.getWordVec()
    class_dict=dataset.useful_ac_dict

    nSize = len(sub_dataset)
    features = np.zeros((nSize, dimVec))
    labels = np.zeros((nSize,), dtype=np.int32)

    for i in range(nSize):
        sent, class_name = sub_dataset[i]

        words = [w for w in nltk.word_tokenize(sent) if w in tokens]
        features[i, :] = getSentenceFeature(tokens, wordVectors, words)

        labels[i] = class_dict[class_name]

    return features, labels


def prepare_predict_data(predicted_data, dataset):
    data = []
    for (linenum, line) in predicted_data.iterrows():
        data.append((line['r']))

    dimVec=dataset.getWordVec().shape[1]
    tokens=dataset.getTokens()
    wordVectors=dataset.getWordVec()

    nSize = len(data)
    features = np.zeros((nSize, dimVec))

    for i in range(nSize):
        sent = data[i]
        words = [w for w in nltk.word_tokenize(sent) if w in tokens]
        features[i, :] = getSentenceFeature(tokens, wordVectors, words)

    return features


def train_softmaxreg(dataset):
    # config
    regularization = 0.0
    step = 3.0
    mu = 0.9
    update = 'sgd'
    read_cache = True

    dimVectors = dataset.getWordVec().shape[1]

    trainFeatures, trainLabels = prepare_data(sub_dataset=dataset.getTrainData(), dataset=dataset)
    validFeatures, validLabels = prepare_data(sub_dataset=dataset.getValidData(), dataset=dataset)
    testFeatures, testLabels = prepare_data(sub_dataset=dataset.getTestData(), dataset=dataset)

    weights = None
    load = False
    cost = None
    if glob.glob('./cache/saved_weights_*.npy') != [] and read_cache:
        print '[Status]: Loading weights...'
        file = glob.glob('./cache/saved_weights_*.npy')
        for fl in file:
            with open (fl, 'rb') as handle:
                    weights = pickle.load(handle)
            break
        load = True
    else:
        weights = np.random.randn(dimVectors, len(dataset.useful_ac)+1)
        weights, cost, _, _ = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels,
            weights, regularization), weights, step, mu, update, 20000, None, False, PRINT_EVERY=100)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print "Train accuracy (%%): %f" % trainAccuracy

    # Test on dev set
    _, _, pred = softmaxRegression(validFeatures, validLabels, weights)
    validAccuracy = accuracy(validLabels, pred)
    print "Valid accuracy (%%): %f" % validAccuracy

    _, _, pred = softmaxRegression(testFeatures, testLabels, weights)
    print "Test accuracy (%%): %f" % accuracy(testLabels, pred)

    if not load:
        save_weights(weights, str(cost)[:4])
    else:
        pass

