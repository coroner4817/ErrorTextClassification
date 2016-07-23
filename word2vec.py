import numpy as np
import random

from activating_utils import softmax, sigmoid, sigmoid_grad


def normalizeRows(x):
    square_t = np.square(x)
    sum_t = np.sum(square_t, axis=1, dtype=float).reshape((x.shape[0], 1))
    sum_t = np.sqrt(sum_t)
    x = x / sum_t

    return x

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])
    delta = probabilities

    # get the gradient of y_hat - y
    delta[target] -= 1

    N = delta.shape[0]
    D = predicted.shape[0]

    # compute the grad as problem a
    grad = delta.reshape((N,1)) * predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

    indices = [target]
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]

    labels = np.array([1] + [-1 for k in xrange(K)])
    vecs = outputVectors[indices,:]

    # compute the grad as problem c
    t = sigmoid(vecs.dot(predicted) * labels)
    cost = -np.sum(np.log(t))

    delta = labels * (t - 1)
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape((1,predicted.shape[0])))

    # indices has multiple same sampled vec
    for k in xrange(K+1):
        grad[indices[k]] += gradtemp[k,:]

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
        cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    D = inputVectors.shape[1]
    predicted = np.zeros((D,))

    indices = [tokens[cwd] for cwd in contextWords]
    for idx in indices:
        predicted += inputVectors[idx, :]

    cost, gp, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors, dataset)

    for idx in indices:
        gradIn[idx, :] += gp

    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]

    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad