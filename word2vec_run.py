import matplotlib.pyplot as plt
from word2vec import *
from learning_utils import *


def train_word2vec(dataset):
    tokens = dataset.getTokens()
    nWords = dataset.nTokens

    dimVectors = 10
    C = 5

    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
        dimVectors, np.zeros((nWords, dimVectors))), axis=0)

    wordVectors0 = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient),
        wordVectors, 0.03, 60000, None, True, PRINT_EVERY=10)

    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])


    class_name, class_vec = dataset.getClassVec(wordVectors)
    visualize(class_name, class_vec, './output/class_vec_plot.png')

    visualizeWords = dataset.getFreqTokens()
    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    visualize(visualizeWords, visualizeVecs, './output/word_vec_plot.png')


def visualize(visualizeWords, visualizeVecs, filename):
    # zero centering
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))

    covariance = 1.0 / len(visualizeWords) * temp.T.dot(temp)

    U,S,V = np.linalg.svd(covariance)

    # after svd only select the first two value as the coordinate
    coord = temp.dot(U[:,0:2])


    for i in xrange(len(visualizeWords)):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i],
            bbox=dict(facecolor='green', alpha=0.1))

    # set plot coordinate info
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig(filename)
    plt.show()

