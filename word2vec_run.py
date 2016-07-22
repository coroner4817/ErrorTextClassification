import matplotlib.pyplot as plt
from word2vec import *
from learning_utils import *
import datetime
import os


def train_word2vec(dataset):
    random.seed(datetime.datetime.now())

    now_suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    tokens = dataset.getTokens()
    nWords = dataset.nTokens

    # dimVectors = len(dataset.useful_ac) if len(dataset.useful_ac) > 10 else 10
    dimVectors = 15
    C = 5
    step = 0.08
    iterations = 60000
    CostAndGradient = negSamplingCostAndGradient

    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
        dimVectors, np.zeros((nWords, dimVectors))), axis=0)

    wordVectors0, cost, steps = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, CostAndGradient),
        wordVectors, step, iterations, now_suffix, None, True, PRINT_EVERY=100)

    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

    class_dir = os.path.join('.', 'output', 'class_vec_plot_' + now_suffix + '.png')
    word_dir = os.path.join('.', 'output', 'word_vec_plot_' + now_suffix + '.png')

    class_name, class_vec = dataset.getClassVec(wordVectors)
    visualize(class_name, class_vec, class_dir)

    visualizeWords = dataset.getFreqTokens()
    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    visualize(visualizeWords, visualizeVecs, word_dir)

    train_info = ['date_time='+str(now_suffix)+'\n',
                  'cost_and_gradient='+str(CostAndGradient)+'\n',
                  'dimVectors='+str(dimVectors)+'\n',
                  'context='+str(C)+'\n',
                  'sgd_step='+str(steps)+'\n',
                  'iterations='+str(iterations)+'\n',
                  'class_min_count='+str(dataset.class_min_count)+'\n',
                  'token_min_count='+str(dataset.token_min_count)+'\n',
                  'class_number='+str(len(class_name))+'\n',
                  'total_tokens='+str(nWords)+'\n',
                  'freq_tokens_number='+str(len(visualizeWords))+'\n',
                  'table_size='+str(dataset.tablesize)+'\n',
                  'cost='+str(cost)+'\n']

    txt_info = os.path.join('.', 'output', str(now_suffix)+'.txt')

    with open(txt_info, 'w') as f:
        f.writelines(train_info)
    f.close()

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

