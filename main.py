from read_data import get_data_r_ac
from naive_bayes import train_NaiveBayes
from data_util import ParettoDataset
from word2vec_run import train_word2vec
from softmax_regression import train_softmaxreg
from neural_nets import train_nn


data_pd = get_data_r_ac(data_folder='data', get_ac=True, read_cache=True)

dataset = ParettoDataset(data_pd)

# acc ~ 50%
# train_NaiveBayes(dataset)

trained_dataset = train_word2vec(dataset)

# acc ~ 77%
# train_softmaxreg(trained_dataset)

# acc ~ 97%
train_nn(trained_dataset)