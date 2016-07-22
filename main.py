from read_data import get_data_r_ac
from train_NaiveBayes import train_NaiveBayes
from data_util import ParettoDataset
from word2vec_run import train_word2vec


data_pd = get_data_r_ac(data_folder='data', get_ac=True, read_cache=True)
dataset = ParettoDataset(data_pd)

train_word2vec(dataset)

# train_NaiveBayes(dataset)

