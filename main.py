from read_data import get_train_data_r_ac, get_pred_data_r
from naive_bayes import train_NaiveBayes
from data_util import ParettoDataset
from word2vec_run import train_word2vec
from softmax_regression import train_softmaxreg
from neural_nets import train_nn


data_train = get_train_data_r_ac(data_folder='train_data', read_cache=True)
data_pred = get_pred_data_r(data_folder='predict_data', read_cache=True)

dataset = ParettoDataset(data_train)

# acc ~ 48%
# train_NaiveBayes(dataset)

trained_dataset = train_word2vec(dataset)

# acc ~ 77%
# train_softmaxreg(trained_dataset)

# acc ~ 97%
prediction, uncertain_idx = train_nn(trained_dataset, predicted_dataset=data_pred)

# prediction.to_csv('./output/predict.csv')