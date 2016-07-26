from softmax_regression import prepare_data, prepare_predict_data
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from read_data import get_train_data_r_ac
import os, datetime, glob
import numpy as np
import pandas as pd


def save_model(model):
    now_suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    json_name = 'saved_architecture_' + now_suffix + '.json'
    weight_name = 'saved_nnmodel_weights_' + now_suffix + '.h5'

    json_string = model.to_json()
    open(os.path.join('.', 'cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('.', 'cache', weight_name), overwrite=True)

    # open(os.path.join('.', 'output', json_name), 'w').write(json_string)
    # model.save_weights(os.path.join('.', 'output', weight_name), overwrite=True)


def read_model(json_name, weight_name):
    model = model_from_json(open(os.path.join(json_name)).read())
    model.load_weights(os.path.join(weight_name))
    return model


def train_nn(dataset, predicted_dataset=None):
    # config
    batch_size = 1024
    nb_classes = len(dataset.useful_ac)
    nb_epoch = 500
    read_cache = True

    trainFeatures, trainLabels = prepare_data(sub_dataset=dataset.getTrainData(),dataset=dataset)
    validFeatures, validLabels = prepare_data(sub_dataset=dataset.getValidData(), dataset=dataset)
    testFeatures, testLabels = prepare_data(sub_dataset=dataset.getTestData(), dataset=dataset)

    trainFeatures = trainFeatures.astype('float32')
    validFeatures = validFeatures.astype('float32')
    testFeatures = testFeatures.astype('float32')

    print trainFeatures.shape[0], 'train samples'
    print validFeatures.shape[0], 'valid samples'
    print testFeatures.shape[0], 'test samples'

    trainLabels = np_utils.to_categorical(trainLabels, nb_classes)
    validLabels = np_utils.to_categorical(validLabels, nb_classes)
    testLabels = np_utils.to_categorical(testLabels, nb_classes)

    # train
    model = None
    load_model = False
    jason_name = None
    weight_name = None
    if glob.glob('./cache/saved_architecture_*.json') != [] and \
            glob.glob('./cache/saved_nnmodel_weights_*.h5') != [] and read_cache:
        print '[Status]: Loading nn model...'

        file = glob.glob('./cache/saved_architecture_*.json')
        for fl in file:
            jason_name = fl
            break
        file = glob.glob('./cache/saved_nnmodel_weights_*.h5')
        for fl in file:
            weight_name = fl
            break

        model = read_model(jason_name, weight_name)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        load_model = True
    else:
        model = Sequential()
        model.add(Dense(512, input_shape=(dataset.getWordVec().shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        history = model.fit(trainFeatures, trainLabels,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(validFeatures, validLabels))

    score = model.evaluate(testFeatures, testLabels, verbose=0)

    print 'Test score:', score[0]
    print 'Test accuracy:', score[1]

    if not load_model:
        save_model(model)
    else:
        pass

    if predicted_dataset is not None:
        pred_pd_data = []
        for (linenum, line) in predicted_dataset.iterrows():
            pred_pd_data.append((line['r']))

        predictFeatures = prepare_predict_data(pred_pd_data, dataset)
        predictFeatures = predictFeatures.astype('float32')
        prediction = model.predict(predictFeatures, batch_size=1024)

        uncertain_list = []
        for i in range(len(prediction)):
            max_fir = np.amax(prediction[i])
            max_sec = sorted(prediction[i])[-2]
            if max_fir / max_sec < 1.8:
                uncertain_list.append(i)

        output_prediction = []
        for p in np.argmax(prediction, axis=1):
            output_prediction.append(dataset.useful_ac[p])

        # data_ori = get_train_data_r_ac(data_folder='predict_data', read_cache=False)

        output = pd.DataFrame({
            'r': pred_pd_data,
            'pred_ac': output_prediction
        })

        return output, uncertain_list
    else:
        return None, None