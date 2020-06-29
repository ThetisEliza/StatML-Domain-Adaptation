import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from tensorflow import keras as keras
from enum import Enum
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def fitnevalue(method_str, model_str, *folds, **kwargs):
    results = []
    for source_trains, target_train, dev, test in folds:
        # print(method_str)
        method = eval(method_str)()
        mse = method.fitnevalue(source_trains, target_train, dev, model_str, **kwargs)
        msePredict = method.predict(test)
        print(msePredict)
        results.append(mse)
    return results


def plot_history(*historys):

    colors = 'brg'

    for history, color in list(zip(historys, list(colors))):
        history_dict = history.history

        mse = history_dict['mse']
        val_mse = history_dict['val_mse']

        epochs = range(1, len(mse) + 1)

        # “bo”代表 "蓝点"
        plt.plot(epochs, mse, color+'o', label='Training loss')
        # b代表“蓝色实线”
        plt.plot(epochs, val_mse, color, label='Validation loss')

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


class MethodEnum:
    SRC_ONLY = 'SrcOnly'
    TGT_ONLY = 'TgtOnly'
    ALL = 'All'
    WEIGHTED = 'Weighted'
    PRED = 'Pred'
    FEATURE_AUGMENTATION = 'FeatureAugmentation'
    Methods = [SRC_ONLY, TGT_ONLY, ALL, WEIGHTED, PRED, FEATURE_AUGMENTATION]


class NeuralNetwork():

    def fit(self, train_X, train_Y, dev_X, dev_Y):
        shape = train_X[0].shape
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=shape),
            # layers.Dense(128, activation='relu', ),
            # layers.Dense(128, activation='relu', ),
            # layers.Dense(4, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000001),),
            # layers.Dense(2, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000001),),
            # layers.Dropout(0.5),
            layers.Dense(1, )
        ])

        optimizer = keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])

        history = model.fit(
            train_X, train_Y,
            batch_size=256,
            epochs=80,
            validation_data=(dev_X, dev_Y),
            verbose=1,
        )
        result = model.evaluate(dev_X, dev_Y)

        shape = train_X[0].shape
        model = keras.Sequential([
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1), input_shape=shape),
            # layers.Dense(256, activation='relu', ),
            layers.Dropout(0.5),
            # layers.Dense(512, activation='relu', ),
            # layers.Dense(4, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000001),),
            # layers.Dense(2, activation='relu', kernel_regularizer=keras.regularizers.l2(0.000001),),
            # layers.Dropout(0.5),
            layers.Dense(1, )
        ])

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])

        history2 = model.fit(
            train_X, train_Y,
            batch_size=512,
            epochs=80,
            validation_data=(dev_X, dev_Y),
            verbose=1,
        )
        result = model.evaluate(dev_X, dev_Y)

        plot_history(history, history2)

        print(result)

        self.model = model
        return model.evaluate(dev_X, dev_Y)[1]

    def predict(self, testX):

        testY = self.model.predict(testX)[:,0]
        return testY

class Method:
    def __init__(self):
        self.name = 'Abstract Method'
        self.model = None

    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        self.model = eval(model_str)(**kwargs)
        return None

    def predict(self, test):
        print(test.shape)
        test_X = test[:, :-1]
        test_Y = test[:, -1]
        pred_Y = self.model.predict(test_X)
        test_mse = mean_squared_error(test_Y, pred_Y)
        return test_mse


class SrcOnly(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        train = np.vstack(source_trains)
        # print(train[:1])
        train_X = train[:, :-1]
        train_Y = train[:, -1]

        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]

        if model_str == 'NeuralNetwork':
            dev_mse = self.model.fit(train_X, train_Y, dev_X, dev_Y)

        else:
            self.model.fit(train_X, train_Y)
            pred_Y = self.model.predict(dev_X)
            dev_mse = mean_squared_error(dev_Y, pred_Y)
        return dev_mse


class TgtOnly(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        train = target_train

        train_X = train[:, :-1]
        train_Y = train[:, -1]
        self.model.fit(train_X, train_Y)

        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        if model_str == 'NeuralNetwork':
            dev_mse = self.model.fit(train_X, train_Y, dev_X, dev_Y)

        else:
            self.model.fit(train_X, train_Y)
            pred_Y = self.model.predict(dev_X)
            dev_mse = mean_squared_error(dev_Y, pred_Y)
        return dev_mse


class All(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        source_train = np.vstack(source_trains)
        train = np.vstack([target_train, source_train])
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]

        # nmlzr = Normalizer()
        # train_X = nmlzr.fit_transform(train_X)
        # dev_X = nmlzr.transform(dev_X)

        if model_str == 'NeuralNetwork':
            dev_mse = self.model.fit(train_X, train_Y, dev_X, dev_Y)
        else:
            self.model.fit(train_X, train_Y)
            pred_Y = self.model.predict(dev_X)
            dev_mse = mean_squared_error(dev_Y, pred_Y)

        return dev_mse


class Weighted(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        source_train = np.vstack(source_trains)
        ratio = round(len(source_train) / len(target_train))
        train = source_train
        for _ in range(ratio):
            train = np.vstack([train, target_train])
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        self.model.fit(train_X, train_Y)

        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        if model_str == 'NeuralNetwork':
            dev_mse = self.model.fit(train_X, train_Y, dev_X, dev_Y)

        else:
            self.model.fit(train_X, train_Y)
            pred_Y = self.model.predict(dev_X)
            dev_mse = mean_squared_error(dev_Y, pred_Y)
        return dev_mse


class Pred(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        source_train = np.vstack(source_trains)
        source_train_X = source_train[:, :-1]
        source_train_Y = source_train[:, -1]
        self.ax_model = eval(model_str)(**kwargs)
        self.ax_model.fit(source_train_X, source_train_Y)
        target_train_X = target_train[:, :-1]
        target_train_Y = np.expand_dims(self.ax_model.predict(target_train_X), axis=1)
        train_X = np.hstack([target_train_X, target_train_Y])
        train_Y = target_train[:, -1]
        self.model.fit(train_X, train_Y)

        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        dev_target_feature = np.expand_dims(self.ax_model.predict(dev_X), axis=1)
        dev_X = np.hstack([dev_X, dev_target_feature])
        pred_Y = self.model.predict(dev_X)
        mse = mean_squared_error(dev_Y, pred_Y)
        return mse

    def predict(self, dev):
        dev_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        dev_target_feature = np.expand_dims(self.ax_model.predict(dev_X), axis=1)
        dev_X = np.hstack([dev_X, dev_target_feature])
        pred_Y = self.model.predict(dev_X)
        r2 = r2_score(dev_Y, pred_Y)
        mse = mean_squared_error(dev_Y, pred_Y)
        return mse


class LinInt(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        train = target_train
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        self.target_model = eval(model_str)(**kwargs)
        self.target_model.fit(train_X, train_Y)

        train = np.vstack(source_trains)
        train_X = train[:, :-1]
        train_Y = train[:, -1]
        self.source_model = eval(model_str)(**kwargs)
        self.source_model.fit(train_X, train_Y)



class FeatureAugmentation(Method):
    def fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs):
        Method.fitnevalue(self, source_trains, target_train, dev, model_str, **kwargs)
        domains = 3
        domain_trains = [source_train[:, :-1] for source_train in source_trains]
        # domain_trains = [np.vstack([source_train[:, :-1] for source_train in source_trains])]
        domain_trains.append(target_train[:, :-1])

        domain_trains_Y = [source_train[:, -1] for source_train in source_trains]
        domain_trains_Y.append(target_train[:, -1])
        train_Xs = []
        for i in range(domains):
            train_X = [domain_trains[i]]
            train_X.extend(
                [domain_trains[i] if i == j else np.zeros(shape=domain_trains[i].shape) for j in
                 range(domains)])
            train_X = np.hstack(train_X)
            train_Xs.append(train_X)

        len_of_others = sum([train_X.shape[0] for train_X in train_Xs[:-1]])
        len_of_domains = train_Xs[-1].shape[0]
        # print(len_of_others)
        # print(len_of_domains)
        r = round(len_of_others/len_of_domains)
        # print(r)
        for _ in range(r-1):
            train_Xs.append(train_Xs[-1])
            domain_trains_Y.append(domain_trains_Y[-1])

        train_X = np.vstack(train_Xs)
        # print(train_X[:1])
        train_Y = np.hstack(domain_trains_Y)
        # print(train_Y)

        self.domains = domains

        dev_domain_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        dev_X = [dev_domain_X]
        dev_X.extend(
            [dev_domain_X if j == self.domains - 1 else np.zeros(shape=dev_domain_X.shape) for j in
             range(self.domains)]
        )
        dev_X = np.hstack(dev_X)

        print('compare')
        # print(train_X[-1])
        # print(dev_X[-1])

        # nmlzr = Normalizer()
        # train_X = nmlzr.fit_transform(train_X)
        # dev_X = nmlzr.transform(dev_X)

        # print(train_X[-1])
        # print(dev_X[-1])

        if model_str == 'NeuralNetwork':

            dev_mse = self.model.fit(train_X, train_Y, dev_X, dev_Y)
        else:
            self.model.fit(train_X, train_Y)
            pred_Y = self.model.predict(dev_X)
            dev_mse = mean_squared_error(dev_Y, pred_Y)

        return dev_mse

    def predict(self, dev):
        dev_domain_X = dev[:, :-1]
        dev_Y = dev[:, -1]
        dev_X = [dev_domain_X]
        dev_X.extend(
            [dev_domain_X if j == self.domains - 1 else np.zeros(shape=dev_domain_X.shape) for j in
             range(self.domains)]
        )
        dev_X = np.hstack(dev_X)
        pred_Y = self.model.predict(dev_X)
        r2 = r2_score(dev_Y, pred_Y)
        mse = mean_squared_error(dev_Y, pred_Y)
        return mse
