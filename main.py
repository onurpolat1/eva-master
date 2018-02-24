# Convert 3.6
from config.configuration import Config as config
from eva.data import Data
from eva.mllib.ML import *

import sklearn.metrics as metric
import pandas as pd


def main():
    # get config and set defaults

    # data = data.loadfrom_excel("./data/data_no_shift.xlsx")

    data = pd.DataFrame(Data(config).loadfrom_excel("./data/data_no_shift.xlsx"))



    input = data[config.input]

    import numpy as np
    from sklearn.decomposition import PCA
    X = input
    pca = PCA(n_components=6)
    pca.fit(X)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    input = pca.transform(X)



    output = data[config.output]

    # linear Regression
    model = Regressor(config).linear(input, output)
    write_score(model, input, output)

    # "Rigde Regression"
    print("")
    model = Regressor(config).ridge(input, output)
    write_score(model, input, output)

    """
    # "Logistic Regression"
    print("")
    model_log = Regressor(config).logistic(input, output)
    write_score_nn(model_log,input,output)
     """

    # SVM Regression
    print("")
    # input_test, input_train, output_test, output_train = get_data_partitions(data)
    model = Regressor(config).svm(input, output)
    write_score_nn(model, input, output)

    # Decision Tree Regression
    print("")
    model = Regressor(config).decision_tree(input, output)
    write_score_nn(model, input, output)

    # Random Forest Regression
    print("")
    model = Regressor(config).random_forest(input, output)
    write_score_nn(model, input, output)

    # K Nearest Neigbor
    print("")
    model1 = Regressor(config).k_nearest_neigbor(input, output)
    write_score_nn(model1, input, output)

    """ Naive-Bayes Gaussian
    print("")
    model1 = Regressor(config).naive_bayes(input, output)
    write_score_nn(model1, input, output)
    """

    # Multi Layer Perceptron
    # K Nearest Neigbor
    print("")
    model1 = Regressor(config).multi_layer_perceptron(input, output)
    write_score_nn(model1, input, output)


def get_data_partitions(data):
    size = len(data["CPI"])
    size_train = int(size * 0.7)
    print(size)
    input_train = data[config.input][:size_train]
    input_test = data[config.input][size_train:]
    output_train = data[config.output][:size_train]
    output_test = data[config.output][size_train:]
    return input_test, input_train, output_test, output_train


def write_score(model, input, output):
    print(model.__class__.__name__)
    print("Model coefficients :")
    print(model.coef_)
    print("intercept", model.intercept_)
    print(model.score(input, output))

    print("Absolute Mean error : ", metric.mean_absolute_error(output, model.predict(input)))

    print("R^2 Score : ", metric.r2_score(output, model.predict(input)))


def write_score_nn(model, input, output):
    print(model.__class__.__name__)
    """
    print("Model coefficients :")
    print(model.coefs_)
    print("intercept", model.intercepts_[0])
    """
    print(model.score(input, output))
    print("Absolute Mean error : ", metric.mean_absolute_error(output, model.predict(input)))
    print("R^2 Score : ", metric.r2_score(output, model.predict(input)))

    # print(model.loss_)


if __name__ == "__main__":
    main()
