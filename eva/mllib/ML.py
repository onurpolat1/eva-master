from eva.mllib.sklearn.ML import Regressor_Sklearn
from eva.mllib.sklearn.ML import Classifier_Sklearn


class Regressor:

    def __init__(self, config):
        if config.platform == "sklearn":
            self.platform = Regressor_Sklearn()

    def linear(self, input, output):
        return self.platform.linear(input, output)

    def ridge(self, input, output):
        return self.platform.ridge(input, output)

    def logistic(self, input, output):
        return self.platform.logistic(input, output)

    def svm(self, input, output):
        return self.platform.svm(input, output)

    def decision_tree(self, input, output):
        return self.platform.decision_tree(input, output)

    def random_forest(self, input, output):
        return self.platform.random_forest(input, output)

    def k_nearest_neigbor(self, input, output):
        return self.platform.k_nearest_neigbor(input, output)

    def naive_bayes(self, input, output):
        return self.platform.naive_bayes(input, output)

    def multi_layer_perceptron(self, input, output):
        return self.platform.multi_layer_perceptron(input, output)


class Classifier:

    def __init__(self, config):
        if config.platform == "sklearn":
            self.platform = Classifier_Sklearn()

    def linear(self, input, output):
        return self.platform.linear(input, output)

    def ridge(self, input, output):
        return self.platform.ridge(input, output)

    def multi_layer_perceptron(self, input, output):
        return self.platform.multi_layer_perceptron(input, output)
