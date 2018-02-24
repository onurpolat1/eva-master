from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from statsmodels.tsa.vector_ar import var_model

class Regressor_Sklearn:

    def __init__(self):
        pass

    def linear(self, input, output):
        model = linear_model.LinearRegression()
        model.fit(input, output)
        return model

    def ridge(self, input, output):
        model = linear_model.Ridge(alpha=10)
        model.fit(input, output)
        return model

    def logistic(self, input, output):
        model = linear_model.LogisticRegression(C=0.01, solver='liblinear')
        model.fit(input, output)
        return model

    def svm(self, input, output):
        model = svm.SVR(C=50, gamma=0.001, epsilon=0.2)
        model.fit(input, output)
        return model

    def decision_tree(self, input, output):
        model = tree.DecisionTreeRegressor(max_features='sqrt', max_depth=5)
        model.fit(input, output)
        return model

    def random_forest(self, input, output):
        model = RandomForestRegressor(200, max_features='sqrt', max_depth=11)
        model.fit(input, output)
        return model

    def k_nearest_neigbor(self, input, output):
        model = KNeighborsRegressor(n_neighbors=2, p=1)
        model.fit(input, output)
        return model

    def naive_bayes(self, input, output):
        model = GaussianNB()
        model.fit(input, output)
        return model

    def var(self, input, output):
        model = var_model.VAR()
        model.fi
        model.fit(input, output)
        return model

    def multi_layer_perceptron(self, input, output):
        model = MLPRegressor(hidden_layer_sizes=(11, 11), alpha=.001, activation='relu', solver='lbfgs')
        model.fit(input, output)
        return model


class Classifier_Sklearn:

    def __init__(self):
        pass

    def linear(self, input, output):
        model = linear_model.LinearRegression()
        model.fit(input, output)
        return model

    def multi_layer_perceptron(self, input, output):
        model = MLPClassifier(hidden_layer_sizes=11, alpha=.001, activation='relu', solver='lbfgs')
        model.fit(input, output)
        return model

    def logistic(self, input, output):
        pass
