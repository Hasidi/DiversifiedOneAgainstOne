from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


# from operator import itemgetter
n_cv = 5

def build_pair_classifiers(data_set: pd.DataFrame):
    classifiers_dict = get_classifiers()
    class_labels = list(data_set.iloc[:, -1].unique())
    n_dom = len(class_labels)
    best_pair_classifiers_set = []
    for i in range(n_dom):
        for j in range(i + 1, n_dom):
            pair_data_points = instances_selection(class_labels[i], class_labels[j], data_set)
            pair_classifiers = build_subset_pair_classifiers(pair_data_points, classifiers_dict)
            name, best_classifier = choose_best_classifier(pair_classifiers)
            best_pair_classifiers_set.append(best_classifier)
    return best_pair_classifiers_set


# public
def classify_new_instance(instance, pair_classifiers):
    list_votes = []
    instance = np.array(instance)[:-1].reshape(1, -1)
    for classifier in pair_classifiers:
        class_value = classifier.predict(instance)
        list_votes.append(class_value)
    return max(list_votes, key=list_votes.count)


def get_classifiers():
    """
    chooses the models that will be tested
    :return:
    """
    classifiers = {}
    classifiers['DecisionTree'] = tree.DecisionTreeClassifier()
    classifiers['NaiveBais'] = MultinomialNB()
    return classifiers


'''
choose only the instances which classification is c1 or c2
'''


def instances_selection(c1: str, c2: str, data_set: pd.DataFrame):
    class_column = data_set.columns[-1]
    res = data_set.loc[data_set[class_column].isin([c1, c2])]
    return res


'''
train m models form classifiers_set of size m on dataset data_set
'''


def build_subset_pair_classifiers(data_set : pd.DataFrame, classifiers_dict: dict, eval_function='accuracy'):
    classifiers_model_dict = {}
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    # X = encode_categorical_features(X)  # check if it is by ref
    for name, classifier in classifiers_dict.items():
        trained_classifier = classifier.fit(X, y)
        eval_scores = cross_val_score(classifier, X, y, cv=n_cv)
        classifiers_model_dict[name] = (trained_classifier, np.mean(eval_scores))
    return classifiers_model_dict

'''
choose the best classifer based on accuracy from pair_classifiers which contain m trained models
'''


# ("tree")-> (model, score)
def choose_best_classifier(pair_classifiers: dict):
    sorted_list = sorted(pair_classifiers.items(), key=lambda x: x[1][1], reverse=True)
    name, model_score = list(OrderedDict(sorted_list).items())[0]
    return name, model_score[0]
