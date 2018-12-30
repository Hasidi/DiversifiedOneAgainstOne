print("Welcome to DOAO Program...")

from sklearn import metrics

import DOAO
import DataLoader


def main():
    data_file_path = r"Data\iris_data.csv"
    columns_types = {'A1': 'Numeric',
                     'A2': 'Numeric',
                     'A3': 'Numeric',
                     'Class': 'Object'}

    data_set = DataLoader.load_cleaned_encoded_data(data_file_path, columns_types)
    train_set, test_set = DataLoader.split_train_test(data_set, 0.2)
    pair_classifiers = DOAO.build_pair_classifiers(train_set)

    output_predicted_df = data_set.copy()
    predicted_classification = []
    for index, instance in test_set.iterrows():
        classification = DOAO.classify_new_instance(instance, pair_classifiers)
        predicted_classification.append(classification)
        output_predicted_df.loc[[index], 'DOAO_Predicted_class'] = classification
        # print('Instance.num[{}]: classifies as: [{}]'.format(list(instance), classification[0]))
    acc = metrics.accuracy_score(test_set.iloc[:, -1], predicted_classification)
    print("Accuracy of DOAO model: {}".format(acc))

main()
