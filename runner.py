import csv
from data_handler import DataHandler
from classifier import RuleClassifier
from sklearn.metrics import classification_report


def write_output(res, set_name):
    """
    Writes to csv the is and the predicted status
    :param res: RuleClassifier results (list of tuples)
    :param set_name: string (train or test)
    """
    with open(f'results_{set_name}.csv', 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id', 'status'])
        for row in res:
            csv_out.writerow(row)


def run_classifier(split_dataset):
    """
    Performs classification and outputs metrics report
    :param split_dataset: dataframe (train or test)
    :return: list of tuples (id, status)
    """
    true_labels = split_dataset['status'].tolist()
    predicted_labels = []
    rc = RuleClassifier(split_dataset, kb_path='kb.json')

    for index, row in rc.data.iterrows():
        rc.classify(row['text'], row['row_id'])

    for result in rc.res:

        predicted_labels.append(result[1])

    print(classification_report(true_labels, predicted_labels,
                                target_names=['Smoker', 'Non Smoker', 'Unknown', 'Former Smoker']))

    return rc.res


def main():
    handler = DataHandler(filepath='dataset/smoker_status.csv')
    handler.clean_df()
    train, test = handler.split_testsets()

    print("RUN ON TRAIN \n")
    results_train = run_classifier(train)
    write_output(results_train, 'train')

    print("RUN ON TEST \n")
    results_test = run_classifier(test)
    write_output(results_test, 'test')


if __name__ == u"__main__":
    main()
