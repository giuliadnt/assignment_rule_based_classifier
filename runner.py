import csv
from data_handler import DataHandler
from classifier import RuleClassifier
from sklearn.metrics import classification_report


def write_output(res, set_name):
    with open(f'results_{set_name}.csv', 'w') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['status', 'text'])
        for row in res:
            csv_out.writerow(row)


def run_classifier(split_dataset, true_labels):
    predicted_labels = []
    rc = RuleClassifier(split_dataset, kb_path='kb.json')

    for text in rc.data:
        rc.classify(text)

    for result in rc.res:
        predicted_labels.append(result[0])

    print(classification_report(true_labels, predicted_labels,
                                target_names=['Smoker', 'Non Smoker', 'Unknown', 'Former Smoker']))

    return rc.res


def main():
    handler = DataHandler(filepath='dataset/smoker_status.csv')
    handler.clean_df()
    X_train, X_test, y_train, y_test = handler.split_testsets()

    print("RUN ON TRAIN \n")
    results_train = run_classifier(X_train, y_train)
    write_output(results_train, 'train')

    print("RUN ON TEST \n")
    results_test = run_classifier(X_test, y_test)
    write_output(results_test, 'test')


if __name__ == u"__main__":
    main()
