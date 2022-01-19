import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(self.filepath)

    def clean_df(self):
        """
        basic preprocessing applied to dataframe's text column
        (removal or replacement of special chars, punctuation, multiple spaces)
        :return: None
        """
        self.df['text'] = self.df['text'].replace({r'\s\s+': ' ', '\n': '', '[,().:;#]': '', ' - ': ' ', '- ': ' '}, regex=True)
        self.df['text'] = self.df['text'].str.strip()
        self.df['text'] = self.df['text'].str.lower()

        # return self.df[self.df['text'].apply(lambda x: len(x.split()) > 1)]

    def get_classes_sets(self, status_value):
        """
        Method used for exploratory analysys
        :param status_value: one of the df classes (Non Smoker, Smoker, Former Smoker, Unknown)
        :return: list of all text items in a specific class
        """
        filtered_df = self.df.loc[self.df['status'] == status_value]
        return filtered_df['text'].tolist()

    @staticmethod
    def get_most_frequent_tokens_per_class(filtered_df_list):
        """
        Method used for exploratory analysys
        returns dict in descending order (most frequent tokens first)
        stopwords are removed
        :param filtered_df_list:
        :return: frequency dict
        """
        stopwords = ['of', 'to', 'and', 'in', 'at', 'by', 'with', 'for', 'a', 'the', '']
        tokens = [record.split() for record in filtered_df_list]
        bag_of_words = [token for sent in tokens for token in sent]
        no_stopwords = [word for word in bag_of_words if word not in stopwords]
        return Counter(no_stopwords)

    def split_testsets(self):
        """
        Splits of original df in train and test (to verify results on unseen data)
        :return:
        """
        # X = self.df['text']
        # y = self.df['status']
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
        train, test  = train_test_split(self.df, test_size=0.3, random_state=5)
        return train, test





