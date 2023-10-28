import pandas as pd
import math as math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self):
        """
        """

        # conditonal probabilities
        self.class_probabilities = pd.DataFrame()
        self.feature_probabilities = {}

        # other variables
        self.class_labels = []

    def fit(self, dataframe: pd.DataFrame, label_column: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features. 
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        # class instances
        self.class_labels = dataframe[label_column].unique()

        # calculate class probabilities
        self.class_probabilities = dataframe[label_column].value_counts(
            normalize=True)

        feature_columns = dataframe.drop(columns=label_column)

        # calculate feature probabilities
        for feature_column in feature_columns:
            self.feature_probabilities[feature_column] = pd.DataFrame()
            match dataframe.dtypes[feature_column]:
                case 'float64':  # continuous value
                    df = pd.DataFrame(self.class_labels,
                                      columns=[label_column])
                    df['mean'] = dataframe.groupby(by=label_column)[
                        feature_column].mean()
                    df['std'] = dataframe.groupby(by=label_column)[
                        feature_column].std()

                    self.feature_probabilities[feature_column] = df
                case 'bool':  # discrete value
                    grouped = dataframe.groupby(by=feature_column)[
                        label_column]
                    self.feature_probabilities[feature_column] = grouped.value_counts(
                        normalize=True).reset_index()
                case other:
                    raise (
                        TypeError('Features need to either be continuous or boolean'))

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        pass
