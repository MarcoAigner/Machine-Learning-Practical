import pandas as pd
import math as math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self):
        """
        :param continuous: list containing a bool for each feature column to be analyzed. True if the feature column
                           contains a continuous feature, False if discrete
        """

        self.prob_discrete = {}
        self.gaus_variables = {}
        self.labels = []
        self.prior = {}
        self.discrete = {}

        pass

    def calculate_continuous(self, data: pd.DataFrame, column: str, target_name: str, labels: list):
        variable_continuous = {}

        for label in labels:
            variables_per_label = {}
            filtered_data = data[data[target_name] == label]
            variables_per_label["mean"] = filtered_data[column].mean()
            variables_per_label["std"] = filtered_data[column].std()
            variable_continuous[label] = variables_per_label

        continous_variables = pd.DataFrame.from_dict(variable_continuous)

        return continous_variables

    def calculate_discrete(self, target_labels: list, con_prob: pd.Series, current_label: str):

        label_dict = {label: None for label in target_labels}
        for label_feature in target_labels:
            try:
                label_dict[label_feature] = con_prob.loc[current_label][label_feature]
            except:
                label_dict[label_feature] = 0.0

        return label_dict

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features. 
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """

        self.labels = data[target_name].unique()
        column_dict = {column: {} for column in data.columns.values[1:-1]}
        self.discrete = {label: column_dict.copy() for label in self.labels}

        for column in data.columns:
            if column != target_name:
                # calculcate continous
                if data[column].dtypes == float:
                    self.gaus_variables[column] = self.calculate_continuous(
                        data, column, target_name, self.labels)

                # calculate discrete
                else:
                    for label in self.labels:
                        # calculate discrete
                        con_prob = data.groupby([target_name, column]).size(
                        ) / data.groupby([target_name]).size()
                        self.discrete[label][column] = self.calculate_discrete(
                            self.labels, con_prob, label)

        for label in self.labels:
            self.prob_discrete[label] = pd.DataFrame.from_dict(
                self.discrete[label])

        self.prior = dict(data[target_name].value_counts()/data.shape[0])

        return self.gaus_variables, self.prob_discrete

    def predict_probability(self, data: pd.DataFrame, target_name: str):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """

        likelyhood_prior = {}
        prediction_prob = {}
        prediction_list = []

        # store categorical probabilities across all rows
        probs = {key: [] for key in self.labels}

        for index in data.index:

            for label in self.labels:
                likelyhood_list = []
                for column in data.columns:
                    if data[column].dtypes == float:
                        std = self.gaus_variables[column][label]["std"]
                        mean = self.gaus_variables[column][label]["mean"]
                        likelyhood_list.append(
                            ((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((data[column][index]-mean)**2 / (2 * std**2)))))
                    else:
                        feature_label = data[column][index]
                        likelyhood_list.append(
                            self.prob_discrete[label][column][feature_label])

                # variable änder in zähler
                likelyhood_prior[label] = math.prod(
                    likelyhood_list) * self.prior[label]

            evidence = sum(likelyhood_prior.values())
            for label in self.labels:
                if evidence > float(0):
                    prediction_prob[label] = likelyhood_prior[label]/evidence
                else:
                    prediction_prob[label] = float(0)

            prediction_list.append(
                max(prediction_prob, key=prediction_prob.get))

            # append current row's categorical probabilities
            for key in prediction_prob.keys():
                probs[key].append(prediction_prob[key])

        # create columns with the categorical categories
        for key in probs.keys():
            data[key] = probs[key]

        # changed column name to prediction
        data['prediction'] = prediction_list

        return data

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        pass
