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

        # Initialize class attributes
        self.prob_discrete_all = {}  # Dictionary to store conditional probabilities for discrete features
        self.gaus_variables = {}  # Dictionary to store statistics for continuous features
        self.target_labels = []  # List of unique labels in the dataset
        self.prior = {}  # Prior probabilities for each label

        pass

    def calculate_continuous(self, train_data: pd.DataFrame, feature_column: str, target_column: str):
        """
        Calculate and store the mean and std for continuous features.

        :param train_data: pd.DataFrame containing training data
        :param feature_column: name of the feature column
        :param target_column: name of the target column
        :return: DataFrame containing statistics for continuous features
        """
        variable_continuous = {}

        for label in self.target_labels:
            variables_per_label = {}
            filtered_data = train_data[train_data[target_column] == label]
            variables_per_label["mean"] = filtered_data[feature_column].mean()
            variables_per_label["std"] = filtered_data[feature_column].std()
            variable_continuous[label] = variables_per_label

        continous_variables = pd.DataFrame.from_dict(variable_continuous)

        return continous_variables

    def calculate_discrete(self, feature_labels: list, conditional_probability_feature: pd.Series, current_target_label: str):
        """
        Calculate and store conditional probabilities for discrete features.

        :param feature_labels: list of unique feature_labels
        :param conditional_probability_feature: conditional probabilities for discrete features with every combination
        :param current_target_label: current target_label being analyzed
        :return: dictionary of conditional probabilities for discrete features
        """
    
        label_dict = {label: None for label in self.target_labels}

        for label in feature_labels:
            try:
                label_dict[label] = conditional_probability_feature.loc[current_target_label][label]
            except:
                label_dict[label] = 0.0

        return label_dict
    
    # mind 2 labels
    # "x" Zeilen -> Begründen

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features. 
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """

        self.target_labels = data[target_name].unique()
        column_dict = {column: {} for column in data.columns.values[1:-1]}
        prob_discrete_dict = {label: column_dict.copy() for label in self.target_labels}

        for column in data.columns:
            if column != target_name:
                # calculcate continous
                if data[column].dtypes == float:
                    self.gaus_variables[column] = self.calculate_continuous(
                        train_data=data, feature_column=column, target_column = target_name)

                # calculate discrete
                else:
                    for label in self.target_labels:
                        con_prob = data.groupby([target_name, column]).size() / data.groupby([target_name]).size()
                        prob_discrete_dict[label][column] = self.calculate_discrete(
                            feature_labels = data[column].unique(), conditional_probability_feature = con_prob, current_target_label = label)

        for label in self.target_labels:
            self.prob_discrete_all[label] = pd.DataFrame.from_dict(prob_discrete_dict[label])

        self.prior = dict(data[target_name].value_counts()/data.shape[0])

        return  # removed return values

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """

        likelyhood_prior = {}
        prediction_prob = {}
        prediction_list = []

        # store categorical probabilities across all rows
        probs = {key: [] for key in self.target_labels}

        for index in data.index:
            for label in self.target_labels:
                likelyhood_list = []
                for column in data.columns:
                    if data[column].dtypes == float:
                        std = self.gaus_variables[column][label]["std"]
                        mean = self.gaus_variables[column][label]["mean"]
                        likelyhood_list.append(
                            ((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((data[column][index]-mean)**2 / (2 * std**2)))))
                    else:
                        feature_label = data[column][index]
                        likelyhood_list.append(self.prob_discrete_all[label][column][feature_label])

                # variable änder in zähler
                likelyhood_prior[label] = math.prod(likelyhood_list) * self.prior[label]

            evidence = sum(likelyhood_prior.values())
            for label in self.target_labels:
                if evidence > float(0):
                    prediction_prob[label] = likelyhood_prior[label]/evidence
                else:
                    prediction_prob[label] = float(0)

            prediction_list.append(max(prediction_prob, key=prediction_prob.get))

            # append current row's categorical probabilities
            for key in prediction_prob.keys():
                probs[key].append(prediction_prob[key])

        # create columns with the categorical categories
        for key in probs.keys():
            data[key] = probs[key]

        # changed column name to prediction
        data['prediction'] = prediction_list

        return data

    def evaluate_on_data(self, data: pd.DataFrame, test_labels: pd.Series):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        # perform predictions
        predictions = self.predict_probability(data=data)['prediction']  # a pd.Series

        # calculate the normalized confusion matrix
        confusion_matrix = pd.crosstab(
            index=predictions, columns=test_labels, margins=True, normalize=True)

        # extract the accuracy out of the confusion matrix
        accuracy = confusion_matrix.at['All', 'All']

        return accuracy, confusion_matrix
