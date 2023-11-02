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
        self.class_probabilities = pd.DataFrame()  # the priors
        self.feature_probabilities = {}  # the likelihoods

        # other variables
        self.target_labels = []
        self.column_target = None

    def suitable_data(self, data: pd.DataFrame, required_rows: int = 10) -> tuple[bool, int]:
        """
        Checks, whether data is suitable for fitting. Suitability is defined by having at least two distinct class instances as well as a defined number of rows (defaults to 10) 
        :param data: pd.DataFrame containing training data (including the label column)
        :param required_rows: int how many rows data needs to contain to count as suitable for fit
        """
        enough_rows = data.shape[0] >= required_rows
        enough_classes = len(self.target_labels) >= 2

        return (True, required_rows) if enough_rows & enough_classes else (False, required_rows)

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features. 
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        # save some data to class variables
        self.column_target = target_name  # label column name
        # class instances
        self.target_labels = data[self.column_target].unique()

        # check if the data is sufficient for fitting
        data_is_suitable, required_rows = self.suitable_data(data)
        if (not data_is_suitable):
            raise Exception(
                f'Training data needs to contain at least two distinct classes and {required_rows} rows.')

        # calculate class probabilities
        self.class_probabilities = data[self.column_target].value_counts(
            normalize=True).reset_index()  # normalize returns values between 0 and 1

        feature_columns = data.drop(columns=self.column_target)

        # calculate feature probabilities
        for feature_column in feature_columns:
            # nest a dataframe within the parent dictionary
            self.feature_probabilities[feature_column] = pd.DataFrame()
            match data.dtypes[feature_column]:
                case 'float64':  # continuous value
                    df = pd.DataFrame(self.target_labels,
                                      columns=[self.column_target])
                    # aggregate functions are performed on each class instance
                    df['mean'] = data.groupby(by=self.column_target)[
                        feature_column].mean()
                    df['std'] = data.groupby(by=self.column_target)[
                        feature_column].std()
                    self.feature_probabilities[feature_column] = df
                case 'string' | 'bool':  # discrete value
                    grouped = data.groupby(by=self.column_target)[
                        feature_column]
                    self.feature_probabilities[feature_column] = grouped.value_counts(
                        normalize=True).reset_index()  # analogous to class probabilities
                case _:
                    raise (
                        TypeError('Features need to either be continuous or boolean'))

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
                        std = self.feature_probabilities[column]["std"][self.feature_probabilities[column]
                                                                        [self.column_target] == label].iloc[0]
                        mean = self.feature_probabilities[column]["mean"][
                            self.feature_probabilities[column][self.column_target] == label].iloc[0]
                        likelyhood_list.append(
                            ((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((data[column].loc[index]-mean)**2 / (2 * std**2)))))
                    else:
                        feature_label = data[column][index]
                        try:
                            likelyhood_list.append(self.feature_probabilities[column]['proportion'][(
                                self.feature_probabilities[column][self.column_target] == label) & (self.feature_probabilities[column][column] == feature_label)].iloc[0])
                        except:
                            likelyhood_list.append(float(0))

                # variable änder in zähler
                likelyhood_prior[label] = math.prod(
                    likelyhood_list) * self.class_probabilities['proportion'][self.class_probabilities[self.column_target] == label].iloc[0]

            evidence = sum(likelyhood_prior.values())

            for label in self.target_labels:
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
    