import pandas as pd


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
        self.class_labels = []

    def fit(self, dataframe: pd.DataFrame, label_column: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features. 
        :param data: pd.DataFrame containing training data (including the label column)
        :param label_column: str Name of the label column in data
        """
        # save some data to class variables
        self.label_column = label_column  # label column
        self.class_labels = dataframe[label_column].unique()  # class instances

        # calculate class probabilities
        self.class_probabilities = dataframe[label_column].value_counts(
            normalize=True).reset_index()  # normalize returns values between 0 and 1

        feature_columns = dataframe.drop(columns=label_column)

        # calculate feature probabilities
        for feature_column in feature_columns:
            # nest a dataframe within the parent dictionary
            self.feature_probabilities[feature_column] = pd.DataFrame()
            match dataframe.dtypes[feature_column]:
                case 'float64':  # continuous value
                    df = pd.DataFrame(self.class_labels,
                                      columns=[label_column])
                    # aggregate functions are performed on each class instance
                    df['mean'] = dataframe.groupby(by=label_column)[
                        feature_column].mean()
                    df['std'] = dataframe.groupby(by=label_column)[
                        feature_column].std()
                    self.feature_probabilities[feature_column] = df
                case 'string' | "bool":  # discrete value
                    grouped = dataframe.groupby(by=feature_column)[
                        label_column]
                    self.feature_probabilities[feature_column] = grouped.value_counts(
                        normalize=True).reset_index()  # analogous to class probabilities
                case _:
                    raise (
                        TypeError('Features need to either be continuous or boolean'))

    def predict_probability(self, dataframe: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """

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
                        likelyhood_list.append(
                            self.prob_discrete_all[label][column][feature_label])

                # variable änder in zähler
                likelyhood_prior[label] = math.prod(
                    likelyhood_list) * self.prior[label]

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

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        pass
