import pandas as pd
import math as math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self):
        """
        Initialize the NaiveBayes class.

        Attributes:
        - class_probabilities: A DataFrame to store class probabilities (priors).
        - feature_probabilities: A dictionary to store feature probabilities (likelihoods) for each feature column.
        - target_labels: A list to store the unique labels of the target column.
        - column_target: The name of the target column in the dataset.
        """

        # conditonal probabilities
        self.class_probabilities = pd.DataFrame()  # the priors
        self.feature_probabilities = {}  # the likelihoods

        # other variables
        self.target_labels = []  # the labels of the target column
        self.column_target = None  # name of the target column
        self.df_predict = pd.DataFrame()  # save predictions

    def suitable_data(self, data: pd.DataFrame, required_rows: int = 10) -> tuple[bool, int]:
        """
        Checks whether the given data is suitable for fitting a model. Suitability is determined by the presence of at least two distinct class instances and a minimum number of rows (default is 10).
        :param data: pd.DataFrame containing training data, including the label column.
        :param required_rows: An integer indicating the minimum number of rows required for data to be considered suitable for model fitting (default is 10).
        :return: A tuple containing a boolean value indicating suitability (True if suitable, False if not) and the specified minimum number of rows.
        """

        # Check if there are enough rows in the data (equal to or greater than the required number)
        enough_rows = data.shape[0] >= required_rows

        # Check if there are enough distinct class labels (at least two)
        enough_classes = len(self.target_labels) >= 2

        # Determine if the data is suitable for fitting based on both conditions
        is_suitable = enough_rows and enough_classes

        # Return a tuple with a boolean value indicating suitability and the specified minimum number of rows
        return (is_suitable, required_rows)

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fit the model using the provided training data and store relevant conditional probabilities for discrete and continuous features.
        :param data: pd.DataFrame containing training data, including the label column.
        :param target_name: Name of the label column in the data.
        """

        # Store the target column name in a class variable
        self.column_target = target_name

        # Get unique class labels from the target column
        self.target_labels = data[self.column_target].unique()

        # Check if the data is suitable for model fitting
        data_is_suitable, required_rows = self.suitable_data(data)

        # Raise an exception if the data is not suitable for fitting
        if not data_is_suitable:
            raise Exception(
                f'Training data must contain at least two distinct classes and {required_rows} rows.')

        # Calculate class probabilities and store them in a DataFrame
        self.class_probabilities = data[self.column_target].value_counts(
            normalize=True).reset_index()

        # Extract the feature columns
        feature_columns = data.columns[data.columns !=
                                       self.column_target].values

        # Calculate feature probabilities for each feature column except for the target_column
        # .values since an index is returned
        for feature_column in feature_columns:
            # Initialize a DataFrame within the parent dictionary to store probabilities
            self.feature_probabilities[feature_column] = pd.DataFrame()

            # Check the data type of the feature column
            if data.dtypes[feature_column] == 'float64':
                # For continuous values, calculate and store mean and standard deviation for each class instance
                df = pd.DataFrame(self.target_labels, columns=[
                                  self.column_target])
                df['mean'] = data.groupby(by=self.column_target)[
                    feature_column].mean()
                df['std'] = data.groupby(by=self.column_target)[
                    feature_column].std()
                self.feature_probabilities[feature_column] = df
            elif data.dtypes[feature_column] in ['string', 'bool']:
                # For discrete values, calculate and store normalized value counts for each class instance
                grouped = data.groupby(by=self.column_target)[feature_column]
                self.feature_probabilities[feature_column] = grouped.value_counts(
                    normalize=True).reset_index()
            else:
                # Raise an error for unsupported feature types
                raise TypeError(
                    'Features must be either continuous or boolean')

        return

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted (e.g., test data)
        :return: pd.DataFrame containing probabilities for all categories and the classification result
        """

        # Initialize dictionaries and lists to store probabilities and predictions
        likelyhood_prior = {}   # Stores the likelihood prior for each class
        prediction_prob = {}    # Stores the prediction probabilities for each class
        prediction_list = []    # Stores the final predicted class for each row in the data

        # Create a dictionary to store probabilities for each class
        # where keys are the class labels (target labels)
        probs = {key: [] for key in self.target_labels}

        # Loop through each row (data point) in the input DataFrame
        for index in data.index:
            for label in self.target_labels:  # iterate each label instance
                likelyhood_list = []  # Stores the likelihood for each feature
                for column in data.columns:
                    # Check if the feature is continuous (float data type)
                    if data[column].dtypes == float:
                        # Lookup likelihood for continuous features
                        std = self.feature_probabilities[column]["std"][self.feature_probabilities[column]
                                                                        [self.column_target] == label].iloc[0]
                        mean = self.feature_probabilities[column]["mean"][self.feature_probabilities[column]
                                                                          [self.column_target] == label].iloc[0]
                        likelyhood_list.append(((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-(
                            (data[column].loc[index]-mean)**2 / (2 * std**2)))))
                    else:
                        # Get feature value for discrete probability
                        feature_label = data[column][index]
                        try:
                            # Calculate the conditional probability for the feature
                            likelihood = self.feature_probabilities[column]['proportion'][
                                (self.feature_probabilities[column][self.column_target] == label) & (self.feature_probabilities[column][column] == feature_label)].iloc[0]
                            likelyhood_list.append(likelihood)
                        except:
                            likelyhood_list.append(float(0))

                # Calculate the likelihood prior for the current class
                likelyhood_prior[label] = math.prod(
                    likelyhood_list) * self.class_probabilities['proportion'][self.class_probabilities[self.column_target] == label].iloc[0]

            # Calculate the evidence (sum of likelihood priors)
            evidence = sum(likelyhood_prior.values())

            # Calculate the final prediction probabilities
            for label in self.target_labels:
                if evidence > float(0):
                    prediction_prob[label] = likelyhood_prior[label]/evidence
                else:
                    prediction_prob[label] = float(0)

            # Determine the predicted class label by selecting the one with the highest probability
            prediction_list.append(
                max(prediction_prob, key=prediction_prob.get))

            # Append the prediction probabilities for each class
            for key in prediction_prob.keys():
                probs[key].append(prediction_prob[key])

        # Create columns in the predict DataFrame for each predicted class probability
        for key in probs.keys():
            self.df_predict[key] = probs[key].copy()

        # Add the final column 'prediction' for the predicted class label
        self.df_predict['prediction'] = prediction_list.copy()

        # keep index of input data to distinguish which paitents are in the test_set
        self.df_predict.set_index(data.index.copy(), inplace=True)

        # concat both datasets in order to have a nice dataframe with all the information
        # ALERT -> import not to add columns directly into the orginal input data (cause this is going to cause a change in the data.columns
        # and our loop would try to perform the loop and calcuate the likeleyhood probability for the new columns: True and False and predict
        self.df_predict = pd.concat(
            [data.copy(), self.df_predict.copy()], axis=1)

        # Return the input DataFrame with added columns for probabilities and predictions
        return self.df_predict

    def evaluate_on_data(self, data: pd.DataFrame, test_labels: pd.Series):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        # perform predictions
        predictions = self.predict_probability(
            data=data)['prediction']  # a pd.Series

        # calculate the normalized confusion matrix
        confusion_matrix = pd.crosstab(
            index=predictions, columns=test_labels, margins=True, normalize=True)

        # extract the accuracy out of the confusion matrix
        accuracy = confusion_matrix.at['All', 'All']

        return accuracy, confusion_matrix
