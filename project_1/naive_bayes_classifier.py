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

        self.feature_prob = {}
        self.prior = 0

        pass

    def calculate_prior(self, data :pd.DataFrame, Y: str):
        """
        Calculates the P(Y=y) for all possible y
        :param data: pd.DataFrame containing training data (including the label column)
        :param Y: str Name of the label column in data
        """

        labels_count = data[Y].value_counts()
        for label,count in labels_count.items():
                labels_count[label] = count / len(data) 
        return labels_count


    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        
        labels = sorted(list(data[target_name].unique()))
        prob_continuous = {}
        

        for column in data.columns:
            label_prob = {}
            for label in labels:

                # calculcate continous
                if data[column].dtypes == float:
                    filtered_data = data[data[target_name]==label]
                    mean,std = filtered_data[column].mean(), filtered_data[column].std()
                    p_dict = {}
                    for value in data[column]:
                        p_dict[value] = ((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((value-mean)**2 / (2 * std**2 ))))
                    prob_continuous[label] = p_dict

                # calculate discrete
                else:
                    con_prob = data.groupby([target_name, column]).size()/ data.groupby([target_name]).size()
                    list_prob = []
                    for option in con_prob.loc[label].index:
                        list_prob.append({option: con_prob.loc[label][option]})
                    
                    label_prob[label] = list_prob 
                    self.feature_prob[column] = label_prob 

        return prob_continuous, self.feature_prob



    def predict_probability(self, data: pd.DataFrame, target_name: str):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        #call fit method here

        prior = {index: data.groupby([target_name]).size().loc[index] for index in data.groupby([target_name]).size().index}


        pass

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        pass


    def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
        feat = list(df.columns)
        df = df[df[Y]==label]
        mean, std = df[feat_name].mean(), df[feat_name].std()
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
        return p_x_given_y