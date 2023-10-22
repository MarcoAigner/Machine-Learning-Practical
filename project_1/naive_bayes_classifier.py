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

        self.prob_discrete = None
        self.gaus_variables = {}
        self.labels = []
        self.prior = 0

        pass

    def calculate_prior(self, data: pd.DataFrame, Y: str):
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
        variable_continuous = {}
        prob_discrete = {key: None for key in data.columns.values[1:]}

        self.labels = list(data[target_name].unique())
        
        for column in data.columns:
            label_prob = {}
            for label in self.labels:

                # calculcate continous
                if data[column].dtypes == float:
                    variables_per_label = {}
                    filtered_data = data[data[target_name]==label]
                    variables_per_label["mean"] = filtered_data[column].mean()
                    variables_per_label["std"] = filtered_data[column].std()
                    variable_continuous[label] = variables_per_label
                    self.gaus_variables[column] = pd.DataFrame.from_dict(variable_continuous)

                # calculate discrete
                else:
                    con_prob = data.groupby([target_name, column]).size()/ data.groupby([target_name]).size()
                    
                    try: 
                        label_prob[label] = con_prob.loc[label][True]
                    except:
                        label_prob[label] = 0.0   

                    prob_discrete[column] = label_prob 

            
        self.prob_discrete = pd.DataFrame.from_dict(prob_discrete) 

        self.prior = {index: data.groupby([target_name]).size().loc[index]/data.shape[0] for index in data.groupby([target_name]).size().index}

        return self.gaus_variables, self.prob_discrete



    def predict_probability(self, data: pd.DataFrame, target_name: str):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted    X_test
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        
        likelyhood_list= []
        likelyhood_prior = {}
        prediction_prob = {}
        prediction_list = []
      
        for index in data.index:
            for label in self.labels:
                for column in data.columns:
                    if data[column].dtypes == float:
                        std = self.gaus_variables[column][label]["std"]
                        mean = self.gaus_variables[column][label]["mean"]
                        likelyhood_list.append(((1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((data[column][index]-mean)**2 / (2 * std**2 )))))
                    else:
                        if data[column][index] == True:
                            likelyhood_list.append(self.prob_discrete[column][label])
                    

                likelyhood_prior[label] = math.prod(likelyhood_list) * self.prior[label]
                
                
            evidence = sum(likelyhood_prior.values())
            for label in self.labels:
                if evidence > float(0):
                    prediction_prob[label] = likelyhood_prior[label]/evidence
                else:
                    prediction_prob[label] = float(0)
            
            prediction_list.append(max(prediction_prob, key=prediction_prob.get))
        
        data[target_name] = prediction_list

        return data

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """

        pass


    