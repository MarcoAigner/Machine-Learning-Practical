import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
from sklearn.decomposition import PCA
import numpy as np
from sklearn.inspection._plot.decision_boundary import DecisionBoundaryDisplay
from pandas.core.indexes.base import Index
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import IsolationForest


import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from pandas.core.indexes.base import Index
import random


def generate_anomalies(quantity: int):
    """ Generates a list of anomaly datapoints based on random values.  

    The ranges out of which the attributes' values are picked are hardcoded, based on the distribution of phase_3 data

    Args:
        quantity (int): How many anomalies should get generated

    Returns:
        df (pd.DataFrame): dataframe with (quantity) rows of generated anomaly datapoints
    """

    df = pd.DataFrame()

    for i in range(0, quantity):
        age = random.randint(15, 70)
        sex = random.choice(['m', 'f'])
        alb = random.randint(10, 80)
        alp = random.randint(10, 200)
        alt = random.randint(15, 170)
        ast = random.randint(15, 350)
        bil = random.randint(3, 40)
        che = random.randint(3, 25)
        chol = random.randint(1, 12)
        crea = random.randint(5, 70)
        ggt = random.randint(5, 300)
        prot = random.randint(0, 100)

        new_row = [age, sex, alb, alp, alt,
                   ast, bil, che, chol, crea, ggt, prot]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.columns = ['Age', 'Sex', 'ALB', 'ALP', 'ALT',
                  'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

    return df


def explain_prediction_dt(trained_decision_tree: DecisionTreeClassifier, data_sample: np.ndarray, class_labels: list[str]) -> None:
    """ Prints a predicted class and a decision path for a given decision tree and data sample

    Args:
        trained_decision_tree (DecisionTreeClassifier): Decision tree classifier on which .fit() has already been called
        data_sample (np.ndarray): One row of sample data to predict upon
        class_names (list[str]): List of class names according to the numerical class encodings

    Returns:
        None: Only prints
    """

    # get the predicted class and path taken
    predicted_class = get_predicted_class(
        trained_decision_tree, data_sample, class_labels)
    decision_path = get_decision_path(trained_decision_tree, data_sample)

    # format the outputs
    decision_path_string = list(map(str, decision_path))
    other_nodes = ', '.join(decision_path_string[:-1])
    last_node = decision_path_string[-1]

    # print
    print(f'Predicted Class: {predicted_class}')
    print(f'Path taken: Nodes {other_nodes} and {last_node}')

def explain_prediction_iso(trained_isolation_forest: IsolationForest, data_sample: np.ndarray):
    """ Prints a predicted anomaly for a given data sample

    Args:
        trained_isolation_forest (IsolationForest): Isolationforest on which .fit() has already been called
        data_sample (np.ndarray): One row of sample data to predict upon

    Returns:
        None: Only prints
    """
    predicted_anomaly = trained_isolation_forest.predict(data_sample)
    decision = trained_isolation_forest.decision_function(data_sample)
                
    if predicted_anomaly == -1:
        print("This patient shows an anomaly")
    else:
        print("This patient shows no anomaly")
    print("Calculated value for the anomaly is:", decision, "\n")


def get_predicted_class(trained_decision_tree: DecisionTreeClassifier, data_sample: np.ndarray, class_labels: list[str]) -> str:
    """ Predicts a sample using a DecisionTree and returns a class label

    Args:
        trained_decision_tree (DecisionTreeClassifier): Decision tree classifier on which .fit() has already been called
        data_sample (np.ndarray): One row of sample data to predict upon
        class_names (list[str]): List of class names according to the numerical class encodings

    Returns:
        predicted_class (str): The textual representation of the predicted class
    """

    prediction = trained_decision_tree.predict(data_sample)[0]
    predicted_class = class_labels[prediction]

    return predicted_class


def get_decision_path(trained_decision_tree: DecisionTreeClassifier, data_sample: np.ndarray) -> list[int]:
    """ Returns a list of nodes in given decison tree passes while predicting a data sample

    Args:
        trained_decision_tree (DecisionTreeClassifier): Decision tree classifier on which .fit() has already been called
        data_sample (np.ndarray): One row of sample data to predict upon

    Returns:
        passed_nodes (list[int]): A list of the nodes the decision tree passes while predicting ordered from root to leaf
    """
    decision_path = pd.DataFrame(
        trained_decision_tree.decision_path(data_sample).toarray()).transpose()
    passed_nodes = decision_path.index[decision_path[0] == 1].to_list()
    return passed_nodes


def plot_decision_tree(decision_tree: DecisionTreeClassifier, feature_names: list, class_names: list[str]) -> None:
    """ Plots a fitted decision tree

     Args:
        decision_tree (DecisionTreeClassifier): an already fitted decison tree classifier
        feature_names (list): list of the feature names (list(X_train.columns))
        class_names (list[str]): textual representation of the classes

    Returns:
        None: Prints out the plot
    """
    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree,  # classifier to plot
              filled=True,  # color-code according to classes
              feature_names=feature_names,  # plotted on top of a box
              class_names=class_names,  # plotted at the bottom of a box
              impurity=False,  # for easier understanding
              precision=2,  # round values
              node_ids=True  # show node ids
              )

    plt.title('Hemogram-Based Classification of Blood Donors', fontsize=15)

    plt.show()


def decision_boundary_plot(df: pd.DataFrame, estimator, imputed_rows: Index) -> DecisionBoundaryDisplay:
    """ Plots the decision boundary given primary components and an estimator and returns it

    Displays original data points as circles and imputed ones as pyramids. 
    Color-codes inliners and outliers.

    Args:
        df (pd.DataFrame): dataframe containing principal components and a column encoding 'outliers'
        estimator: sklearn estimator object to generate the decision boundary for
        imputed_rows (pandas.core.indexes.base.Index): Indices of imputed rows

    Returns:
        disp (sklearn.inspection._plot.decision_boundary.DecisionBoundaryDisplay): plotted decision boundary display object
    """

    # enhance readability
    first_two_pcs = df.iloc[:, :2]
    rows_imputed = df.iloc[imputed_rows]
    rows_not_imputed = df.drop(index=imputed_rows)

    # decision boundary (background)
    disp = DecisionBoundaryDisplay.from_estimator(
        estimator=estimator,
        X=first_two_pcs,
        xlabel='1st principal component',
        ylabel='2nd principal component',
    )

    # original datapoints
    scatter_not_imputed = disp.ax_.scatter(rows_not_imputed.iloc[:, 0],
                                           rows_not_imputed.iloc[:, 1], s=10, c=rows_not_imputed['outlier'], cmap='Set3', marker='o', label='not imputed')
    # imputed datapoints
    scatter_imputed = disp.ax_.scatter(rows_imputed.iloc[:, 0],
                                       rows_imputed.iloc[:, 1], s=10, c=rows_imputed['outlier'], cmap='Set1', marker='^', label='imputed')

    # legend for normal values
    legend_labels = ['not imputed', 'imputed']
    legend1_handles = [disp.ax_.scatter([], [], c='black', marker='o', s=10),
                       disp.ax_.scatter([], [], c='black', marker='^', s=10)]
    legend1 = disp.ax_.legend(handles=legend1_handles,
                              labels=legend_labels, loc='upper left')

    # create separate legends for 'o' and '^' markers with -1 color for outliers
    legend2_o = disp.ax_.legend(*scatter_not_imputed.legend_elements(),
                                title='Outliers (o)', loc='upper right', bbox_to_anchor=(1, 0.5))
    legend2_hat = disp.ax_.legend(*scatter_imputed.legend_elements(),
                                  title='Outliers (^)', loc='lower right', bbox_to_anchor=(1, 0.5))

    # add both legends to the plot
    disp.ax_.add_artist(legend1)
    disp.ax_.add_artist(legend2_o)
    disp.ax_.add_artist(legend2_hat)

    # title
    disp.ax_.set_title('Decision Boundary')

    return disp


def feature_distribution_plot(df: pd.DataFrame, suptitle: str) -> matplotlib.figure.Figure:
    """ Plots data distributions and returns it

    Plots one histplot including distribution for each column of a given dataframe
    Additionally displays a boxplot with the varability of the whole dataframe

    Args:
        ax (matplotlib.axes.Axes): Axes the plot is drawn onto
        df (pd.DataFrame): the dataframe containing the columns of interest
        suptitle (str): title above all subplots; argument for plt.suptitle()

    Returns:
        ax (plt.axes): axes containing the plot
    """

    fig, axes = plt.subplots(nrows=3, ncols=4,
                             figsize=(12, 8), constrained_layout=True)

    # single distribution plots
    for col, ax in zip(df.columns, axes.flat):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)

    # boxplot in last axes
    last_axis = axes.flat[-1]
    last_axis.set_title('Dataset Variability')
    sns.boxplot(data=df, ax=last_axis)
    plt.xticks(rotation=90)

    plt.suptitle(suptitle)

    return fig


def scree_plot(ax: matplotlib.axes.Axes, pca: PCA, title: str) -> matplotlib.axes.Axes:
    """ Plots a scree plot on a given axes and returns it

    Contains a bar-chart indicating the contributed explained variance ratio per principal component
    Additionally displays the cumulative explained variance for multiple components.

    Args:
        ax (matplotlib.axes.Axes): Axes the plot is drawn onto
        pca (PCA): fitted principal component analysis object containing its principal components 
        title (str): title for the axes

    Returns:
        ax (plt.Axes): axes containing the plot
    """

    # calculate summed explained ratios
    explained_ratio = cumulative_explained_ratio(pca)

    # plots
    barplot = sns.barplot(pca.explained_variance_ratio_,
                          ax=ax, label='partial')
    sns.lineplot(explained_ratio, ax=ax, drawstyle='steps-mid',
                 color='orange', label='cumulative')

    # labels
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xlabel('Principal Component')

    # grid
    ax.grid()
    ax.set_axisbelow(True)

    # legend
    ax.legend(loc='center right')

    # axes
    ax.set_yticks(np.linspace(0, 1, 21))  # steps of five on y-axis
    ax.set_xticks(range(0, 11), range(1, 12, 1))  # type: ignore

    # set a title
    ax.set_title(title)

    # display the values
    barplot.bar_label(ax.containers[0],  # type: ignore
                      fmt='{0:.2f}', padding=2)  # type: ignore

    return ax


def pca_plot(ax: matplotlib.axes.Axes, first_pc: pd.Series, second_pc: pd.Series, title: str) -> matplotlib.axes.Axes:
    """ Plots two principal components against each other and returns the axes

    Args:
        ax (matplotlib.axes.Axes): Axes the plot is drawn onto
        first_pc (pd.Series): first principal component's values
        second_pc (pd.Series): second principal component's values
        title (str): title for the axes

    Returns:
        ax (plt.Axes): axes containing the plot
    """

    ax.scatter(first_pc, second_pc)

    ax.set_xlabel('1st principal component')
    ax.set_ylabel('2nd principal component')

    ax.set_title(title)

    return ax


def cumulative_explained_ratio(pca: PCA) -> list[float]:
    """ Returns a list of the cumulative explained ratios of principal component

    First entry will be the explained ratio of the first principal component.
    Second entry will be the sum of the explained ratios of first and second and so on  

    Args:
        pca (PCA): fitted principal component analysis object containing its principal components 

    Returns:
        cumulative_explained_ratio (list[float]): added up explained variance ratios
    """
    # calculate cumulative explained ratios for plotting
    cumulative_explained_ratio: list[float] = []

    # loop over each principal component's explained variance ratio
    for index, value in enumerate(pca.explained_variance_ratio_):
        if len(cumulative_explained_ratio) == 0:
            # save the first value as is
            cumulative_explained_ratio.append(value)
        else:
            # add subsequent values to their predecessor and save
            cumulative_explained_ratio.append(
                value + cumulative_explained_ratio[index - 1])

    return cumulative_explained_ratio


def imputation_and_accuracy(train_data, test_data, features, imputer_dict):
    """ Returns a dataframe including the accuracies for all columns containing nan-values using different imputation methods

    Args:
        train_data: data that should be used to train the imputer
        test_data: data that should be used to test the imputer
        features: list of feature names containing nan-values
        imputer_dict: 
                key: imputation name
                value: instanz of the matching imputer

    Returns:
        dataframe: accuracy for all columns containing nan-values using different imputation methods
    """
    # Initialize a DataFrame to store accuracy results
    accuracy_df = pd.DataFrame(columns=features, index=imputer_dict.keys())

    for feature in features:
        for strategy, imputer in imputer_dict.items():
            # Fit and transform on train data
            imputer.fit(train_data)

            # Impute only the NaN values in the test set
            test_data_imputed = test_data.copy()

            # Remove original values of the target feature in the test set
            test_data_imputed[feature] = None

            test_data_imputed_df = imputer.transform(test_data_imputed)

           # Check the type of your target variable and choose the appropriate metric
            accuracy_df.loc[strategy, feature] = mean_squared_error(
                test_data[feature], test_data_imputed_df[feature])

    # Return the DataFrame containing accuracy results
    return accuracy_df
