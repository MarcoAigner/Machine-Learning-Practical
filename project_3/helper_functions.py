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
    disp.ax_.scatter(rows_not_imputed.iloc[:, 0],
                     rows_not_imputed.iloc[:, 1], s=10, c=rows_not_imputed['outlier'], cmap='Set3', marker='o', label='not imputed')
    # imputed datapoints
    disp.ax_.scatter(rows_imputed.iloc[:, 0],
                     rows_imputed.iloc[:, 1], s=10, c=rows_imputed['outlier'], cmap='Set1', marker='^', label='imputed')

    # legend
    disp.ax_.legend()
    disp.ax_.get_legend().legend_handles[0].set_color('black')
    disp.ax_.get_legend().legend_handles[1].set_color('black')

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
            if test_data.dtypes[feature] == 'float64':
                # It's a regression problem
                accuracy_df.loc[strategy, feature] = mean_squared_error(test_data[feature], test_data_imputed_df[feature])
                
            else:
                # It's a classification problem
                accuracy_df.loc[strategy, feature] = accuracy_score(test_data[feature], test_data_imputed_df[feature].astype("int64"))
                
            # Print accuracy for each feature and imputer
            #print(f"Accuracy for {feature} with {strategy} imputer: {accuracy}")

    # Return the DataFrame containing accuracy results
    return accuracy_df
