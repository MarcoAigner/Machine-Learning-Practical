import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
from sklearn.decomposition import PCA
import numpy as np


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