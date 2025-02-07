import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, column, title=None):
    """
    Perform univariate analysis and plot distribution.
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name for analysis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(title or f"Univariate Analysis of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def bivariate_analysis(df, x_column, y_column, title=None):
    """
    Perform bivariate analysis and plot relationship.
    Args:
        df (pd.DataFrame): Input dataframe.
        x_column (str): X-axis column name.
        y_column (str): Y-axis column name.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column, hue='class' if 'class' in df.columns else None)
    plt.title(title or f"Bivariate Analysis of {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def correlation_matrix(df):
    """
    Plot correlation matrix heatmap.
    Args:
        df (pd.DataFrame): Input dataframe.
    """
    corr = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()