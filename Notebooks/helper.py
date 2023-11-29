import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lines(df, x, y, title1, title2, xlabel, ylabel, x_ticks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=df, x=x, y=y, ax=ax1)
    ax1.set_title(title1)

    sns.lineplot(data=df, x=x, y=y.diff().fillna(0), ax= ax2)
    ax2.set_title(title2)

    for ax in fig.axes:
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_ticks)
        plt.sca(ax)
        plt.grid(True, alpha=0.3)

    plt.show()