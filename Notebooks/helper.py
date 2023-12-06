import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


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


def adf_test(series):
    """
    Performs the Augmented Dickey-Fuller test on the given series and prints the results.
    """
    result = adfuller(series, autolag='AIC')
    adf_stats = result[0]
    p_value = result[1]
    critical_values = result[4]

    print(f'Augmented Dickey-Fuller Test:')
    print(f'ADF-Statistik: {adf_stats}')
    print(f'p-Wert: {p_value}')
    print(f'Kritische Werte: {critical_values}\n')


def plot_acf_pacf(series, lags=20):
    """
    Plots the ACF and PACF of the given series.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plot_acf(series, lags=lags, ax=plt.gca())
    plt.title('Autokorrelationsfunktion (ACF)')
    
    plt.subplot(1, 2, 2)
    plot_pacf(series, lags=lags, method='ols', ax=plt.gca())
    plt.title('Partielle Autokorrelationsfunktion (PACF)')
    
    plt.tight_layout()
    plt.show()


def arima_model(data, p, d, q):
    """
    Trains an ARIMA model with the given parameters and returns the fitted model.
    """
    arima_model = ARIMA(data, order=(p, d, q))
    arima_result = arima_model.fit()
    return arima_result


def plot_arima_model(data, arima_result, title, ylabel):
    """
    Plots the given data and the fitted ARIMA model.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(data, label='Historische Daten')
    
    predictions = arima_result.predict(start=1, end=len(data) - 1, typ='levels')

    plt.plot(predictions, label='Prognose')

    plt.title(title)
    plt.xlabel('Jahr')
    plt.ylabel(ylabel)
    plt.xticks(data.index[::2])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_residuals(residuals, title):
    """
    Plots the residuals of the given series.
    """
    # Create a subplot grid of 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # First subplot for the line plot of residuals
    sns.lineplot(data=residuals, ax=axs[0])
    axs[0].set_title('Linienplot von' + title)
    axs[0].set_xlabel('Jahr')
    axs[0].set_ylabel('Residuen')
    axs[0].grid(True, alpha=0.3)
    
    # Adjust the ticks if necessary
    if hasattr(residuals, 'index'):
        axs[0].set_xticks(residuals.index[::2])

    # Second subplot for the KDE plot of residuals
    sns.kdeplot(data=residuals, fill=True, ax=axs[1])
    axs[1].set_title('KDE Plot von ' + title)
    axs[1].set_xlabel('Residuen')

    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.show()


def get_predictions(n, model, data):
    """
    Returns the predictions for the next n years.
    """

    # Erstellen der Prognose für die nächsten 10 Jahre
    prognose_jahre = n
    prognose_start = data['Jahr'].iloc[-1] + 1  # Startjahr der Prognose
    prognose_ende = prognose_start + prognose_jahre - 1  # Endjahr der Prognose

    # Erstellen der Prognose
    prognose = model.get_forecast(steps=prognose_jahre)
    prognose_mean = prognose.predicted_mean
    prognose_conf_int = prognose.conf_int(alpha=0.05)  # 95% Konfidenzintervall

    return prognose_start, prognose_ende, prognose_mean, prognose_conf_int
