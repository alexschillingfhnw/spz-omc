import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Unterdrücken aller Warnungen
warnings.filterwarnings('ignore')


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


def test_stationarity(timeseries, lags=8, plot=True):
    
    print('Ergebnisse des Dickey-Fuller Tests:')

    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    if plot:
        # Create figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        # Plot the original time series
        axes[0].plot(timeseries)
        axes[0].set_title('Original')
        axes[0].set_xlabel('Zeit')
        axes[0].set_ylabel('Zeitreihe')
        axes[0].grid(True)
        
        # Plot ACF
        plot_acf(timeseries, lags=lags, ax=axes[1])
        axes[1].set_title('ACF')
        
        # Plot PACF
        plot_pacf(timeseries, lags=lags, ax=axes[2])
        axes[2].set_title('PACF')
        
        plt.tight_layout()
        plt.show()


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


def find_best_arima(series):
    """
    Fits ARIMA models with combinations of parameters and finds the best model based on AIC.
    """
    best_aic = float('inf')
    best_order = None
    best_model = None

    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    # Fit ARIMA model and calculate AIC
                    model_fit = arima_model(series, p, d, q)
                    aic = model_fit.aic

                    # Check if we have a new best model
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue

    return best_order, best_model

# This function can now be called with a time series and a range of parameters to find the best ARIMA model

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
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(data=residuals, ax=axs[0])
    axs[0].set_title('Linienplot von ' + title)
    axs[0].set_xlabel('Jahr')
    axs[0].set_ylabel('Residuen')
    axs[0].grid(True, alpha=0.3)
    
    # Adjust the ticks if necessary
    if hasattr(residuals, 'index'):
        axs[0].set_xticks(residuals.index[::2])

    sns.kdeplot(data=residuals, fill=True, ax=axs[1])
    axs[1].set_title('KDE Plot von ' + title)
    axs[1].set_xlabel('Residuen')

    plt.tight_layout()
    plt.show()


def fit_trend_model(series):
    """
    Fits a linear trend model to the series.
    """
    X = np.arange(len(series)).reshape(-1, 1)  # Time variable
    Y = series.values
    model = OLS(Y, sm.add_constant(X)).fit()
    trend = model.predict(sm.add_constant(X))
    residuals = series - trend
    return trend, residuals, model

def check_stationarity(residuals):
    """
    Performs ADF test on residuals to check for stationarity.
    """
    adf_result = adfuller(residuals)
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    for key, value in adf_result[4].items():
        print(f'Critial Values: {key}, {value}')

    if adf_result[1] < 0.05:
        print("Die Residuen sind stationär.")
    else:
        print("Die Residuen sind nicht stationär und könnten von einem ARIMA-Modell profitieren.")


def plot_trend(series, trend, title='Trend Modell'):
    """
    Plots the original series and the fitted trend line.
    
    :param series: The time series data as a Pandas Series.
    :param trend: The fitted trend values as a Pandas Series or array.
    :param title: The title of the plot.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(series, label='Original Series')
    plt.plot(series.index, trend, label='Trend', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
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


def shapiro_test(residuals):
    shapiro_test = stats.shapiro(residuals)
    return {
        "Wert": shapiro_test[0],
        "p-Wert": shapiro_test[1],
        "Interpretation": "Normalverteilt" if shapiro_test[1] > 0.05 else "Nicht normalverteilt"
    }


def plot_residuals_histogram(residuals, num_bins=15):
    """
    Plots a histogram of the residuals overlaid with a normal distribution fit.
    """
    plt.hist(residuals, bins=num_bins, density=True, label='Residuen', alpha=0.6)
  
    # Wahrscheinlichkeitsdichte der Normalverteilung anpassen
    mu, sig = norm.fit(residuals)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sig)
  
    # Normalverteilungskurve plotten
    plt.plot(x, p, 'k', linewidth=2, label='Normalverteilung')
    plt.title('Histogramm der Residuen mit Normalverteilungsfit')
    plt.legend()
    plt.show()


def plot_residuals_ecdf(residuals):
    """
    Plots the empirical cumulative distribution function (ECDF) of the residuals
    alongside the CDF of a normal distribution fit to the residuals.
    """
    ecdf = ECDF(residuals)
    
    # Fit a normal distribution to the data
    mu, sigma = norm.fit(residuals)
    
    # Calculate the CDF for the fitted normal distribution
    x = np.linspace(min(residuals), max(residuals), 100)
    fitted_cdf = norm.cdf(x, loc=mu, scale=sigma)

    plt.plot(ecdf.x, ecdf.y, label='Empirische Verteilung (Residuen)')
    plt.plot(x, fitted_cdf, label='Normalverteilung')

    plt.title('Vergleich der empirischen Verteilung mit der Normalverteilung')
    plt.xlabel('Wert der Residuen')
    plt.ylabel('Kumulierte Wahrscheinlichkeit')
    plt.legend()
    plt.show()


def plot_qq(residuals):
    """
    Creates a Q-Q plot to compare the quantiles of residuals to a normal distribution.
    """
    fig = qqplot(residuals, line='s')
    plt.title('QQ-Plot, Normalverteilung')
    plt.show()
