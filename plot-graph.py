import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs_model import black_scholes  # import your BS function

def plot_price_vs_volatility():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigmas = np.linspace(0.05, 1.0, 50)

    prices = [black_scholes(S, K, T, r, sigma, option_type='call') for sigma in sigmas]
    df = pd.DataFrame({'Volatility': sigmas, 'Option Price': prices})

    sns.set(style="darkgrid")
    sns.lineplot(data=df, x='Volatility', y='Option Price')
    plt.title("Option Price vs Volatility (Black-Scholes)")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Call Option Price")
    plt.show()

def plot_ml_vs_analytical(model):
    # Generate random test data
    n = 200
    S = np.random.uniform(80, 120, n)
    K = np.random.uniform(80, 120, n)
    T = np.random.uniform(0.1, 2.0, n)
    r = np.random.uniform(0.01, 0.05, n)
    sigma = np.random.uniform(0.1, 0.5, n)

    analytical = [black_scholes(S[i], K[i], T[i], r[i], sigma[i]) for i in range(n)]
    ml = model.predict(np.column_stack([S, K, T, r, sigma]))

    df = pd.DataFrame({'Analytical': analytical, 'ML_Predicted': ml})
    sns.set(style="whitegrid")
    sns.scatterplot(data=df, x="Analytical", y="ML_Predicted", color='blue')
    plt.title("ML Predicted vs Analytical Option Prices")
    plt.xlabel("Analytical Price")
    plt.ylabel("ML Predicted Price")
    plt.plot([min(analytical), max(analytical)], [min(analytical), max(analytical)], 'r--')  # y=x line
    plt.show()

def plot_price_vs_strike():
    S = 100
    K_values = np.linspace(80, 120, 50)
    T = 1
    r = 0.05
    sigma = 0.2

    prices = [black_scholes(S, K, T, r, sigma) for K in K_values]
    df = pd.DataFrame({'Strike': K_values, 'Option Price': prices})

    sns.lineplot(data=df, x='Strike', y='Option Price')
    plt.title("Option Price vs Strike")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.show()
