import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import joblib

from black_scholes import black_scholes

# Price vs Volatility
def plot_price_vs_volatility():
    S, K, T, r = 100, 100, 1, 0.05
    sigmas = np.linspace(0.05, 1.0, 50)
    prices = [black_scholes(S, K, T, r, sigma, option_type='call') for sigma in sigmas]
    df = pd.DataFrame({'Volatility (σ)': sigmas, 'Option Price': prices})

    sns.set(style="darkgrid")
    sns.lineplot(data=df, x='Volatility (σ)', y='Option Price')
    plt.title("Option Price vs Volatility (Black-Scholes)")
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Call Option Price ($)")
    plt.show()

# Price vs Strike Price
def plot_price_vs_strike():
    S, T, r, sigma = 100, 1, 0.05, 0.2
    strikes = np.linspace(50, 150, 50)
    prices = [black_scholes(S, K, T, r, sigma, option_type='call') for K in strikes]
    df = pd.DataFrame({'Strike Price (K)': strikes, 'Option Price': prices})

    sns.set(style="darkgrid")
    sns.lineplot(data=df, x='Strike Price (K)', y='Option Price')
    plt.title("Option Price vs Strike Price (Black-Scholes)")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Call Option Price ($)")
    plt.show()

# Price vs Time to Maturity
def plot_price_vs_maturity():
    S, K, r, sigma = 100, 100, 0.05, 0.2
    times = np.linspace(0.1, 5.0, 50)
    prices = [black_scholes(S, K, T, r, sigma, option_type='call') for T in times]
    df = pd.DataFrame({'Time to Maturity (T)': times, 'Option Price': prices})

    sns.set(style="darkgrid")
    sns.lineplot(data=df, x='Time to Maturity (T)', y='Option Price')
    plt.title("Option Price vs Time to Maturity (Black-Scholes)")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Call Option Price ($)")
    plt.show()

# ML vs Analytical
def plot_ml_vs_analytical(model_path="model.pkl"):
    model = joblib.load(model_path)
    n = 200
    S = np.random.uniform(80, 120, n)
    K = np.random.uniform(80, 120, n)
    T = np.random.uniform(0.1, 2.0, n)
    r = np.random.uniform(0.01, 0.05, n)
    sigma = np.random.uniform(0.1, 0.5, n)

    X = np.stack([S, K, T, r, sigma], axis=1)
    y_true = np.array([black_scholes(S[i], K[i], T[i], r[i], sigma[i], option_type='call') for i in range(n)])
    y_pred = model.predict(X)

    df = pd.DataFrame({'Black-Scholes Price': y_true, 'ML Predicted Price': y_pred})

    sns.set(style="darkgrid")
    sns.scatterplot(data=df, x='Black-Scholes Price', y='ML Predicted Price')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Fit')
    plt.title("ML Predicted vs Black-Scholes Analytical")
    plt.xlabel("Analytical Price")
    plt.ylabel("ML Predicted Price")
    plt.legend()
    plt.show()

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-Scholes Visualizer")
    parser.add_argument("--plot", required=True, choices=[
        "volatility", "strike", "maturity", "ml_vs_bs"
    ], help="Choose the plot to display")
    parser.add_argument("--model", default="model.pkl", help="Path to trained ML model (for ml_vs_bs)")

    args = parser.parse_args()

    if args.plot == "volatility":
        plot_price_vs_volatility()
    elif args.plot == "strike":
        plot_price_vs_strike()
    elif args.plot == "maturity":
        plot_price_vs_maturity()
    elif args.plot == "ml_vs_bs":
        plot_ml_vs_analytical(args.model)
