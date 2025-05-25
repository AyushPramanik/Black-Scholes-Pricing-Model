import pandas as pd
def generate_dataset(n_samples=100000):
    np.random.seed(42)
    S = np.random.uniform(50, 150, n_samples)
    K = np.random.uniform(50, 150, n_samples)
    T = np.random.uniform(0.01, 2, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)

    data = pd.DataFrame({'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma})
    data['price'] = black_scholes(S, K, T, r, sigma, option_type='call')
    return data
