from sklearn.ensemble import RandomForestRegressor

def train_model(data):
    X = data[['S', 'K', 'T', 'r', 'sigma']]
    y = data['price']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model
