import joblib

def save_model(model, path='bs_model.pkl'):
    joblib.dump(model, path)

def load_model(path='bs_model.pkl'):
    return joblib.load(path)



def save_model(model, path='bs_model.pkl'):
    joblib.dump(model, path)

def load_model(path='bs_model.pkl'):
    return joblib.load(path)
