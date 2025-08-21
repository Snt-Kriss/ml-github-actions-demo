import joblib
import numpy as np

def predict(sample):
    model= joblib.load("model.joblib")
    return model.predict([sample])

if __name__ == "__main__":
    sample= [5.1, 3.5, 1.4, 0.2]
    print("Prediction:", predict(sample)) 