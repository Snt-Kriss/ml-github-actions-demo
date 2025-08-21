import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model():
    iris= load_iris()
    X, y= iris.data, iris.target
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

    clf= LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds= clf.predict(X_test)
    acc= accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2f}")

    joblib.dump(clf, "model.joblib")

    return acc

if __name__ == "__main__":
    train_and_save_model()