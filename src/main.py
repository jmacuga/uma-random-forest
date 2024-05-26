from datasets.mushrooms import MushroomDataset
from tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier, TournamentRandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    path = "./data/agaricus-lepiota.data"
    dataset = MushroomDataset(path=path)
    dataset.clean()
    X_train, X_val, y_train, y_val = dataset.split(test_size=0.2, random_state=42)

    rf = RandomForestClassifier(10, 3)
    rf.fit(X_train, y_train)
    correct = 0
    for sample, target in zip(X_val, y_val):
        prediction = rf.predict(sample)
        if prediction == target:
            correct += 1
        print("Prediction: " + str(prediction) + " Target: " + str(target))
    print("\nCorrect: " + str(correct))
    print("Num of test samples: " + str(len(X_val)))
    print("Accuracy: " + str(correct / len(X_val)))
