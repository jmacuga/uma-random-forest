from datasets.mushrooms import MushroomDataset
from tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    path = "/home/user/uma-random-forest/data/mushroom/agaricus-lepiota.data"
    dataset = MushroomDataset(path=path)
    dataset.clean()
    X_train, X_val, y_train, y_val = dataset.split(test_size=0.2, random_state=42)

    print(len(X_train))
    print(X_train.shape)
    print(len(y_train))
    print(y_train.shape)
    # print(y_val)

    dc = DecisionTreeClassifier(3)

    correct = 0
    dc.fit(X_train, y_train)
    for sample, target in zip(X_val, y_val):
        prediction = dc.predict(sample)
        if prediction == target:
            correct += 1
        print("Prediction: " + str(prediction) + " Target: " + str(target))
    print("\nCorrect: " + str(correct))
    print("Num of test samples: " + str(len(X_val)))
    print("Accuracy: " + str(correct / len(X_val)))
