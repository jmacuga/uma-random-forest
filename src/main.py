from datasets.mushrooms import MushroomDataset
from tree import RandomizedDecisionTreeClassifier, DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import math

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
    # dc.get_best_split(X_train, y_train)

    n_features = round(math.sqrt(X_train.shape[1]))
    # dc = RandomizedDecisionTreeClassifier(3, max_features=n_features)
    dc = DecisionTreeClassifier(3)

    start = time.process_time()
    dc.fit(X_train, y_train)
    print(f"Time: {time.process_time() - start}s")

    correct = 0
    for sample, target in zip(X_val, y_val):
        prediction = dc.predict(sample)
        if prediction == target:
            correct += 1
        # print("Prediction: " + str(prediction) + " Target: " + str(target))
    print("\nCorrect: " + str(correct))
    print("Num of test samples: " + str(len(X_val)))
    print("Accuracy: " + str(correct / len(X_val)))
