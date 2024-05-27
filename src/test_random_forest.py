from datasets.mushrooms import MushroomDataset
from tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier, TournamentRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import math, time


if __name__ == "__main__":
    path = "./data/mushroom/agaricus-lepiota.data"
    dataset = MushroomDataset(path=path)
    dataset.clean()
    X_train, X_val, y_train, y_val = dataset.split(test_size=0.2, random_state=42)

    print(len(X_train))
    print(X_train.shape)
    print(len(y_train))
    print(y_train.shape)

    n_features = round(math.sqrt(X_train.shape[1]))
    print(f"n_features: {n_features}")

    rf = RandomForestClassifier(n_trees=10, max_depth=3, max_features=n_features)
    rf.fit(X_train, y_train)

    start = time.process_time()

    print(f"Time: {time.process_time() - start}s")

    y_preds = [rf.predict(x) for x in X_val]

    accuracy = accuracy_score(y_val, y_preds)
    confusion_matrix = confusion_matrix(y_val, y_preds)

    print("Random forest:\n")
    print("Num of test samples: " + str(len(X_val)))
    print(f"Accuracy: {accuracy}")
    print("Confusion matrix:")
    print(confusion_matrix)
