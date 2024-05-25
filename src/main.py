from datasets.mushrooms import MushroomDataset
from tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier, TournamentRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


if __name__ == "__main__":
    path = "./data/agaricus-lepiota.data"
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
    dc = DecisionTreeClassifier(7)

    start = time.process_time()
    dc.fit(X_train, y_train)
    print(f"Time: {time.process_time() - start}s")

    y_preds = dc.predict(X_val)

    accuracy = accuracy_score(y_val, y_preds)
    confusion_matrix = confusion_matrix(y_val, y_preds)

    print("Num of test samples: " + str(len(X_val)))
    print(f"Accuracy: {accuracy}")
    print("Confusion matrix:")
    print(confusion_matrix)
