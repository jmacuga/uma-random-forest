from datasets.mushrooms import MushroomDataset


if __name__ == "__main__":
    dataset = MushroomDataset()
    print(len(dataset))

    X_train, y_train, X_val, y_val = dataset.split(test_size=0.2, random_state=42)
    print(len(X_train))
    X_train, y_train, X_val, y_val = dataset.split(test_size=0.2)
    print(len(X_train))
