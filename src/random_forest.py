import numpy as np


class RandomForestClassifier:
    def __init__(self, n_trees: int, max_depth: int):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def __repr__(self):
        return f"RandomForestClassifier(n_trees={self.n_trees}, max_depth={self.max_depth})"

    def predict(x: np.array):
        """
        Predict class for input x.

        Parameters
        ----------
        x : np.array,
            The input sample.
        """
        pass

    def fit(X: np.array, y: np.array):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        pass

    def _bootstrap(X: np.array, y: np.array):
        """
        Create a bootstrap sample from the data.

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        pass

    def _build_tree(X: np.array, y: np.array, depth: int):
        """
        Build a tree from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.

        depth : int,
            The current depth of the tree.
        """
        pass


class TournamentRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_trees: int, max_depth: int):
        super().__init__(n_trees, max_depth)

    def __repr__(self):
        return f"TournamentRandomForestClassifier(n_trees={self.n_trees}, max_depth={self.max_depth})"

    def _build_tree(X: np.array, y: np.array, depth: int):
        pass
