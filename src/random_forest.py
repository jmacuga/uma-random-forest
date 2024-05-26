import numpy as np
from tree import DecisionTreeClassifier, TournamentDecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_trees: int, max_depth: int):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def __repr__(self):
        return f"RandomForestClassifier(n_trees={self.n_trees}, max_depth={self.max_depth})"

    def predict(self, x: np.array):
        """
        Predict class for input x.

        Parameters
        ----------
        x : np.array,
            The input sample.
        """
        prediction_table = {}
        for tree in self.trees:
            prediction = tree.predict(x)
            if prediction in prediction_table:
                prediction_table[prediction] += 1
            else:
                prediction_table[prediction] = 1
        return max(prediction_table)
                
    def fit(self, X: np.array, y: np.array):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        for _ in range(self.n_trees):
            bootstrap_X, bootstrap_y = self._bootstrap(X, y)
            self._build_tree(bootstrap_X, bootstrap_y, self.max_depth)

    def _bootstrap(self, X: np.array, y: np.array):
        """
        Create a bootstrap sample from the data.

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        # bootstrap_indexes = np.random.choice(len(X), len(X))
        # bootstrap_X = np.array([X[i] for i in bootstrap_indexes])
        # bootstrap_y = np.array([y[i] for i in bootstrap_indexes])
    
        bootstrap_indexes = np.random.choice(len(X), len(X))
        bootstrap_X = []
        bootstrap_y = []
        for i in bootstrap_indexes:
            bootstrap_X.append(X[i])
            bootstrap_y.append(y[i])
        bootstrap_X = np.array(bootstrap_X)
        bootstrap_y = np.array(bootstrap_y)
        return bootstrap_X, bootstrap_y

    def _build_tree(self, X: np.array, y: np.array, depth: int):
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
        dc = DecisionTreeClassifier(depth)
        dc.fit(X, y)
        self.trees.append(dc)

class TournamentRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_trees: int, max_depth: int):
        super().__init__(n_trees, max_depth)

    def __repr__(self):
        return f"TournamentRandomForestClassifier(n_trees={self.n_trees}, max_depth={self.max_depth})"

    def _build_tree(self, X: np.array, y: np.array, depth: int):
        dc = TournamentDecisionTreeClassifier(depth)
        dc.fit(X, y)
        self.trees.append(dc)
