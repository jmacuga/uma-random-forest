import numpy as np
from tree import RandomizedDecisionTreeClassifier, RandomizedTournamentDecisionTreeClassifier
import logging

logging.basicConfig(level=logging.INFO)


class RandomForestClassifier:
    def __init__(self, n_trees: int, max_depth: int, max_features: int = None, max_split_values: int = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.max_features = max_features
        self.max_split_values = max_split_values
        logging.info(f"RandomForestClassifier: n_trees={n_trees}, max_depth={max_depth}")

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
        return max(prediction_table, key=prediction_table.get)

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
        dc = RandomizedDecisionTreeClassifier(
            depth, max_features=self.max_features, max_split_values=self.max_split_values
        )
        dc.fit(X, y)
        self.trees.append(dc)


class TournamentRandomForestClassifier(RandomForestClassifier):
    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        tournament_size: int = 2,
        max_features: int = None,
    ):
        super().__init__(n_trees, max_depth, max_features)
        self.tournament_size = tournament_size
        logging.info(
            f"TournamentRandomForestClassifier: n_trees={n_trees}, max_depth={max_depth}, tournament_size={tournament_size}"
        )

    def __repr__(self):
        return f"TournamentRandomForestClassifier(n_trees={self.n_trees}, max_depth={self.max_depth})"

    def _build_tree(self, X: np.array, y: np.array, depth: int):
        dc = RandomizedTournamentDecisionTreeClassifier(
            depth,
            max_features=self.max_features,
            tournament_size=self.tournament_size,
        )
        dc.fit(X, y)
        self.trees.append(dc)
