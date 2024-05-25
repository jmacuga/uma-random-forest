import numpy as np
from utils.tree import entropy_func
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)


class Group:
    def __init__(self, group_classes: np.array):
        """Class representing group of classes in split

        Parameters
        ----------
        group_classes : np.array
            Array of classes in the group
        """
        self.group_classes = group_classes
        self.entropy = self.group_entropy()
        # logging.info(self.__repr__() + " created")

    def __len__(self) -> int:
        return self.group_classes.size

    def __repr__(self) -> str:
        return f"Group(group_classes={self.group_classes}, entropy={self.entropy}"

    def group_entropy(self) -> float:
        class_counts = Counter(self.group_classes)
        total_count = len(self.group_classes)
        entropy = sum(entropy_func(count, total_count) for count in class_counts.values())
        return entropy


class Node:
    def __init__(
        self,
        split_feature: int,
        split_val: float,
        depth: int = None,
        child_node_a: "Node" = None,
        child_node_b: "Node" = None,
        val: float = None,
    ):
        """Class representing singl enode in a tree

        Parameters
        ----------
        split_feature : int
            Index of feature to split on
        split_val : float
            Value to split on
        depth : int or None
            Depth of the node in the tree
        child_node_a : Node or None
            Node representing left child
        child_node_b : Node or None
            Node representing right child
        val : float or None
            Value of the node if it is a leaf
        """
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val
        logging.info(self.__repr__() + " created")

    def __repr__(self) -> str:
        return f"Node(split_feature={self.split_feature}, split_val={self.split_val}, depth={self.depth})"

    def predict(self, data: np.array):
        if self.val is not None:
            return self.val
        if data[self.split_feature] <= self.split_val:
            return self.child_node_a.predict(data)
        else:
            return self.child_node_b.predict(data)


class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None
        logging.info(self.__repr__() + " created")

    def __repr__(self) -> str:
        return f"DecisionTreeClassifier(max_depth={self.max_depth})"

    def fit(self, X: np.array, y: np.array) -> None:
        """Build a tree from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.

        """
        self.tree = self.build_node(X, y, self.depth)
        logging.debug(self.tree)

    def predict(self, X: np.array) -> np.array:
        return self.tree.predict(X)

    @staticmethod
    def get_split_entropy(group_a: Group, group_b: Group) -> float:
        total_count = len(group_a) + len(group_b)
        entropy_a = group_a.entropy * (len(group_a) / total_count)
        entropy_b = group_b.entropy * (len(group_b) / total_count)
        return entropy_a + entropy_b

    @staticmethod
    def get_split_values(feature_values: np.array) -> np.array:
        if feature_values.size == 0:
            raise ValueError("Feature values are empty")

        logging.debug(f"Feature values:")
        sorted_feature_values = np.sort(np.unique(feature_values))
        if sorted_feature_values.size == 1:
            return sorted_feature_values
        return [
            np.mean([sorted_feature_values[i], sorted_feature_values[i + 1]])
            for i in range(len(sorted_feature_values) - 1)
        ]

    @staticmethod
    def get_information_gain(parent_group, child_group_a, child_group_b):
        split_entropy = DecisionTreeClassifier.get_split_entropy(child_group_a, child_group_b)
        return parent_group.entropy - split_entropy

    def get_best_feature_split(self, feature_values: np.array, classes: np.array):
        max_information_gain, best_feature_split = 0, None
        split_values = DecisionTreeClassifier.get_split_values(feature_values)
        logging.debug(f"Split values: {split_values}")
        group_classes = Group(classes)

        if len(split_values) == 1:
            return split_values[0], 0

        for i in range(len(split_values) - 1):
            val = (split_values[i] + split_values[i + 1]) / 2
            group_a = Group(classes[feature_values <= val])
            group_b = Group(classes[feature_values > val])
            information_gain = DecisionTreeClassifier.get_information_gain(group_classes, group_a, group_b)

            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature_split = val
        return best_feature_split, max_information_gain

    def get_best_split(self, X: np.array, y: np.array) -> tuple[int, float]:
        max_information_gain, max_feature, best_split_val = 0, None, None
        for feature in range(X.shape[1]):
            split_val, information_gain = self.get_best_feature_split(X[:, feature], y)

            if information_gain > max_information_gain:
                max_information_gain = information_gain
                max_feature = feature
                best_split_val = split_val
        return max_feature, best_split_val

    def build_node(self, data, classes, depth):
        if data.size == 0 or classes.size == 0:
            raise ValueError("Data is empty")

        if len(np.unique(classes)) == 1 or depth == self.max_depth:
            logging.debug(f"Most common {Counter(classes).most_common(1)}")
            val = Counter(classes).most_common(1)[0][0]
            logging.debug("Leaf - returning: " + str(val))
            return Node(None, None, depth, val=val)

        split_feature, split_val = self.get_best_split(data, classes)
        indeces_a = data[:, split_feature] <= split_val
        indeces_b = data[:, split_feature] > split_val

        if len(indeces_a) == 0 or len(indeces_b) == 0:
            val = Counter(classes).most_common(1)[0][0]
            logging.debug("One group empty -> Leaf - returning: " + str(val))
            return Node(None, None, depth, val=val)

        children_nodes = []

        for indeces in [indeces_a, indeces_b]:
            child_data = data[indeces]
            child_classes = classes[indeces]
            logging.debug("Child classes: ")
            children_nodes.append(self.build_node(child_data, child_classes, depth + 1))

        return Node(
            split_feature,
            split_val,
            depth,
            children_nodes[0],
            children_nodes[1],
        )


class RandomizedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth: int, max_features: int = None):
        super().__init__(max_depth=max_depth)
        self.max_features = max_features

    def get_best_split(self, X: np.array, y: np.array) -> tuple[int, float]:
        self.feature_indices_ = np.random.choice(X.shape[1], self.max_features, replace=False)
        logging.debug(f"Feature indeces: {self.feature_indices_}")
        X_subset = X[:, self.feature_indices_]
        logging.debug(f"X_subset: {X_subset}")
        split_feature, split_val = super().get_best_split(X_subset, y)
        return self.feature_indices_[split_feature], split_val


class TournamentDecisionTreeClassifier(RandomizedDecisionTreeClassifier):
    def __init__(self, max_depth: int):
        super().__init__(max_depth)

    def __repr__(self):
        return f"TournamentDecisionTreeClassifier(max_depth={self.max_depth})"
