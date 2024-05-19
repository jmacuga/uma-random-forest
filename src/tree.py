from typing import Callable
import numpy as np
from utils.tree import entropy_func
from collections import Counter


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

    def __len__(self) -> int:
        return self.group_classes.size

    def __repr__(self) -> str:
        return f"Group(group_classes={self.group_classes})"

    def group_entropy(self) -> float:
        return sum(
            self.entropy_func(Counter(self.group_classes)[class_val], len(self.group_classes))
            for class_val in np.unique(self.group_classes)
        )


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

    def __repr__(self) -> str:
        return f"DecisionTreeClassifier(max_depth={self.max_depth})"

    def fit(self, X: np.array, y: np.array) -> None:
        '''Build a tree from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        '''
        self.tree = self.build_node(X, y, self.depth)

    def predict(self, X: np.array) -> np.array:
        return self.tree.predict(X)

    @staticmethod
    def get_split_entropy(group_a: Group, group_b: Group) -> float:
        return sum(group.entropy * (len(group) / (len(group_a) + len(group_b))) for group in [group_a, group_b])

    def get_split_values(self, feature_values: np.array) -> np.array:
        sorted_feature_values = np.sort(feature_values)
        return [
            np.mean([sorted_feature_values[i], sorted_feature_values[i + 1]])
            for i in range(len(sorted_feature_values) - 1)
        ]

    def get_information_gain(self, parent_group, child_group_a, child_group_b):
        split_entropy = DecisionTreeClassifier.get_split_entropy(child_group_a, child_group_b)
        return parent_group.entropy - split_entropy

    def get_best_feature_split(self, feature_values: np.array, classes: np.array):
        max_information_gain, best_feature_split = 0, None
        split_values = self.get_split_values(feature_values)
        for val in split_values:
            group_a = Group(classes[feature_values <= val])
            group_b = Group(classes[feature_values > val])
            information_gain = self.get_information_gain(Group(classes), group_a, group_b)

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
        if len(np.unique(classes)) == 1 or depth == self.max_depth:
            val = Counter(classes).most_common(1)[0][0]
            return Node(None, None, depth, val=val)
        split_feature, split_val = self.get_best_split(data, classes)
        indeces_a = data[:, split_feature] <= split_val
        indeces_b = data[:, split_feature] > split_val

        if len(indeces_a) == 0 or len(indeces_b) == 0:
            val = Counter(classes).most_common(1)[0][0]
            return Node(None, None, depth, val=val)

        children_nodes = []

        for indeces in [indeces_a, indeces_b]:
            child_data = data[indeces]
            child_classes = classes[indeces]
            children_nodes.append(self.build_node(child_data, child_classes, depth + 1))

        return Node(
            split_feature,
            split_val,
            depth,
            children_nodes[0],
            children_nodes[1],
        )


class ID3DecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth: int):
        super().__init__(max_depth)

    def __repr__(self):
        return f"ID3DecisionTreeClassifier(max_depth={self.max_depth})"


class TournamentDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth: int):
        super().__init__(max_depth)

    def __repr__(self):
        return f"TournamentDecisionTreeClassifier(max_depth={self.max_depth})"
