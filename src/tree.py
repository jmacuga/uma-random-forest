from typing import Callable
import numpy as np


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

    def __len__(self):
        return self.group_classes.size

    def __repr__(self):
        return f"Group(group_classes={self.group_classes})"

    def group_entropy(self):
        # return sum(
        #     entropy_func(
        #         Counter(self.group_classes)[class_val], len(self.group_classes)
        #     )
        #     for class_val in np.unique(self.group_classes)
        # )
        pass


class Node:
    def __init__(
        self,
        split_feature: int,
        split_val: float,
        depth: int = None,
        child_node_a: Node = None,
        child_node_b: Node = None,
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

    def __repr__(self):
        return f"Node(split_feature={self.split_feature}, split_val={self.split_val}, depth={self.depth})"

    def predict(self, data: np.array):
        if self.val is not None:
            return self.val
        if data[self.split_feature] <= self.split_val:
            return self.child_node_a.predict(data)
        else:
            return self.child_node_b.predict(data)


class DecisionTreeClassifier(object):
    def __init__(self, max_depth, entropy_func: Callable[[int, int], float]):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None
        self.entropy_func = entropy_func

    def __repr__(self):
        return f"DecisionTreeClassifier(max_depth={self.max_depth})"

    def fit(self, X: np.array, y: np.array):
        '''Build a tree from the training set (X, y).

        Parameters
        ----------
        X : np.array,
            The training input samples.

        y : np.array,
            The target values.
        """
        '''
        pass

    def predict(self, X: np.array):
        pass

    @staticmethod
    def _get_split_entropy(group_a: Group, group_b: Group):
        pass

    def _get_split_values(self, feature_values: np.array):
        pass

    def _get_best_feature_split(self, feature_values: np.array, classes: np.array):
        pass

    def _get_best_split(self, X: np.array, y: np.array):
        pass
        # return max_feature, best_split_val

    def _build_node(self, data, classes, depth):
        pass


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
