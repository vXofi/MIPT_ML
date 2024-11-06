import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    probs = np.mean(y, axis=0)
    #unique, counts = np.unique(y, return_counts=True)
    #probabilities_by_class = counts / len(y)
    entr = - np.sum((probs) * np.log(probs + EPS))
    
    return entr
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    #unique, counts = np.unique(y, return_counts=True)
    #probabilities_by_class = counts / len(y)

    probs = np.mean(y, axis=0)
    gin = 1 - np.sum(probs ** 2)
    
    return gin
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    var = np.sum((y - np.mean(y)) ** 2) / y.size
    
    return var

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    med = np.sum(np.abs(y - np.median(y))) / y.size
    
    return med


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold # RENAME
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        left_mask = X_subset[:, feature_index] < threshold

        X_left, y_left = X_subset[left_mask], y_subset[left_mask]
        X_right, y_right = X_subset[~left_mask], y_subset[~left_mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_mask = X_subset[:, feature_index] < threshold

        y_left, y_right = y_subset[left_mask], y_subset[~left_mask]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        feature_index = None
        threshold = None
        minim_value = float('inf')
        total_h = self.criterion(y_subset)

        for index in range(X_subset.shape[1]):
            unique_values = np.unique(X_subset[:, index])

            for td in unique_values:
                y_left, y_right = self.make_split_only_y(index, td, X_subset, y_subset)

                if y_left.size == 0 or y_right.size == 0:
                    continue

                left_size, right_size = y_left.shape[0], y_right.shape[0]
                total_size = left_size + right_size
                crit_left, crit_right = self.criterion(y_left), self.criterion(y_right)

                if (left_size * crit_left + right_size * crit_right) < minim_value:
                    minim_value = (left_size * crit_left + right_size * crit_right)
                    feature_index = index
                    threshold = td

        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        if depth >= self.max_depth or y_subset.shape[0] <= self.min_samples_split:
            if self.classification:
                proba = np.mean(y_subset, axis=0)
                return Node(feature_index=None, threshold=None, proba=proba)
            else:
                value = np.mean(y_subset)
                return Node(feature_index=None, threshold=None, proba=value)

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        if feature_index is None:
            if self.classification:
                proba = np.mean(y_subset, axis=0)
                return Node(feature_index=None, threshold=None, proba=proba)
            else:
                value = np.mean(y_subset)
                return Node(feature_index=None, threshold=None, proba=value)

        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node = Node(feature_index=feature_index, threshold=threshold)
        new_node.left_child = self.make_tree(X_left, y_left, depth + 1)
        new_node.right_child = self.make_tree(X_right, y_right, depth + 1)

        return new_node 
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        def simple_predict(x):
            node = self.root
            while node.left_child or node.right_child:
                if x[node.feature_index] < node.value and node.left_child:
                    node = node.left_child
                elif node.right_child:
                    node = node.right_child
                else:
                    break
            if self.classification:
                return node.proba.argmax()
            return node.proba
        
        y_predicted = np.array([simple_predict(x) for x in X]).reshape(-1, 1)
        
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        def simple_predict(x):
            node = self.root
            while node.left_child or node.right_child:
                if x[node.feature_index] < node.value and node.left_child:
                    node = node.left_child
                elif node.right_child:
                    node = node.right_child
                else:
                    break
            return node.proba
        
        y_predicted = np.array([simple_predict(x) for x in X])
        
        return y_predicted
