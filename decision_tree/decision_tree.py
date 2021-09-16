import numpy as np
from numpy.core.defchararray import split
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Question:
    def __init__(self, index, val):
        self.index = index
        self.val = val
    def testQuestion(self, x):
        return x[self.index] == self.val
class Node:
    def __init__(self, quest, yt, yf):
        self.quest = quest
        self.yt, self.yf = yt, yf
        self.left, self.right = None, None
        self.leaf = False
        self.value = False
class DecisionTree:
    def __init__(self, normalised=False):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.possible = []  # distinct values for each column
        self.valN = []  # number of distinct values at each column
        self.cols = []
        self.nCols = 0 # number of columns
        self.tree = None
        self.rules = []
        self.normalised = normalised
    def printQuest(self, q):
        if q != None:
            return (self.cols[q.index], q.val)
        else:
            return None
    def split(self, x, y, quest):
        Xt, Xf = [], []  # rows true/false
        Yt, Yf = [], []  # index true/false
        for i, row in enumerate(x):
            if quest.testQuestion(row):
                Xt.append(row)
                Yt.append(y[i])
            else:
                Xf.append(row)
                Yf.append(y[i])
        return (Xt, Xf, Yt, Yf)
    def buildTree(self, X, y):
        quest, Xt, Xf, yt, yf = self.bestSplit(X,y)
        this = Node(quest, yt, yf)
        this.leaf = False
        if quest != None:
            this.left = self.buildTree(Xt, yt) # left = true
            this.right = self.buildTree(Xf, yf) # right = false
        else:
            this.leaf = True
            this.value = y[0]

        return this
    def fit(self, X, y):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement
        self.cols = X.columns
        self.nCols = len(self.cols)
        X = np.array(X)
        y = np.array(y)
        
        for i in range(X.shape[1]): 
            uniq = np.unique(X[:, i])
            self.possible.append(list(uniq))  # add possible values
            self.valN.append(len(uniq))  # and how many
        index = np.argmax(self.valN)
        print(index)
        self.tree = self.buildTree(X, y)
    def traverse(self, x, node):
        if node.quest.testQuestion(x):
            node = node.left
        else:
            node = node.right
        if not node.leaf:
            return self.traverse(x, node)
        else:
            return node.value

    def traverse2(self, node, current):
        """
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        myQ = self.printQuest(node.quest) # get question
        nodeL = node.left
        if not nodeL.leaf: # is a node. continue tranverse
            current.append(myQ) # is true therefore add to current
            self.traverse2(nodeL, current) # continue traversing
        else:
            self.rules.append((current+[myQ], nodeL.value)) # went all the way left, aka this is true
         
        nodeR = node.right # go right
        if not nodeR.leaf:
            current = []
            self.traverse2(nodeR, current)
        else:
            if len(current) > 1:
                del current[0]
        

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        X = np.array(X)
        y = []
        for x in X:
            t = self.traverse(x, self.tree)
            y.append(t)
        return np.array(y)
    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        self.traverse2(self.tree, [])
        return self.rules
    def bestSplit(self, X, y):
        best = [None, None, None, None, None]   
        bestVal = 0
        currentEntropy = entropyRows(y)
        X = np.array(X)
        if currentEntropy > 0:
            for i in range(self.nCols):
                counts = getCounts(X[:, i])
                splitEntropy = -np.sum(counts/len(X[:, i])*np.log(counts/len(X[:, i])))
                for z, val in enumerate(self.possible[i]):
                    quest = Question(i, val)
                    Xt, Xf, yt, yf = self.split(X, y, quest)
                    if len(yt) != 0 or len(yf) != 0:
                        ig = infoGain(yt, yf, currentEntropy)
                        if self.normalised:
                            ig /= splitEntropy+0.1
                        if ig> bestVal:
                            best = [quest, Xt, Xf, yt, yf]
                            bestVal = ig
        print(self.printQuest(best[0]))
        return best


# --- Some utility functions
def infoGain(l, r, currentEntropy): # information gain = reduction in entropy
    w1 = float(len(l))/(len(l) + len(r)) # total size
    return currentEntropy-w1*entropyRows(l) - (1-w1)*entropyRows(r)

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()
def entropyRows(x):
    return entropy(getCounts(x))
def getCounts(x):
    return np.unique(x, return_counts=True)[1]
def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))
