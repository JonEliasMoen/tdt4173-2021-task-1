import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import decision_tree as dt  # <-- Your implementation

class testKmeans(unittest.TestCase):
    def testInit(self):
        data_1 = pd.read_csv('data_1.csv')
        # Separate independent (X) and dependent (y) variables
        X = data_1.drop(columns=['Play Tennis'])
        y = data_1['Play Tennis']

        # Create and fit a Decrision Tree classifier
        model_1 = dt.DecisionTree()  # <-- Should work with default constructor
        model_1.fit(X, y)

        # Verify that it perfectly fits the training set
        print(f'Accuracy: {dt.accuracy(y_true=y, y_pred=model_1.predict(X)) * 100 :.1f}%')
   

if __name__ == '__main__':
    unittest.main()
