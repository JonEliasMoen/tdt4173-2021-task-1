import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import decision_tree as dt  # <-- Your implementation

class testDecision(unittest.TestCase):
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
        
        for rules, label in model_1.get_rules():
            conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
            print(f'{"✅" if label == "Yes" else "❌"} {conjunction} => {label}')
        

    def testInit2(self):
        data_2 = pd.read_csv('data_2.csv')
        data_2_train = data_2.query('Split == "train"')
        data_2_valid = data_2.query('Split == "valid"')
        data_2_test = data_2.query('Split == "test"')
        X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
        X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
        X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
        data_2.Split.value_counts()

        model_2 = dt.DecisionTree(normalise=True, thres=0)  # <-- Feel free to add hyperparameters 
        model_2.fit(X_train, y_train)

        print(f'Train: {dt.accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
        print(f'Valid: {dt.accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
        print(f'Test: {dt.accuracy(y_test, model_2.predict(X_test)) * 100 :.1f}%')



if __name__ == '__main__':
    unittest.main()
