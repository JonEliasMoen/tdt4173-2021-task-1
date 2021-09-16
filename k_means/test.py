import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import k_means as km # <-- Your implementation
def euc(x,y):
    return np.linalg.norm(x - y, ord=2, axis=-1)

class testKmeans(unittest.TestCase):
    def testInit(self):
        data_1 = pd.read_csv('data_1.csv')
        X = np.array(data_1[['x0', 'x1']])

        for z in ["First K", "Frogy"]:
            k = 2
            model_1 = km.KMeans(k, z)
            self.assertEqual(model_1.k, k)
            model_1.fit(X)
            self.assertEqual(model_1.centroids.shape, (k,X.shape[1]))
            self.assertEqual(model_1.Xcent.shape[0], X.shape[0])
            self.assertEqual(len(model_1.cent), k)


            a = model_1.predict(X)
            self.assertEqual(np.sum(np.where(a == -1, 1,0)), 0)
            print(z, k, "silhouette", km.euclidean_silhouette(X,a))
            print(z, k , "distortion", km.euclidean_distortion(X,a))
    def testInit2(self):
        data_1 = pd.read_csv('data_2.csv')
        X = np.array(data_1[['x0', 'x1']])

        for z in ["First K", "Frogy"]:
            k = 10
            model_1 = km.KMeans(k, z, preprocessing=True)
            self.assertEqual(model_1.k, k)
            model_1.fit(X)
            self.assertEqual(model_1.centroids.shape, (k,X.shape[1]))
            self.assertEqual(model_1.Xcent.shape[0], X.shape[0])
            self.assertEqual(len(model_1.cent), k)


            a = model_1.predict(X)
            self.assertEqual(a.shape[0], X.shape[0])
            self.assertEqual(np.max(a)+1, k)
            self.assertEqual(np.sum(np.where(a == -1, 1,0)), 0)
            self.assertEqual(euc(np.array([1,1]), np.array([1,2])), 1)
            cents = model_1.centroids
            for i,j in enumerate(a):
                clusterDist = euc(X[i], cents[j])
                for z in range(k):
                    if z != j:
                        otherDist = euc(X[i], cents[z])
                        self.assertTrue(clusterDist < otherDist)


            print(z, k, "silhouette", km.euclidean_silhouette(X,a))
            print(z, k, "distortion ", km.euclidean_distortion(X,a))
    """
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    """

if __name__ == '__main__':
    unittest.main()
