#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: zhaofeng-shu33
import unittest
import numpy as np

import neural_net_cross_entropy
import utility

class TestNeural_Net_Cross_Entropy_8(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.ce8 = neural_net_cross_entropy.TwoLayerNet(8,3,8, std = 1.0)
        self.X8 = np.eye(8) 
        
    def test_forward(self):
        result = self.ce8._forward(self.X8)
        self.assertEqual(result.shape, (8,8))

    def test_predict(self):
        result = self.ce8.predict(self.X8)        
        self.assertTrue((result == np.array([3, 4, 0, 3, 7, 5, 1, 4])).all())

    def test_loss(self):
        np.random.seed(1)
        Y8 = np.random.choice(8, size = 8)
        loss, _ = self.ce8.loss(self.X8, Y8, reg = 0.1)
        self.assertEqual("{0:.4f}".format(loss),'6.2102')

    def test_gradient(self):
        np.random.seed(1)
        Y8 = np.random.choice(8, size = 8)        
        rel_error_total = utility.report_gradient(self.ce8, self.X8, Y8, regularization = 0.1)
        self.assertEqual("{0:.2f}".format(rel_error_total),'0.00')

    def test_train(self):
        pass

class TestNeural_Net_Cross_Entropy_2(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.ce2 = neural_net_cross_entropy.TwoLayerNet(2,2,1, std = 1.0)
        self.X2 = np.array([0,0,1,1,0,1,1,0]).reshape(4,2)
        self.ce2.params['W1'] = np.array([[-0.62, 1.08],[1.17, 0.41]])
        self.ce2.params['W2'] = np.array([[1.22, -1.11]]).T
        
    def test_forward(self):
        result = self.ce2._forward(self.X2)
        self.assertEqual(result.shape, (4,1))

    def test_predict(self):
        result = self.ce2.predict(self.X2)
        self.assertTrue((result == np.array([0, 0, 1, 0], dtype = np.bool)).all())

    def test_loss(self):
        np.random.seed(1)
        Y2 = np.random.choice(2, size = 4)
        loss, _ = self.ce2.loss(self.X2, Y2, reg = 0.01)
        self.assertEqual("{0:.4f}".format(loss),'0.7327')

    def test_gradient(self):
        np.random.seed(1)
        Y2 = np.random.choice(2, size = 4)        
        rel_error_total = utility.report_gradient(self.ce2, self.X2, Y2, regularization = 0.01)
        self.assertEqual("{0:.2f}".format(rel_error_total),'0.00')

    def test_train(self):
        pass

if __name__ == '__main__':
    unittest.main()




