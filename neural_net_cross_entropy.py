#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: Stanford cs231n 2016 winter assignment teaching-assistant, Justin Johnson
# modified by zhaofeng-shu33
# file-description: implementation of a two-layer neural-network with cross-entropy loss
# the first layer uses tanh as activation function
# the second layer uses sigmoid(for binary classification) or sofmax(for multiclass classification) activation function
import numpy as np
import matplotlib.pyplot as plt
import pdb
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad
def report_gradient(net, X, y):
  loss, grads = net.loss(X, y, reg=0.1)

  # these should all be less than 1e-8 or so
  for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros([1,hidden_size])
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros([1,output_size])

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = np.zeros([N,len(b2)])
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    first_layer_output=np.dot(X,W1)+b1
    activated_result=self._tanh(first_layer_output) 
    # the first layer activation function can not be changed, 
    # since the corresponding bp uses it.
    second_layer_output=np.dot(activated_result,W2)+b2
    #softmax_output=self._softmax(second_layer_output)
    scores=second_layer_output
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    if(max(self.params['b2'].shape)>1): # use softmax function as activation
      forward_g=np.exp(scores)
      forward_h=np.sum(forward_g,axis=1)    
      L_loss=-1*scores[list(range(len(y))),y]+np.log(forward_h)
    else: # use sigmoid function as activation
      L_loss = scores.reshape(len(scores)) * (1- y) + np.log(1+np.exp(-scores.reshape(len(scores))))
    cross_entropy_loss = np.average(L_loss)
    L2_regularization_loss = 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))
    loss = cross_entropy_loss + L2_regularization_loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    backward_score=np.zeros(scores.shape)          
    if(max(self.params['b2'].shape)>1):
      backward_score[list(range(len(y))),y]=-1
      backward_score+=forward_g/np.transpose(forward_h+np.zeros([len(b2),1]))
    else:
      backward_score = 1 - y.reshape(len(y),1) - 1/(np.exp(scores) + 1)
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    grads['b2'] = np.average(backward_score,axis=0)
    grads['W2'] = np.dot(activated_result.T,backward_score)/N
    grads['W2'] +=reg*W2 # plus regularization part
    backward_activated_result=np.dot(backward_score,W2.T)
    # backward_first_layer_output=backward_activated_result*(first_layer_output>0) # for relu
    backward_first_layer_output=backward_activated_result / np.power(np.cosh(first_layer_output), 2) # for tanh    
    grads['b1']=np.average(backward_first_layer_output,axis=0)
    grads['W1']=np.dot(X.T,backward_first_layer_output)/N
    grads['W1'] +=reg*W1 # plus regularization part
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, Y,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            verbose=False, record_history = True):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - verbose: boolean; if true print progress during optimization.
    - record_history: boolean; if true return the history object.
    """
    num_train = X.shape[0]

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    
    for it in range(num_iters):
      loss, grads = self.loss(X, Y, reg=reg)
      loss_history.append(loss)

      self.params['W1'] -= learning_rate*grads['W1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['b2'] -= learning_rate*grads['b2']
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if record_history:
        # Check accuracy
        train_acc = (self.predict(X) == Y).mean()
        train_acc_history.append(train_acc)

      # Decay learning rate
      learning_rate *= learning_rate_decay
    if record_history:
      return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
      }
    else:
      return
  
  def _relu(self, X):
    return np.maximum(X, 0)
  def _tanh(self, X):
    return np.tanh(X)
  def _softmax(self, X):
    tmp=np.exp(X)
    return tmp/sum(tmp)
  def _sigmoid(self, X):
    return 1/(1+ np.exp(-X))
  def _forward(self, X):
    """
    Forward propogation.
    """
    first_layer_output=np.dot(X,self.params['W1'])+self.params['b1']
    activated_result=self._tanh(first_layer_output)
    second_layer_output=np.dot(activated_result,self.params['W2'])+self.params['b2']
    output = None
    if((max(self.params['b2'].shape))>1):
      softmax_output = self._softmax(second_layer_output)
      output = softmax_output
    else:
      sigmoid_output = self._sigmoid(second_layer_output)
      output = sigmoid_output
    return output

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    output = self._forward(X)
    if((max(self.params['b2'].shape))>1):
      y_pred = np.argmax(output,axis=1)
    else:
      y_pred = output > 0.5
      y_pred = y_pred.reshape(max(y_pred.shape))
    return y_pred

def int2bin(i):
    if i == 0: return np.array([0,0,0,0,0,0,0,0])
    s = []
    cnt = 0
    while i:
        s.append(i & 1)
        cnt += 1
        i = int(i/2)
    bin_array = np.zeros(8)
    s.reverse()
    bin_array[(8-len(s)):8] = s
    return bin_array

if __name__ == '__main__':
  # for task 1, xor problem
  X = np.array([0,0,1,1,0,1,1,0]).reshape(4,2)
  Y = np.array([0,0,1,1])
  result = 0
  for i in range(10):
    b = TwoLayerNet(2,2,1, std =1.0)
    b.train(X,Y,learning_rate=1e-2, num_iters=8000, reg=0, learning_rate_decay = 1, record_history = False)
    print((b.predict(X) == Y).mean())
    if(abs((b.predict(X) == Y).mean() -1 )< 0.1):
      result += 0.1
  print("Training 10 times from different initial weights, num_iters = 8000, average %f achieves 100 percent\
    correctness", result)
    
  # for task 2, identity function
  X = np.eye(8)
  Y = np.arange(8)
  b = TwoLayerNet(8,3,8, std = 1.0)
  X_eval = np.zeros([256, 8])
  for i in range(256):
      X_eval[i,:] = int2bin(i)
  b.train(X,Y,learning_rate=1e-2, num_iters=8000, reg=0, record_history = False, learning_rate_decay = 1)
  print([b.loss(X,Y)[0], (b.predict(X) == Y).mean()])
  print('identity function error %f' % np.mean(np.abs(X_eval - b._forward(X_eval) )))

  
