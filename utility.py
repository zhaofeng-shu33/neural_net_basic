#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: Stanford cs231n 2016 winter assignment teaching-assistant, Justin Johnson
# modified by zhaofeng-shu33
import numpy as np
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
  
def report_gradient(net, X, y, regularization):
  _, grads = net.loss(X, y, reg=regularization)
  f = lambda W: net.loss(X, y, reg=regularization)[0]  
  rel_error_total = 0
  for param_name in grads:
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    rel_error_total += rel_error(param_grad_num, grads[param_name])
  return rel_error_total

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

