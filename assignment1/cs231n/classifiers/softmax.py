import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_dim = W.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    s = X[i].dot(W)
    s -= s.max()
    exp_sum = np.sum(np.exp(s))
    correct_class_prob = np.exp(s[y[i]]) / exp_sum
    #print(correct_class_prob)
    loss += -np.log(correct_class_prob)
    for j in range(num_classes):
        if j == y[i]:
            dW[:,j] += X[i]*(correct_class_prob-1)
        else:
            dW[:,j] += X[i]*np.exp(s[j])/exp_sum
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_dim = W.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  s = X.dot(W)
  s_exp = np.exp(s)
  correct_s_exp = s_exp[range(num_train),y]
  s_exp_sum = np.sum(s_exp, axis = 1)
  mask = -np.log(correct_s_exp/s_exp_sum)
  loss = np.sum(mask)
  loss /= num_train
  loss += reg * np.sum(W*W)
  s_exp[range(num_train),y] -= s_exp_sum
  dW = X.T.dot(s_exp/np.reshape(s_exp_sum,(num_train,1)))
  dW /= num_train
  dW += 2*reg*W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

