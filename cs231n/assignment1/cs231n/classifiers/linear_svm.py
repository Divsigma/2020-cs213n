from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes): 
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]*1.0
                dW[:, y[i]] -= X[i]*1.0

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    '''
    # verbose version:
    # try exchanging the loop of k and j !
    for i in range(num_train):
        scores = X[i].dot(W)
        for k in range(num_classes):
            if k == y[i]:
                for j in range(num_classes):
                    if j == y[i]:
                        continue
                    margin = scores[j] - scores[y[i]] + 1
                    if margin > 0:
                        dW[:, k] -= X[i]*1.0
            else:
                margin = scores[k] - scores[y[i]] + 1
                if margin > 0:
                    dW[:, k] += X[i]*1.0
                
    dW = dW / num_train + 2*reg*W
    '''
    
    dW /= num_train
    dW += 2*reg*W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    
    scores = X.dot(W)
    margin = np.maximum(0, scores.T - scores[range(num_train), y] + 1).T # note delta = 1
    margin[range(num_train), y] = 0
    data_loss = np.sum(margin) * 1.0 / num_train
    reg_loss = reg*np.sum(np.square(W))
    
    loss = data_loss + reg_loss
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    X_effect = (margin > 0).astype('float')                       # 每个样本i在非y[i]的类上产生X[i]的梯度
    X_effect[range(num_train), y] -= np.sum(X_effect, axis=1)   # 每个样本i在y[i]的类上产生sigma(margin gt 0)*X[i]（除y[i]的margin）的梯度
    
    dW = X.T.dot(X_effect)
    dW /= num_train 
    dW += 2*reg*W
    
    
    ''' verbose version: 
    margin_chara = (margin > 0).astype('float')
    margin_chara_sum = np.sum(margin_chara, axis=1).astype('float')
    
    for i in range(num_train):
        dW += (margin_chara[i][:, np.newaxis]*X[i]).T  # broadcast
        # dW[:, y[i]] -= margin_chara[i, y[i]]*X[i]  # margin_chara[i, y[i]] == 0 is always 
        dW[:, y[i]] -= margin_chara_sum[i]*X[i]
    
    dW /= num_train
    dW += 2*reg*W
    '''
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
