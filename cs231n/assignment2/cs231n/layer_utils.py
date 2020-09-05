# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    '''
    Convenience layer that performs affine - batch norm - relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, bn_param: parameters for the batch normalization layer

    Returns a tuple of:
    - relu_out: Output from the ReLU
    - cache: Object to give to the backward pass
    '''
    af_out, af_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(af_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(bn_out)
    cache = (af_cache, bn_cache, relu_cache)
    return relu_out, cache


def affine_bn_relu_backward(dout, cache):
    '''
    Backward pass for affine - batch norm - relu
    
    Inputs:
    - dout: gradient flow from upstream
    - cache: a tuple of cache for affine, batchnorm and relu
    
    Returns a tuple of:
    - dx: gradient of x
    - dw, db: gradient of relu weights
    - dgamma, dbeta: gradient of batchnorm parameters
    '''
    af_cache, bn_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward(drelu, bn_cache)
    dx, dw, db = affine_backward(dbn, af_cache)
    return dx, dw, db, dgamma, dbeta


def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    '''
    Convenience layer that performs affine - layer norm - relu

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, ln_param: parameters for the layer normalization layer

    Returns a tuple of:
    - relu_out: Output from the ReLU
    - cache: Object to give to the backward pass
    '''
    af_out, af_cache = affine_forward(x, w, b)
    ln_out, ln_cache = layernorm_forward(af_out, gamma, beta, ln_param)
    relu_out, relu_cache = relu_forward(ln_out)
    cache = (af_cache, ln_cache, relu_cache)
    return relu_out, cache


def affine_ln_relu_backward(dout, cache):
    '''
    Backward pass for affine - layer norm - relu
    
    Inputs:
    - dout: gradient flow from upstream
    - cache: a tuple of cache for affine, layernorm and relu
    
    Returns a tuple of:
    - dx: gradient of x
    - dw, db: gradient of relu weights
    - dgamma, dbeta: gradient of layernorm parameters
    '''
    af_cache, ln_cache, relu_cache = cache
    drelu = relu_backward(dout, relu_cache)
    dln, dgamma, dbeta = layernorm_backward(drelu, ln_cache)
    dx, dw, db = affine_backward(dln, af_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
