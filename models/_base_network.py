import numpy as np

class _BaseNetwork:
  def __init__(self, input_size=28 * 28, num_classes=10):
    self.input_size = input_size
    self.num_classes = num_classes
    
    self.params = dict()
    self.gradients = dict()
    
    self.cache = dict()
    
  def _init_params(self):
    raise NotImplementedError("Subclasses must implement _init_params")
  
  def forward(self, X, y):
    raise NotImplementedError("Subclasses must implement forward")
  
  def backward(self):
    raise NotImplementedError("Subclasses must implement backward")
    
  @staticmethod
  def softmax(scores):
    """
    Convert class scores into probabilities
    
    Args:
      scores: shape (N, C), raw outputs of the model.
 
    Returns:
      probs: shape (N, C), each row sums to 1.
    """
    
    # prevent large scores overflow. Ex: np.exp(1000) = inf, inf / inf = nan ERROR!
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    
    exp_scores = np.exp(shifted)
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # without keepdims, the sum has shape (N,), and dividing (N, C) by (N,) broadcasts along the wrong axis.
    
    return prob
  
  @staticmethod
  def cross_entropy_loss(probs, y):
    """
    Average cross-entropy (CE) loss over a batch.
    
    Args:
      probs: shape (N, C), predicted probabilities (output of softmax).
      y: shape (N,), integer labels.
 
    Returns:
      loss: a scalar, the mean loss over the N examples.
    """
    N = probs.shape[0]
    epsilon = 1e-8
    
    correct = probs[np.arange(N), y]
    loss = np.mean(-np.log(correct + epsilon))
    
    return loss
  
  @staticmethod
  def compute_accuracy(probs, y):
    """
    Compute accuracy
    
    Args:
      probs: shape (N, C), predicted probabilities.
      y: shape (N,), integer labels.
 
    Returns:
      acc: a scalar in [0, 1].
    """
    predictions = np.argmax(probs, axis=1)
    acc = np.mean(predictions == y)
    
    return acc
  
  @staticmethod
  def ReLU(X):
    """
    Rectified Linear Unit: max(0, x), applied element-wise.
    
    Args:
      X: array of any shape (typically hidden-layer pre-activations, (N, H)).
      
    Returns:
      Array of the same shape with negative entries replaced by 0.
    """
    out = np.maximum(0, X)
    
    return out
  
  @staticmethod
  def ReLU_grad(X):
    """
    Derivative of ReLU with respect to its input.
    
    Args:
      X: the same pre-activation input that was passed to ReLU.
 
    Returns:
      Array of the same shape, 1.0 where X > 0 and 0.0 elsewhere.
    """
    # out = np.where(X > 0, 1, 0)
    out = (X > 0).astype(X.dtype)  # keeps everything in X.dtype
    
    return out
  
  @staticmethod
  def sigmoid(X):
    """
    Logistic sigmoid: 1 / (1 + exp(-x)), applied element-wise.
    
    Args:
      X: array of any shape.
 
    Returns:
      Array of the same shape with values in (0, 1).
    """
    out = 1 / (1 + np.exp(-X))
    
    return out
  
  @staticmethod
  def sigmoid_grad(X):
    """
    Derivative of the sigmoid with respect to its input
    
    Args:
      X: the same pre-activation input that was passed to sigmoid.
 
    Returns:
      Array of the same shape with values in (0, 0.25].
    """
    sigmoid_X = _BaseNetwork.sigmoid(X)
    ds = sigmoid_X * (1 - sigmoid_X)
    
    return ds