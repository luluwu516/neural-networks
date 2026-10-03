import numpy as np

from ._base_network import _BaseNetwork


class SoftmaxRegression(_BaseNetwork):
  def __init__(self, input_size=28*28, num_classes=10):
    super().__init__(input_size, num_classes)
    self._init_params()
    
    
  def _init_params(self):
    """
    Initialize weights with small random values and bias with zeros.
    """
    # np.random.seed(1024)
    # self.params["W"] = 0.01 * np.random.randn(self.input_size, self.num_classes)
    # np.random.seed() reset global random state
    rng = np.random.default_rng(1024)  # local generator
    self.params["W"] = (0.01 * rng.standard_normal((self.input_size, self.num_classes))).astype(np.float32)
    self.params["b"] = np.zeros(self.num_classes, dtype=np.float32)
    
    
  def forward(self, X, y):
    """
    The forward pass of the model
    
    Args:
      X: input data, shape (N, 28, 28).
      y: integer labels, shape (N,).
 
    Returns:
      loss: the mean cross-entropy loss of the batch.
      accuracy: the accuracy of the batch
    """
    # flatten
    X = X.reshape(len(X), -1)
    
    z = X.dot(self.params["W"]) + self.params["b"]
    p = self.softmax(z)

    self.cache["X"] = X
    self.cache["y"] = y
    self.cache["z"] = z
    self.cache["p"] = p

    loss = self.cross_entropy_loss(p, y)
    accuracy = self.compute_accuracy(p, y)
    
    return loss, accuracy
  
  
  def backward(self):
    """
    The backward pass of the model. Uses values cached by forward().
    """
    X = self.cache["X"]
    y = self.cache["y"]
    z = self.cache["z"]
    p = self.cache["p"]
    
    N = X.shape[0]
    
    one_hot = np.zeros_like(p)
    one_hot[np.arange(N), y] = 1
    dz = (p - one_hot) / N
    
    self.gradients["W"] = X.T.dot(dz)
    self.gradients["b"] = np.sum(dz, axis=0)
