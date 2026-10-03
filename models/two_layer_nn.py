import numpy as np

from ._base_network import _BaseNetwork


class TwoLayerNeuralNetwork(_BaseNetwork):
  def __init__(self, input_size=28*28, num_classes=10, hidden_size=128):
    super().__init__(input_size, num_classes)
    self.hidden_size = hidden_size
    
    self._init_params()
    
    
  def _init_params(self):
    # (Simpler) Xavier initialization: np.sqrt(1.0 / fan_in), suited to sigmoid
    rng = np.random.default_rng(1024)
    self.params["W1"] = (rng.standard_normal((self.input_size, self.hidden_size)) * np.sqrt(1.0 / self.input_size)).astype(np.float32)
    self.params["W2"] = (rng.standard_normal((self.hidden_size, self.num_classes)) * np.sqrt(1.0 / self.hidden_size)).astype(np.float32)
    
    self.params["b1"] = np.zeros(self.hidden_size, dtype=np.float32)
    self.params["b2"] = np.zeros(self.num_classes, dtype=np.float32)
    
    
  def forward(self, X, y):
    """
    The forward pass of the model
    
    Args:
      X: input data, shape (N, D).
      y: integer labels, shape (N,).
 
    Returns:
      loss: the mean cross-entropy loss of the batch.
      accuracy: the accuracy of the batch.
    """
    # flatten
    X = X.reshape(len(X), -1)
    
    z1 = X.dot(self.params["W1"]) + self.params["b1"]
    a1 = self.sigmoid(z1)
    
    z2 = a1.dot(self.params["W2"]) + self.params["b2"]
    p = self.softmax(z2)
    
    self.cache["X"] = X
    self.cache["y"] = y
    self.cache["z1"] = z1
    self.cache["a1"] = a1
    self.cache["z2"] = z2
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
    z1 = self.cache["z1"]
    a1 = self.cache["a1"]
    z2 = self.cache["z2"]  # never used because there is a shortcut
    p = self.cache["p"]
    
    N = X.shape[0]
    
    one_hot = np.zeros_like(p)
    one_hot[np.arange(N), y] = 1
    dz2 = (p - one_hot) / N
    
    dw2 = a1.T.dot(dz2)
    db2 = np.sum(dz2, axis=0)
    
    da1 = dz2.dot(self.params["W2"].T)
    dz1 = da1 * self.sigmoid_grad(z1)
    
    dw1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0)
    
    self.gradients["W1"] = dw1
    self.gradients["W2"] = dw2
    self.gradients["b1"] = db1
    self.gradients["b2"] = db2