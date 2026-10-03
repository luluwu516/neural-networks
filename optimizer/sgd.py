import numpy as np

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
  def __init__(self, learning_rate=1e-2, reg=1e-5):
    super().__init__(learning_rate, reg)
    
    
  def update(self, model):
    """
    Update model weights based on gradients
    
    Args:
      model: the model with gradients.
    """
    # L2 regularization
    self.apply_L2_regularization(model)
    
    for key in model.params.keys():
      model.params[key] -= (self.learning_rate * model.gradients[key])