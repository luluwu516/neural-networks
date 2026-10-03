class _BaseOptimizer:
  def __init__(self, learning_rate=1e-2, reg=1e-5):
    self.learning_rate = learning_rate
    self.reg = reg
    
  def update(self, model):
    raise NotImplementedError("Subclasses must implement update")
  
  def apply_L2_regularization(self, model):
    """
    Apply L2 penalty to the model. Update the gradient dictionary in the model
    
    Args:
      model: the model with gradients.
    """
    weights = [key for key in model.params if key.startswith("W")]
    
    for w in weights:
      model.gradients[w] += self.reg * model.params[w]