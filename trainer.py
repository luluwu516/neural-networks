import pathlib

import numpy as np
import matplotlib.pyplot as plt

from data.data_processing import generate_batched_data

class Trainer(object):
  def __init__(self, model, optimizer, batch_size=64, epochs=10, seed=16):
    self.model = model
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.epochs = epochs
    self.seed = seed
    self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    
  def train(self, epoch, X, y):
    """
    Run one epoch of mini-batch training. Returns mean loss and accuracy.
    Args:
      epoch: current epoch
      X: input data, shape (N, D).
      y: integer labels, shape (N,).
      
    Returns:
      loss, accuracy
    """
    batches_data, batches_label = generate_batched_data(X, y, batch_size=self.batch_size, shuffle=True, seed=self.seed + epoch)
    losses = []
    accuracies = []
    for X_batch, y_batch in zip(batches_data, batches_label):
      loss, acc = self.model.forward(X_batch, y_batch)
      self.model.backward()
      self.optimizer.update(self.model)
      losses.append(loss)
      accuracies.append(acc)
      
    return np.mean(losses), np.mean(accuracies)

  
  def evaluate(self, X, y):
    """
    Evaluate the model without updating it.
    
    Args:
      X: input data, shape (N, D).
      y: integer labels, shape (N,).
      
    Returns:
      loss, accuracy
    """
    return self.model.forward(X, y)
  
  def fit(self, X_train, y_train, X_val, y_val):
    for epoch in range(self.epochs):
      train_loss, train_acc = self.train(epoch, X_train, y_train)
      val_loss, val_acc = self.evaluate(X_val, y_val)
      
      for key, value in zip(self.history, (train_loss, train_acc, val_loss, val_acc)):
        self.history[key].append(float(value))
      
      print(f"Epoch [{epoch + 1:>2}/{self.epochs:>2}]  "
            f"Train loss : {train_loss:.4f}, Accuracy: {train_acc:.4f}  |  "
            f"Validation loss : {val_loss:.4f}, Accuracy {val_acc:.4f}")
      
    return self.history
  
  def plot_curve(self, plot_type="accuracy", save_path=None):
    """
    Plot training and validation curves.

    Args:
      plot_type: "loss" or "accuracy".
      save_path: optional file path for the figure, e.g. "softmax_loss.png"
    """
    epochs = range(1, self.epochs + 1)

    plt.figure()
    if plot_type == "loss":
      plt.plot(epochs, self.history["train_loss"], label="train")
      plt.plot(epochs, self.history["val_loss"], label="valid")
    else:
      plt.plot(epochs, self.history["train_acc"], label="train")
      plt.plot(epochs, self.history["val_acc"], label="valid")
    
    plt.xlabel("Epoch")
    plt.ylabel(plot_type)
    plt.legend()
    plt.title(f"{plot_type.capitalize()} curve")

    if save_path is not None:
      pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
      plt.savefig(save_path)
    plt.show()
  
  
  