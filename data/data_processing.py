import os
import gzip

import numpy as np
import random


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MNIST")

# Every IDX file begins with a header made of 4-byte big-endian integers, a "magic number":
#  - byte 1-2: always 0x00 0x00
#  - byte 3  : data type of the values
#  - byte 4  : number of dimensions
# Image files: 00 00 08 03 -> 0x00000803 = 2051 (3 dimensions, 8 × 256 + 3 = 2051)
IMAGE_MAGIC = 2051
# Label files: 00 00 08 01 -> 0x00000801 = 2049 (1 dimension, 8 × 256 + 1 = 2049)
LABEL_MAGIC = 2049

def load_mnist_images(filename):
  """
  Load MNIST images from a gzipped IDX file.
  """
  
  with gzip.open(filename, "rb") as f:
  
    # Read the header
    # Images: number of images, rows, columns -> header is 4 * 4 = 16 bytes 
    # Ex: 00 00 08 03 | 00 00 EA 60 | 00 00 00 1C | 00 00 00 1C
    #       2051          60000          28            28
    magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.dtype(">i4"), count=4)
    
    if magic != IMAGE_MAGIC:
      raise ValueError(f"Invalid magic number {magic} in {filename}, expected {IMAGE_MAGIC} for an image file")
    
    images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(num_images, rows, cols)
  
  return images

def load_mnist_labels(filename):
  """
  Load MNIST labels from a gzipped IDX file.
  """
  with gzip.open(filename, "rb") as f:
    
    # Read the header
    # Labels: number of labels -> header is 2 * 4 =  8 bytes
    magic, num_items = np.frombuffer(f.read(8), dtype=np.dtype(">i4"), count=2)
    
    if magic != LABEL_MAGIC:
      raise ValueError(f"Invalid magic number {magic} in {filename}, expected {IMAGE_MAGIC} for an label file")
    
    labels = np.frombuffer(f.read(), dtype=np.uint8)
  
  return labels


def load_data(normalize=True):
  training_data = load_mnist_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
  training_label = load_mnist_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
  test_data = load_mnist_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
  test_label = load_mnist_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
  
  # every image needs exactly one label
  assert len(training_data) == len(training_label), "Training images and labels differ in count"
  assert len(test_data) == len(test_label), "Test images and labels differ in count"
  
  if normalize:
    # use float32 to half the memory
    training_data = training_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0
  
  return training_data, training_label, test_data, test_label


def train_val_split(data, label, train_ratio=0.8):
  """
  Split a dataset into default 80% training and 20% validation sets.
  Run only once during training.
  """
  num_data = data.shape[0]
  split_idx = int(num_data * train_ratio)
  
  train_data = data[:split_idx]
  train_label = label[:split_idx]
  val_data = data[split_idx:]
  val_label = label[split_idx:]
  
  return train_data, train_label, val_data, val_label
  

def generate_batched_data(data, label, batch_size=64, shuffle=False, seed=None):
  """
  Turn raw data into batched forms.
  Run every epoch.
  """
  if seed is not None:
    random.seed(seed)
    np.random.seed(seed)

  num_data = data.shape[0]
  
  if shuffle:
    shuffled_idx = np.arange(num_data)
    np.random.shuffle(shuffled_idx)
    data = data[shuffled_idx]
    label = label[shuffled_idx]
    
  # generate batches of images with the required batch size
  batched_data = []
  batched_label = []
  for start in range(0, num_data, batch_size):
    end = start + batch_size
    batched_data.append(data[start:end])
    batched_label.append(label[start:end])
    
  return batched_data, batched_label


if __name__ == "__main__":
  training_data, training_label, test_data, test_label = load_data()
  print(f"Train Images: {training_data.shape}, Train Labels: {training_label.shape}")
  print(f"Test Images: {test_data.shape}, Test Labels: {test_label.shape}")
  print(f"Pixel dtype: {training_data.dtype}, range: [{training_data.min():.1f}, {training_data.max():.1f}]")
  print(f"Label values: {np.unique(training_label)}")

  print()  
  train_data, train_label, val_data, val_label = train_val_split(training_data, train_label)
  print(f"Train: {train_data.shape}, {train_label.shape}")
  print(f"Val:   {val_data.shape}, {val_label.shape}")
  
  print()
  batch_size = 64
  batches_data, batches_label = generate_batched_data(train_data, train_label, batch_size=batch_size, shuffle=True, seed=16)
  expected_batches = int(np.ceil(len(train_data) / batch_size))
  print(f"Number of batches: {len(batches_data)} (expected {expected_batches})")
  print(f"First batch: X {batches_data[0].shape}, y {batches_label[0].shape}")
  
# Output:
# Train Images: (60000, 28, 28), Train Labels: (60000,)
# Test Images: (10000, 28, 28), Test Labels: (10000,)
# Pixel dtype: float32, range: [0.0, 1.0]      # normalized
# Label values: [0 1 2 3 4 5 6 7 8 9]
# 
# Train: (48000, 28, 28), (48000,)
# Val:   (12000, 28, 28), (12000,)
# 
# Number of batches: 750 (expected 750)
# First batch: X (64, 28, 28), y (64,)