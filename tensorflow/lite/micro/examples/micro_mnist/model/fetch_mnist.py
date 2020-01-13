
import tensorflow as tf

class MNISTDataset:
  """
  Simple MNIST dataset loading and serving object
  """

  def __init__(self):
    mnist = tf.keras.datasets.mnist
    (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
    self.samples = self.x_train.shape[0]
    self.current_offset = 0

  def get_batch(self, batch_size):

    start = self.current_offset
    end = min(self.current_offset + batch_size, self.samples)

    self.current_offset += batch_size
    if self.current_offset >= self.samples:
      self.current_offset = 0

    return self.x_train[start:end], self.y_train[start:end]
