# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env"]

from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import Graph

class Env(object):

  def __init__(self):
    self.session = Session()
    self.graph = Graph()
    #    self.op_factory = tf.OpFactory()
    #    self.run_options = tf.RunOptions()


  def __getattr__(self, attr):
    print("getting attr "+attr)


  def download_tensor(self, tensor_handle):
    """Downloads contents of TensorHandle into numpy array.
    
    Returns:
      numpy array containing the tensor data.
    """
    pass


  def upload_tensor(self, nparray):
    """Uploads nparray to TensorFlow runtime.

    Args:
      nparray: numpy array to convert to TensorHandle
    """
    pass
