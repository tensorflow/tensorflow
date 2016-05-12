# Implementation of Tensor for the immediate API

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Tensor"]

class Tensor(object):

  def __init__(self, env, array):
    self.env = env
    if env:
      self.handle = env.upload_tensor(array)


  @staticmethod
  def from_numpy(env, array):
    """Upload numpy array into TensorFlow and return corresponding Tensor."""

    return Tensor(env, array)


  def to_numpy(self):
    """Convert current Tensor into numpy array."""

    python_handle = self.handle
    tf_handle = python_handle.handle
    array = self.env.download_tensor(tf_handle)

    
  def __str__(self):
    return str(self.to_numpy())


  def __repr__(self):
    return "Tensor(%s)" % (self.__str__())


  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types


  def __add__(self, other):
    return self.env.add(self, other)

