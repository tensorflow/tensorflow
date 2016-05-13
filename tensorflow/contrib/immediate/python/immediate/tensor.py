# Implementation of Tensor for the immediate API

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Tensor"]

class Tensor(object):

  def __init__(self, env, handle):
    self.env = env
    self.handle = handle

  @property
  def tf_handle(self):
    """Give string handle representing this tensor in TF runtime."""
    return self.handle.handle

  @property
  def dtype(self):
    return self.handle._dtype

  def as_numpy(self):
    """Convert current Tensor into numpy array."""

    return self.env.handle_to_numpy(self.handle)

  @staticmethod
  def numpy_to_tensor(env, array):
    handle = env.numpy_to_handle(array)
    return Tensor(env, handle)
  
  def __str__(self):
    return str(self.as_numpy())

  def __repr__(self):
    return "Tensor(%s)" % (self.__str__())


  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types


  def __add__(self, other):
    return self.env.add(self, other)

