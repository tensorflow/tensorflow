# Implementation of Tensor for the immediate API

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Tensor"]

import tensorflow as tf

class Tensor(object):

  def __init__(self, env, handle):
    self.env = env
    self.handle = handle  # TensorHandle object

  @property
  def tf_handle(self):
    """Give string handle representing this tensor in TF runtime.
    This string handle is suitable for feeding get_session_tensor op."""
    return self.handle.handle

  @property
  def dtype(self):
    return self.handle._dtype

  def as_numpy(self):
    """Convert current Tensor into numpy array."""

    return self.env.handle_to_numpy(self.handle)

  def __str__(self):
    return str(self.as_numpy())

  def __repr__(self):
    return "Tensor(%s)" % (self.__str__())


  # Methods to emulate Python numeric type
  # https://docs.python.org/2/reference/datamodel.html#emulating-numeric-types


  def __add__(self, other):
    return self.env.add(self, other)

  def __bool__(self):
    # TODO(yaroslavvb): add in cast after Python-only ops are supported
    #    bool_tensor = self.env.cast(self, dtype=tf.bool)
    return bool(self.env.tensor_to_numpy(self))

  def __nonzero__(self):
    return self.__bool__()

  def __lt__(self, other):
    return self.env.less(self, other)
  
  def __le__(self, other):
    return self.env.less_equal(self, other)

  def __eq__(self, other):
    return self.env.equal(self, other)

  def __ne__(self, other):
    return self.env.not_equal(self, other)

  def __gt__(self, other):
    return self.env.greater(self, other)

  def __ge__(self, other):
    return self.env.greater_equal(self, other)
