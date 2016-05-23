# Implementation of Immediate Env
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["Env"]

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import graph_util
from tensorflow.python.client import session
from tensorflow.python.framework import ops

from tensorflow.python.platform import tf_logging as logging

# Native TF ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_candidate_sampling_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import gen_user_ops

# Python-only ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops

from tensorflow.python.framework import ops

import inspect
import re
import os
import types

from .tensor import Tensor
from .tensor import _ENABLE_DEBUG_LOGGING
from .op import OpFactory

from . import module_rewriter

import numpy as np

class Env(object):
  """Env is an object that manages current graph and session and translates
  user commands into appropriate session.run calls

  It contains implementation of methods that translate between Python and
  immediate Tensor representations.
  """


  # TODO(yaroslavvb): use dtypes.as_dtype(dtype) is not None instead
  # to check if type is supported
  # note: np.int32 has different hash from np.dtype('int32'), so must use later
  supported_numpy_types = {np.dtype('int32'), np.dtype('int64'),
                           np.dtype('float32'), np.dtype('float64'),
                           np.dtype('bool'), np.dtype("|S3")}

  def __init__(self, tf_namespace):
    self.original_tf = tf_namespace
    self.sess = session.Session()
    self.op_factory = OpFactory(self)
    symbol_rewriter = module_rewriter.ImmediateRewriter(self)
    rewriter = module_rewriter.ModuleRewriter(symbol_rewriter, "immediate.")
    self.tf = rewriter(self.original_tf)

    self._DEBUG_LOGGING = False

  # TODO: make private?
  @property
  def g(self):
    return self.sess.graph

  # TODO: make private?
  @property
  def graph_version(self):
    """Gives version of the graph. This can be used for checking if graph
    modifications took place"""
    return self.g.version

  # TODO(yaroslavvb): implement graph caching logic for these ops
  def handle_to_numpy(self, tensor_handle):
    """Downloads contents of TensorHandle and returns corresponding numpy array.

    Args:
      tensor_handle: session_ops.TensorHandle object

    Returns:
      numpy array with a copy of data from tensor_handle
    """

    holder, tensor = session_ops.get_session_tensor(tensor_handle._dtype)

    # TODO(yaroslavvb): use session settings for .run call
    array = self.sess.run(tensor, feed_dict={holder: tensor_handle.handle})
    return array


  def numpy_to_handle(self, array):
    """Uploads numpy array to TensorFlow runtime.

    Args:
      array: numpy array to convert to TensorHandle

    Returns:
      TensorHandle corresponding to given numpy array.
    """

    with self.g.as_default():
      holder = array_ops.placeholder(dtype=array.dtype)
      tensor_handle_op = session_ops.get_session_handle(holder)

    tensor_handle = self.sess.run(tensor_handle_op, feed_dict={holder: array})
    return tensor_handle

  # TODO(yaroslavvb): test bad conversions

  def numpy_to_tensor(self, array, dtype=None):
    """Converts numpy.ndarray or compatible type to immediate.Tensor."""

    # convert to numpy dtype if necessary
    if dtype:
      dtype = dtypes.as_dtype(dtype)
      dtype = dtype.as_numpy_dtype

    if isinstance(array, Tensor):
      raise ValueError("Passed immediate.Tensor instead of numpy into "
                       "numpy_to_tensor.")

    # try to convert Python lists to numpy array
    if not isinstance(array, np.ndarray):
      array = np.array(array, dtype=dtype)
      if not array.dtype in self.supported_numpy_types:
        raise ValueError("Unsupported type %s, only support types %s" % (
            repr(array.dtype), [repr(s) for s in self.supported_numpy_types]))

    # Follow downcasting convention as in python/framework/tensor_util.py#L357
    # python/numpy default float type is float64. We prefer float32 instead.
    if (array.dtype == np.float64) and dtype is None:
      array = array.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif (array.dtype == np.int64) and dtype is None:
      downcasted_array = array.astype(np.int32)
      # Do not down cast if it leads to precision loss.
      if np.array_equal(downcasted_array, array):
        array = downcasted_array

    handle = self.numpy_to_handle(array)
    return Tensor(self, handle)

  def constant(self, values, dtype=None, shape=None, name='Const'):
    np_dtype = None

    if dtype:
      try:
        dtype = dtypes.as_dtype(dtype)
        np_dtype = dtype.as_numpy_dtype
      except TypeError as e:
        raise TypeError("Trying to create constant with dtype=%s, "
                        "got TypeError(%s)" % (dtype, e.message))

    if shape:
      assert (isinstance(values, types.FloatType) or
              isinstance(values, types.IntType) or
              isinstance(values, types.LongType))
      return self.numpy_to_tensor(values*np.ones(shape=shape, dtype=np_dtype),
                                  dtype=dtype)
    return self.numpy_to_tensor(values, dtype)

  
  def tensor_to_numpy(self, tensor):
    return self.handle_to_numpy(tensor.handle)
    handle = self.numpy_to_handle(array)
    return Tensor(env, handle)
  
  # TODO(yaroslavvb): remove?
  def get_session_tensor(self, dtype):
    holder, tensor = session_ops.get_session_tensor(dtype)
    return holder, tensor

  def get_session_handle(self, tf_tensor):
    handle_op = session_ops.get_session_handle(tf_tensor)
    return handle_op

  def run(self, *args, **kwargs):
    """Execute session.run in the current Env."""
    return self.sess.run(*args, **kwargs)
