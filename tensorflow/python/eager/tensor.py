# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Experimental API for TensorFlow's "Eager" mode of execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autograd import core as ag_core
import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import tape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import tensor_shape


# TODO(agarwal): rename to TensorHandle.
class Tensor(tf_ops.Tensor):
  """A TensorFlow Eager Tensor."""

  def __init__(self, value, dtype=None):
    """Creates a Tensor object from a Python object or numpy array.

    May share storage with the numpy array, in which case changes to the numpy
    object will reflect
    in the Tensor.

    Arguments:
      value: A numpy.array or a Python object to create a Tensor for.
      dtype: TensorFlow dtype for the returned Tensor. If None, one will be
        automatically selected.
    """
    # TODO(ashankar): Evaluate if we can and perhaps share code with
    # tf.constant defined in
    # https://www.tensorflow.org/code/tensorflow/python/framework/constant_op.py
    self._id = tf_ops.uid()
    if not isinstance(value, np.ndarray):
      npt = None if dtype is None else dtype.as_numpy_dtype
      value = np.array(value, dtype=npt)
      if dtype is None:
        value = _maybe_modify_numpy_dtype_determination(value)
    elif dtype is not None:
      npt = dtype.as_numpy_dtype
      if npt != value.dtype:
        value = value.astype(npt)
    try:
      value = np.asarray(value, order="C")
      self._handle = pywrap_tensorflow.TFE_Py_NumpyToTensorHandle(value)
    except core._NotOkStatusException as e:  # pylint: disable=protected-access
      raise core._status_to_exception(e.code, e.message)  # pylint: disable=protected-access

    # Almost all TensorFlow kernels for GPU devices keep int32 tensors in host
    # memory.  This change approximates the same behavior for eager execution -
    # keeping int32 tensors in host memory.
    #
    # We do so to preclude the need for callers into such kernels from having to
    # explicitly place the int32 tensors in host memory. For example, prior to
    # this change one needed:
    #
    # with tfe.device('/gpu:0'):
    #   ...  # code here
    #   with tfe.device('/cpu:0'):
    #     shape = tfe.Tensor(...)
    #   y = tfe.ops.random_uniform(.., shape)
    #
    # Without the CPU device block tfe.ops.random_uniform would fail since the
    # kernel expects the shape in host memory.
    #
    # After this change, we simplify the code:
    #
    # with tfe.device('/gpu:0'):
    #   y = tfe.ops.random_uniform(, tfe.Tensor(...))
    #
    # The approximation is not exact since if there are GPU kernels which do not
    # require host memory for int32 tensors, there will be a discrepancy between
    # eager execution and TensorFlow graphs. However, as of July 2017, there
    # were no known GPU kernels that kept int32 tensors in device memory.
    if _in_gpu_device() and value.dtype != np.int32:
      ctx = context.get_default_context()
      # pylint: disable=protected-access
      device_name = ctx.device_name
      with errors.raise_exception_on_not_ok_status() as status:
        self._handle = pywrap_tensorflow.TFE_TensorHandleCopyToDevice(
            self._handle, ctx._handle, device_name, status)
      # pylint: enable=protected-access

    self._dtype = dtypes.as_dtype(
        pywrap_tensorflow.TFE_TensorHandleDataType(self._handle))

    # This mirrors tensorflow.core.framework.ops.Tensor._handle_data Which will
    # be None for tensors of type other than DT_REOSURCE. For DT_RESOURCE
    # tensors, this will contain a serialized HandleData proto with shape
    # inference metadata about shapes and dtypes of resources accessible from
    # this handle.
    self._handle_data = None
    if core.active_trace() is not None:
      core.active_trace().record_tensor("MANUAL",
                                        tape.tensor_id(self),
                                        self.device,
                                        self.shape.num_elements())

  def __del__(self):
    if (pywrap_tensorflow is not None
        and pywrap_tensorflow.TFE_DeleteTensorHandle is not None):
      pywrap_tensorflow.TFE_DeleteTensorHandle(self._handle)
    if core.active_trace() is not None:
      core.active_trace().delete_tensor(tape.tensor_id(self))

  def _numpy_text(self, is_repr=False):
    if self.dtype.is_numpy_compatible:
      numpy_text = repr(self.numpy()) if is_repr else str(self.numpy())
    else:
      numpy_text = "<unprintable>"
    if "\n" in numpy_text:
      numpy_text = "\n" + numpy_text
    return numpy_text

  def __str__(self):
    return "tfe.Tensor(shape=%s, dtype=%s, numpy=%s)" % (
        self.shape, self.dtype.name, self._numpy_text())

  def __repr__(self):
    return "<tfe.Tensor: id=%s, shape=%s, dtype=%s, numpy=%s)>" % (
        self._id, self.shape, self.dtype.name, self._numpy_text(is_repr=True))

  @staticmethod
  def _override_operator(name, func):
    setattr(Tensor, name, func)

  def numpy(self):
    """Returns a numpy array with the same contents as the Tensor.

    The contents of the Tensor must be backed by host memory. The
    as_cpu_tensor() method can be used ensure that this is true.

    TODO(ashankar,agarwal): Perhaps this should NOT reference the underlying
    buffer but instead always explicitly copy? Note that currently it may or may
    not copy based on whether the numpy data is properly aligned or not.

    Returns:
      A numpy array that may share memory with the Tensor object. Any changes
      to one may be reflected in the other.
    """
    # TODO(ashankar): This with status business seems expensive. Profile/avoid?
    cpu = self.as_cpu_tensor()
    with errors.raise_exception_on_not_ok_status() as status:
      return pywrap_tensorflow.TFE_Py_TensorHandleToNumpy(cpu._handle, status)  # pylint: disable=protected-access

  def _copy(self, ctx, device_name):
    """Copies tensor to dest device."""
    # pylint: disable=protected-access
    # Creates a new tensor on the dest device.
    with errors.raise_exception_on_not_ok_status() as status:
      h = pywrap_tensorflow.TFE_TensorHandleCopyToDevice(
          self._handle, ctx._handle, device_name, status)
    new_tensor = _tensor_from_handle(h)
    if core.active_trace() is not None:
      core.active_trace().record_tensor("COPY",
                                        tape.tensor_id(new_tensor),
                                        new_tensor.device,
                                        new_tensor.shape.num_elements())
    return new_tensor
    # pylint: enable=protected-access

  @property
  def device(self):
    return pywrap_tensorflow.TFE_TensorHandleDeviceName(self._handle)

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    """The shape of this Tensor as a TensorShape object."""
    n = pywrap_tensorflow.TFE_TensorHandleNumDims(self._handle)
    # As of May 2017, TFE_TensorHandle objects were always backed by concrete
    # tensors (which have a valid, known shape).  There were vague plans to
    # change this so that the Tensor class can also represent Tensors that have
    # not yet been computed.
    # If that happens, handle that (e.g., if n < 0: return tensor_shape(None))
    # and also handle -1s returned by TFE_TensorHandleDim.
    assert n >= 0, "See comment in source code"
    return tensor_shape.TensorShape(
        [pywrap_tensorflow.TFE_TensorHandleDim(self._handle, x)
         for x in range(n)])

  def get_shape(self):
    """Alias of Tensor.shape."""
    return self.shape

  def _shape_tuple(self):
    """The shape of this Tensor, as a tuple.

    This is more performant than tuple(shape().as_list()) as it avoids
    two list and one object creation. Marked private for now as from an API
    perspective, it would be better to have a single performant way of
    getting a shape rather than exposing shape() and shape_tuple()
    (and heaven forbid, shape_list() etc. as well!). Punting on that for now,
    but ideally one would work things out and remove the need for this method.
    """
    n = pywrap_tensorflow.TFE_TensorHandleNumDims(self._handle)
    # As of May 2017, TFE_TensorHandle objects were always backed by concrete
    # tensors (which have a valid, known shape).  There were vague plans to
    # change this so that the Tensor class can also represent Tensors that have
    # not yet been computed.
    # If that happens, handle that (e.g., if n < 0: return tensor_shape(None))
    # and also handle -1s returned by TFE_TensorHandleDim.
    assert n >= 0, "See comment in source code"
    return tuple(
        pywrap_tensorflow.TFE_TensorHandleDim(self._handle, x)
        for x in range(n))

  def _shape_as_list(self):
    """The shape of the tensor as a list."""
    return list(self._shape_tuple())

  def as_cpu_tensor(self):
    """A copy of this Tensor with contents backed by host memory."""
    return self._copy(context.get_default_context(), "CPU:0")

  def as_gpu_tensor(self, gpu_index=0):
    """A copy of this Tensor with contents backed by memory on the GPU.

    Arguments:
      gpu_index: Identifies which GPU to place the contents on the returned
        Tensor in.

    Returns:
      A GPU-memory backed Tensor object initialized with the same contents
      as this Tensor.
    """
    return self._copy(context.get_default_context(), "GPU:" + str(gpu_index))

  def __bool__(self):
    if self._shape_tuple() != ():  # pylint: disable=g-explicit-bool-comparison
      raise ValueError(
          "Non-scalar tensor %s cannot be converted to boolean." % repr(self))
    if self.dtype != dtypes.bool:
      raise ValueError(
          "Non-boolean tensor %s cannot be converted to boolean." % repr(self))
    return bool(self.as_cpu_tensor().numpy())

  def __nonzero__(self):
    return self.__bool__()

  # Methods not supported / implemented for Eager Tensors.
  @property
  def op(self):
    raise NotImplementedError("op not supported for Eager Tensors.")

  @property
  def graph(self):
    raise NotImplementedError("graph not supported for Eager Tensors.")

  @property
  def name(self):
    raise NotImplementedError("name not supported for Eager Tensors.")

  def set_shape(self, shape):
    raise NotImplementedError("set_shape not supported for Eager Tensors.")

  @property
  def value_index(self):
    raise NotImplementedError("value_index not supported for Eager Tensors.")

  def consumers(self):
    raise NotImplementedError("consumers not supported for Eager Tensors.")

  def _add_consumer(self, consumer):
    raise NotImplementedError("_add_consumer not supported for Eager Tensors.")

  def _as_node_def_input(self):
    raise NotImplementedError(
        "_as_node_def_input not supported for Eager Tensors.")

  def _as_tf_output(self):
    raise NotImplementedError("_as_tf_output not supported for Eager Tensors.")

  def eval(self, feed_dict=None, session=None):
    raise NotImplementedError("eval not supported for Eager Tensors.")


class IndexedSlices(object):
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. @{tf.gather}).
  """

  def __init__(self, values, indices, dense_shape):
    """Creates an `IndexedSlices`."""
    self._values = values
    self._indices = indices
    assert indices.shape[0] == values.shape[0]
    self._dense_shape = dense_shape

  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape


class _Op(object):
  """Fake op for _LazyZero to make its python API tf.Tensor-like."""

  def __init__(self):
    self.type = "Zeros"


class LazyZero(object):
  """Lazily-instantiated zero-valued Tensor used as autograd accumulator."""

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype
    self.op = _Op()

  def __add__(self, other):
    return other

  def __radd__(self, other):
    return other

  def numpy(self):
    return np.zeros(self.shape, self.dtype)


def convert_to_eager_tensor(t, dtype=None):
  if isinstance(ag_core.getval(t), Tensor):
    if dtype is not None and t.dtype != dtype:
      raise TypeError("Expected tensor with type %r not %r" % (dtype, t.dtype))
    return t
  return Tensor(t, dtype=dtype)


def convert_n_to_eager_tensor(values, dtype):
  return [convert_to_eager_tensor(t, dtype) for t in values]


def _tensor_from_handle(handle):
  """'Private' constructor for the Tensor object.

  The existence of a 'handle' is an implementation detail that should be hidden
  from users of this module.  Functions within this module do need to create a
  Tensor object from a handle though.

  One option would be to have an __init__(self, handle) method on the
  Tensor class, but that would make the existence and use of a handle
  'public'.

  Instead, this function avoids exposing a Tensor.__init__ that understands
  handles and yet allows functions within this module to create Tensor
  objects from a handle.

  Arguments:
    handle: A valid TFE_TensorHandle object.

  Returns:
    A Tensor object.
  """
  # pylint: disable=protected-access
  t = Tensor.__new__(Tensor)
  t._id = tf_ops.uid()
  t._handle = handle
  t._dtype = dtypes.as_dtype(pywrap_tensorflow.TFE_TensorHandleDataType(handle))
  t._handle_data = None
  return t
  # pylint: enable=protected-access


# TODO(ashankar): use actual device type.
def _in_gpu_device():
  return context.get_default_context()._device_index > 0  # pylint: disable=protected-access


def _maybe_modify_numpy_dtype_determination(np_array):
  """Tweak numpy dtype determination.

  numpy prefers int64 and float64, we prefer int32 and float32.
  (int32 is often used as the "shape" input to various operations,
  many of which only support int32 shapes).
  This preference is copied from tensor_util.make_tensor_proto
  (https://goto.google.com/numpy_prefs_156503903)

  Args:
    np_array: A numpy ndarray
  Returns:
    A numpy ndarray whose dtype may have been modified.
  """
  if np_array.dtype == np.float64:
    return np_array.astype(np.float32)
  if np_array.dtype == np.int64:
    # Downcast iff there is no precision loss.
    downcasted = np_array.astype(np.int32)
    if np.array_equal(downcasted, np_array):
      return downcasted
  return np_array
