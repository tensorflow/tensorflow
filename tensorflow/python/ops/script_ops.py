# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Script Language Operators. See the @{$python/script_ops} guide."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

# Used by py_util.cc to get tracebacks.
import traceback  # pylint: disable=unused-import
import weakref

import numpy as np
import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


class EagerFunc(object):
  """A wrapper for a function owned by an EagerPyFunc."""

  def __init__(self, func, Tout):
    """Constructs an EagerFunc.

    Args:
      func: The function to wrap.
      Tout: A list of datatypes for the output; an empty list if the output is
            None.
    """
    self._func = func
    self._out_dtypes = Tout

  def _convert(self, value, dtype):
    if isinstance(value, resource_variable_ops.ResourceVariable):
      raise RuntimeError(
          "Attempting to return a variable from an eagerly executed py_func. "
          "Only numeric data structures like Tensors or NumPy arrays should "
          "be returned; to return the value of a variable, make sure to obtain "
          "the Tensor backing it by calling `.read_value()` on the variable in "
          "question: %s" % value)
    return ops.convert_to_tensor(value, dtype=dtype)

  def __call__(self, on_gpu, args):
    """Passes `args` to `self._func`, which is executed eagerly."""
    with context.eager_mode():
      ret = self._func(*args)
      maybe_copy_to_gpu = lambda x: x if not on_gpu else x.gpu()
      if isinstance(ret, (tuple, list)):
        return [
            maybe_copy_to_gpu(self._convert(x, dtype=dtype))
            for (x, dtype) in zip(ret, self._out_dtypes)
        ]
      elif ret is None:
        return ret
      else:
        return maybe_copy_to_gpu(self._convert(ret, dtype=self._out_dtypes[0]))


class FuncRegistry(object):
  """A helper class to keep track of registered py functions.

  FuncRegistry keeps a map from unique tokens (string) to python
  functions, which takes numpy arrays and outputs numpy arrays.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._unique_id = 0  # GUARDED_BY(self._lock)
    # Only store weakrefs to the funtions. The strong reference is stored in
    # the graph.
    self._funcs = weakref.WeakValueDictionary()

  def insert(self, func):
    """Registers `func` and returns a unique token for this entry."""
    token = self._next_unique_token()
    # Store a weakref to the function
    self._funcs[token] = func
    return token

  def remove(self, token):
    """Removes the registered function corresponding to `token`."""
    self._funcs.pop(token, None)

  @staticmethod
  def _convert(value, dtype=None):
    """Converts an arg to numpy, avoiding dangerous string and unicode dtypes.

    Numpy pads with zeros when using string and unicode dtypes if different
    components of a tensor have different lengths.  This is bad: ignoring the
    padding is wrong for text data, and removing the padding is wrong for binary
    data.  To avoid this bug, we redo the conversion using an object dtype.
    Additionally, we convert unicode strings to (byte-)strings for
    compatibility.

    Args:
      value: Value to convert to a numpy array.
      dtype: (Optional.) Desired NumPy type for the returned value.

    Returns:
      A numpy array.
    """
    result = np.asarray(value, dtype=dtype, order="C")
    if result.dtype.char == "S" and result is not value:
      return np.asarray(value, order="C", dtype=object)
    elif result.dtype.char == "U" and result is not value:
      value = np.vectorize(lambda x: x.encode("utf8"))(value)
      return np.asarray(value, order="C", dtype=object)
    elif result.dtype.char == "U":
      return result.astype(np.bytes_)
    else:
      return result

  def __call__(self, token, on_gpu, args):
    """Calls the registered function for `token` with args.

    Args:
      token: A key into this `FuncRegistry` identifying which function to call.
      on_gpu: A boolean indicating whether or not `token`'s corresponding
        operation was placed on GPU; only used if the function registered for
        `token` is an `EagerPyFunc`.
      args: The arguments to pass to the function registered for `token`.

    Returns:
      The output of the function registered for `token`.

    Raises:
      ValueError: if no function is registered for `token`.
    """
    func = self._funcs.get(token, None)
    if func is None:
      raise ValueError("callback %s is not found" % token)
    if isinstance(func, EagerFunc):
      return func(on_gpu, args)
    else:
      ret = func(*args)
      # Strings seem to lead to a memory leak here if they're not wrapped in a
      # list.
      if isinstance(ret, six.binary_type):
        ret = [ret]
      # Ensures that we return either a single numpy array or a list of numpy
      # arrays.
      if isinstance(ret, (tuple, list)):
        return [self._convert(x) for x in ret]
      else:
        return self._convert(ret)

  def size(self):
    """Returns how many functions are currently registered."""
    return len(self._funcs)

  def _next_unique_token(self):
    """Returns a unique token."""
    with self._lock:
      uid = self._unique_id
      self._unique_id += 1
    return "pyfunc_%d" % uid

# Global registry for py functions.
_py_funcs = FuncRegistry()

pywrap_tensorflow.InitializePyTrampoline(_py_funcs)


def _internal_py_func(func, inp, Tout, stateful=None, eager=False, name=None):
  """See documentation for py_func and eager_py_func."""

  is_list_or_tuple = False
  if isinstance(Tout, (list, tuple)):
    is_list_or_tuple = True
  else:
    Tout = [Tout]

  if eager:
    func = EagerFunc(func, Tout)

  token = _py_funcs.insert(func)
  # We tie the registered function's lifetime with the current default graph,
  # i.e., when the current graph is destroyed, we remove its py funcs.
  graph = ops.get_default_graph()

  # pylint: disable=protected-access
  while isinstance(graph, function._FuncGraph):
    # If the py_func was declared inside a _FuncGraph, its lifetime should be
    # bound to that of the outer graph instead.
    graph = graph._outer_graph

  # TODO(zhifengc): Consider adding a Graph method to collect
  # `cleanup` objects in one of its member.
  if not hasattr(graph, "_py_funcs_used_in_graph"):
    graph._py_funcs_used_in_graph = []

  # Store a reference to the function in the graph to ensure it stays alive
  # as long as the graph lives. When the graph is destroyed, the function
  # is left to the garbage collector for destruction as well.
  graph._py_funcs_used_in_graph.append(func)
  # pylint: enable=protected-access

  if eager:
    result = gen_script_ops.eager_py_func(
        input=inp, token=token, Tout=Tout, name=name)
  else:
    if stateful:
      result = gen_script_ops.py_func(
          input=inp, token=token, Tout=Tout, name=name)
    else:
      result = gen_script_ops.py_func_stateless(
          input=inp, token=token, Tout=Tout, name=name)
  return result if is_list_or_tuple else result[0]


def eager_py_func(func, inp, Tout, name=None):
  """Wraps a python function into a TensorFlow op that executes it eagerly.

  This function allows expressing computations in a TensorFlow graph as
  Python functions. In particular, it wraps a Python function `func`
  in a TensorFlow operation that executes it with eager exeuction enabled. As a
  consequence, `tf.contrib.eager.py_func` makes it possible to express control
  flow using Python constructs (`if`, `while`, `for`, etc.), instead of
  TensorFlow control flow constructs (@{tf.cond}, @{tf.while_loop}). For
  example, you might use `tf.contrib.eager.py_func` to implement the log huber
  function:

  ```python
  def log_huber(x, m):
    if tf.abs(x) <= m:
      return x ** 2
    else:
      return m ** 2 * (1 - 2 * tf.log(m) + tf.log(x ** 2))

  x = tf.placeholder(tf.float32)
  m = tf.placeholder(tf.float32)

  y = tf.contrib.eager.py_func(func=log_huber, inp=[x, m], Tout=tf.float32)

  with tf.Session() as sess:
    # The session executes `log_huber` eagerly. Given the feed values below,
    # it will take the second branch, so `output` evaluates to 7.24372.
    output = sess.run(y, feed_dict={x: 3.0, m: 2.0})
  ```

  You can also use `tf.contrib.eager.py_func` to debug your models at runtime
  using Python tools, i.e., you can isolate portions of your code that
  you want to debug, wrap them in Python functions and insert `pdb` tracepoints
  or print statements as desired, and wrap those functions in
  `tf.contrib.eager.py_func`.

  For more information on eager execution, see @{$guide/eager}.

  `tf.contrib.eager.py_func` is similar in spirit to @{tf.py_func}, but unlike
  the latter, the former lets you use TensorFlow operations in the wrapped
  Python function. In particular, while @{tf.py_func} only runs on CPUs and
  wraps functions that take NumPy arrays as inputs and return NumPy arrays as
  outputs, `tf.contrib.eager.py_func` can be placed on GPUs and wraps functions
  that take Tensors as inputs, execute TensorFlow operations in their bodies,
  and return Tensors as outputs.

  `tf.contrib.eager.py_func` is not differentiable, though a gradient may be
  implemented in the future; if you would like to differentiate through it,
  please file an issue on Github.

  Like @{tf.py_func}, `tf.contrib.eager.py_func` has the following limitations
  with respect to serialization and distribution:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.contrib.eager.py_func()`. If you are using distributed
    TensorFlow, you must run a `tf.train.Server` in the same process as the
    program that calls `tf.contrib.eager.py_func()` and you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).


  Args:
    func: A Python function which accepts a list of `Tensor` objects
      having element types that match the corresponding `tf.Tensor` objects
      in `inp` and returns a list of `Tensor` objects (or a single
      `Tensor`, or `None`) having element types that match the
      corresponding values in `Tout`.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns; an empty list
      if no value is returned (i.e., if the return value is `None`).
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes; an empty list
    if `func` returns None.
  """
  return _internal_py_func(func=func, inp=inp, Tout=Tout, eager=True, name=name)


@tf_export("py_func")
def py_func(func, inp, Tout, stateful=True, name=None):
  """Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func`, which takes numpy arrays as its
  arguments and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  input = tf.placeholder(tf.float32)
  y = tf.py_func(my_func, [input], tf.float32)
  ```

  **N.B.** The `tf.py_func()` operation has the following known limitations:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.py_func()`. If you are using distributed TensorFlow, you
    must run a `tf.train.Server` in the same process as the program that calls
    `tf.py_func()` and you must pin the created operation to a device in that
    server (e.g. using `with tf.device():`).

  Args:
    func: A Python function, which accepts `ndarray` objects as arguments and
      returns a list of `ndarray` objects (or a single `ndarray`). This function
      must accept as many arguments as there are tensors in `inp`, and these
      argument types will match the corresponding `tf.Tensor` objects
      in `inp`. The returns `ndarray`s must match the number and types defined
      `Tout`.
      Important Note: Input and output numpy `ndarray`s of `func` are not
      guaranteed to be copies. In some cases their underlying memory will be
      shared with the corresponding TensorFlow tensors.
      In-place modification or storing `func` input or return values in
      python datastructures without explicit (np.)copy
      can have non-deterministic consequences.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) If True, the function should be considered stateful.
      If a function is stateless, when given the same input it will return the
      same output and have no observable side effects. Optimizations such as
      common subexpression elimination are only performed on stateless
      operations.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes.
  """
  if context.executing_eagerly():
    result = func(*[x.numpy() for x in inp])
    result = nest.flatten(result)

    result = [x if x is None else ops.convert_to_tensor(x) for x in result]
    if len(result) == 1:
      # Mimic the automatic unwrapping in graph-mode py_func
      result, = result
    return result

  return _internal_py_func(
      func=func, inp=inp, Tout=Tout, stateful=stateful, eager=False, name=name)


ops.NotDifferentiable("PyFunc")
ops.NotDifferentiable("PyFuncStateless")
