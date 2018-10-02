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

"""Script Language Operators.

@@py_func
"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.script_ops import py_func as _py_func
from tensorflow.python.util import nest

__all__ = ['py_func']


def py_func(func,
            args=(),
            kwargs=None,
            output_types=None,
            output_shapes=None,
            stateful=True,
            name=None):
  """Wraps a python function and uses it as a TensorFlow op.

  This function is a wrapper around `tf.py_func` and improve it with kwargs
  and output_shapes. Further it changed some argument names.

  Given a python function `func`, which takes numpy arrays as its
  inputs and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  inp = tf.placeholder(tf.float32)
  y = tf.py_func(my_func, [inp], tf.float32)
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
    func: A Python function, which accepts a list of NumPy `ndarray` objects
      having element types that match the corresponding `tf.Tensor` objects
      in `inp`, and returns a list of `ndarray` objects (or a single `ndarray`)
      having element types that match the corresponding values in `Tout`.
    args: A list of `Tensor` objects.
    kwargs: A dict with `Tensor` objects as values.
    output_types: A nested structure of tensorflow data types or a single
      tensorflow data type if there is only one, indicating what `func` returns.
    output_shapes: Same as output_types, except the types are replaces with
      shapes (optional).
    stateful: (Boolean.) If True, the function should be considered stateful.
      If a function is stateless, when given the same input it will return the
      same output and have no observable side effects. Optimizations such as
      common subexpression elimination are only performed on stateless
      operations.
    name: A name for the operation (optional).

  Returns:
    Tensorflow op that wraps the input python function.
  """

  if kwargs is None:
    kwargs = {}

  if not isinstance(args, (list, tuple)):
    raise TypeError('args must be list and not {}. args: {}'.format(
        type(args), args))

  if not isinstance(kwargs, dict):
    raise TypeError('kwargs must be dict and not {}. args: {}'.format(
        type(kwargs), kwargs))

  # For dynamic type inference use callable output_types and output_shapes
  if callable(output_types):
    # If callable assume same signature and call with tensors and get the types
    output_types = output_types(*args, **kwargs)
  if callable(output_shapes):
    # If callable assume same signature and call with tensors and get the shapes
    output_shapes = output_shapes(*args, **kwargs)

  flat_output_types = nest.flatten(output_types)
  args = (args, kwargs)
  flat_args = nest.flatten(args)

  def python_function_wrapper(*py_args):
    py_args, py_kwargs = nest.pack_sequence_as(args, py_args)

    ret = func(*py_args, **py_kwargs)
    # TODO(alextp): Catch Exceptions and improve msg, because tensorflow
    # ist not able to preserve the traceback, i.e. the Exceptions does not
    # contain any information where the Exception was raised.
    nest.assert_shallow_structure(output_types, ret)
    return nest.flatten(ret)

  flat_values = _py_func(
      python_function_wrapper,
      flat_args,
      flat_output_types,
      stateful=stateful,
      name=name)

  if output_shapes is not None:
    # I am not sure if this is nessesary
    output_shapes = nest.map_structure_up_to(
        output_types, tensor_shape.as_shape, output_shapes)

    flattened_shapes = nest.flatten(output_shapes)
    for ret_t, shape in zip(flat_values, flattened_shapes):
      ret_t.set_shape(shape)

  return nest.pack_sequence_as(output_types, flat_values)
