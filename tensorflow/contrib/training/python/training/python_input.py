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
"""Operations for asynchronously reading data from python into queues.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import script_ops


def _process_yielded_dict(feature_values, keys, features, dtypes, shapes):
  """Read feature_values from the generator and emit a proper output dict."""
  if not isinstance(feature_values, dict):
    raise TypeError("generator must return dict, saw: %s" % feature_values)

  processed_values = {}
  for pk in keys:
    if feature_values.get(pk, None) is not None:
      processed_values[pk] = np.asarray(
          feature_values[pk], dtype=dtypes[pk].as_numpy_dtype)
      check_shape = tensor_shape.TensorShape(processed_values[pk].shape)
      if not shapes[pk].is_compatible_with(check_shape):
        raise ValueError(
            "Feature '%s' has shape %s that is incompatible with declared "
            "shape: %s" % (pk, shapes[pk], check_shape))
      continue
    if isinstance(features[pk], parsing_ops.FixedLenFeature):
      if features[pk].default_value is not None:
        processed_values[pk] = np.asarray(
            features[pk].default_value, dtype=dtypes[pk].as_numpy_dtype)
    elif isinstance(features[pk], parsing_ops.FixedLenSequenceFeature):
      processed_values[pk] = np.empty(
          [0] + features[pk].shape.aslist(), dtype=dtypes[pk].as_numpy_dtype)
    else:
      raise ValueError(
          "Expected generator to return key '%s' with non-empty value" % pk)

  return processed_values


def python_input(generator, features, name=None):
  """Easily feed data from a python generator into TensorFlow queues.

  Example usage:

  ```python
  def generator():
    for i in range(3):
      yield {"value": i}

  features = {
    "value": tf.FixedLenFeature(shape=[], dtype=dtypes.int32)
  }

  tensor_dict = tf.contrib.training.python_input(generator, features)
  batched_dict = tf.train.batch(
    tensor_dict, batch_size=2, allow_smaller_final_batch=True)

  s = tf.Session()
  tf.train.start_queue_runners()

  batch1 = s.run(batched_dict)  # returns {"value": np.array([0, 1])}
  batch2 = s.run(batched_dict)  # returns {"value": np.array([2])}
  s.run(batched_dict)  # error: Queue is closed (generator finished at i==3)
  ```

  Args:
    generator: A python generator that takes no arguments, and yields dicts
      containing a single minibatch entry one at a time.
    features: A python `dict` mapping keys expected from the generator to
      instances of `tf.FixedLenFeature`, or `tf.FixedLenSequenceFeature`.
    name: (Optional) A name for the operations.

  Returns:
    A dict mapping keys of the `features` dict to `Tensor` objects.
    These `Tensor` objects are outputs of a queue that is fed by `generator`.

  Raises:
    TypeError: If generator is not callable or features is not a dict.
    TypeError: If any of features' values are not a Feature object.
    NotImplementedError: If any of features' values are instances of
      `SparseFeature` or `VarLenFeature`  (these are not currently supported).
    ValueError: If any FixedLenSequenceFeatures contain a default value
      (this field is not supported).
    ValueError: if any FixedLenSequenceFeatures have allow_missing=False
      (this field is not supported).
  """
  if not callable(generator):
    raise TypeError("generator must be callable, saw: %s" % generator)
  if not isinstance(features, dict):
    raise TypeError("features must be a dict, saw: %s"
                    % type(features).__name__)

  with ops.name_scope(name, "python_input"):
    shapes = {}
    dtypes = {}
    for k, v in features.items():
      if isinstance(v, parsing_ops.FixedLenFeature):
        if v.default_value is not None:
          value = ops.convert_to_tensor(v.default_value, dtype=v.dtype, name=k)
          shapes[k] = value.shape
          dtypes[k] = value.dtype
        else:
          tensor_shape.TensorShape(v.shape).assert_is_fully_defined()
          shapes[k] = tensor_shape.TensorShape(v.shape)
          dtypes[k] = v.dtype
      elif isinstance(v, parsing_ops.VarLenFeature):
        raise NotImplementedError("VarLenFeature not supported")
      elif isinstance(v, parsing_ops.SparseFeature):
        raise NotImplementedError("SparseFeature not supported")
      elif isinstance(v, parsing_ops.FixedLenSequenceFeature):
        if v.default_value is not None:
          raise ValueError("FixedLenSequenceFeature with default value not "
                           "supported")
        if not v.allow_missing:
          raise ValueError("FixedLenSequenceFeature with allow_missing=False "
                           "not supported")
        tensor_shape.TensorShape(v.shape).assert_is_fully_defined()
        shapes[k] = tensor_shape.TensorShape([None]).concatenate(v.shape)
        dtypes[k] = v.dtype
      else:
        raise TypeError(
            "Expected value for features key '%s' to be one of "
            "FixedLenFeature, VarLenFeature, SparseFeature, or "
            "FixedLenSequenceFeature.  Got: %s" % (k, v))

    keys = list(shapes.keys())
    dtypes_list = [dtypes[pk] for pk in keys]

    counter = [0]
    lock = threading.Lock()
    iterator = iter(generator())

    def generator_iter():
      """Iterate through generator output and return np.arrays to py_func."""
      with lock:
        try:
          feature_values = next(iterator)
          counter[0] += 1
        except StopIteration as e:
          raise StopIteration("Iteration finished.  Processed %d entries (%s)"
                              % (counter[0], e))

      processed_dict = _process_yielded_dict(
          feature_values, keys, features, dtypes, shapes)
      return [processed_dict[pk] for pk in keys]

    generator_pyfunc_values = script_ops.py_func(
        generator_iter, inp=[], Tout=dtypes_list, stateful=True)

    pyfunc_input = {k: v for (k, v) in zip(keys, generator_pyfunc_values)}
    for k, v in shapes.items():
      pyfunc_input[k].set_shape(v)

  return pyfunc_input


__all__ = ["python_input"]
