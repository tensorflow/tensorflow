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
"""The Counter Dataset."""
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.Counter", v1=[])
def CounterV2(start=0, step=1, dtype=dtypes.int64):
  """Creates a `Dataset` that counts from `start` in steps of size `step`.

  Unlike `tf.data.Dataset.range` which will stop at some ending number,
  `Counter` will produce elements indefinitely.

  >>> dataset = tf.data.experimental.Counter().take(5)
  >>> list(dataset.as_numpy_iterator())
  [0, 1, 2, 3, 4]
  >>> dataset.element_spec
  TensorSpec(shape=(), dtype=tf.int64, name=None)
  >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
  >>> dataset.element_spec
  TensorSpec(shape=(), dtype=tf.int32, name=None)
  >>> dataset = tf.data.experimental.Counter(start=2).take(5)
  >>> list(dataset.as_numpy_iterator())
  [2, 3, 4, 5, 6]
  >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
  >>> list(dataset.as_numpy_iterator())
  [2, 7, 12, 17, 22]
  >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
  >>> list(dataset.as_numpy_iterator())
  [10, 9, 8, 7, 6]

  Args:
    start: (Optional.) The starting value for the counter. Defaults to 0.
    step: (Optional.) The step size for the counter. Defaults to 1.
    dtype: (Optional.) The data type for counter elements. Defaults to
      `tf.int64`.

  Returns:
    A `Dataset` of scalar `dtype` elements.
  """
  with ops.name_scope("counter"):
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    step = ops.convert_to_tensor(step, dtype=dtype, name="step")
    return dataset_ops.Dataset.from_tensors(0).repeat(None).scan(
        start, lambda state, _: (state + step, state))


@tf_export(v1=["data.experimental.Counter"])
def CounterV1(start=0, step=1, dtype=dtypes.int64):
  return dataset_ops.DatasetV1Adapter(CounterV2(start, step, dtype))


CounterV1.__doc__ = CounterV2.__doc__

if tf2.enabled():
  Counter = CounterV2
else:
  Counter = CounterV1
