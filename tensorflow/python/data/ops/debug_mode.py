# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Python debug mode enabler."""

from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export


DEBUG_MODE = False


@tf_export("data.experimental.enable_debug_mode")
def enable_debug_mode():
  """Enables debug mode for tf.data.

  Example usage with pdb module:
  ```
  import tensorflow as tf
  import pdb

  tf.data.experimental.enable_debug_mode()

  def func(x):
    # Python 3.7 and older requires `pdb.Pdb(nosigint=True).set_trace()`
    pdb.set_trace()
    x = x + 1
    return x

  dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  dataset = dataset.map(func)

  for item in dataset:
    print(item)
  ```

  The effect of debug mode is two-fold:

  1) Any transformations that would introduce asynchrony, parallelism, or
  non-determinism to the input pipeline execution will be forced to execute
  synchronously, sequentially, and deterministically.

  2) Any user-defined functions passed into tf.data transformations such as
  `map` will be wrapped in `tf.py_function` so that their body is executed
  "eagerly" as a Python function as opposed to a traced TensorFlow graph, which
  is the default behavior. Note that even when debug mode is enabled, the
  user-defined function is still traced  to infer the shape and type of its
  outputs; as a consequence, any `print` statements or breakpoints will be
  triggered once during the tracing before the actual execution of the input
  pipeline.

  NOTE: As the debug mode setting affects the construction of the tf.data input
  pipeline, it should be enabled before any tf.data definitions.

  Raises:
    ValueError: When invoked from graph mode.
  """
  if context.executing_eagerly():
    toggle_debug_mode(True)
  else:
    raise ValueError("`enable_debug_mode() is only supported in eager mode.")


def toggle_debug_mode(debug_mode):
  global DEBUG_MODE
  DEBUG_MODE = debug_mode
