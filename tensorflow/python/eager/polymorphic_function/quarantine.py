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
"""Internal APIs to be removed in the future."""

from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TODO(b/244360504): Remove this API in favour of the graph transformation API.
def add_function_callback(function_callback):
  """Add a callback function for Function creation.

  The callback function has the signature:

    `def function_callback(function: AtomicFunction) -> AtomicFunction`

  Repeated registration of the same callback function will cause repeated
  transformations.

  After a callback is added, it can be removed with the
  `remove_function_callback()` method.

  Args:
    function_callback: The callback to add.
  """
  atomic_function.FUNCTION_TRANSFORMS.append(function_callback)


# TODO(b/244360504): Remove this API in favour of the graph transformation API.
def remove_function_callback(function_callback):
  """Remove an already-added function callback.

  See the doc string of `add_function_callback()` for more information.

  Args:
    function_callback: The callback to remove.
  """
  atomic_function.FUNCTION_TRANSFORMS.remove(function_callback)


# TODO(b/244360504): Remove this API in favour of the graph transformation API.
def clear_function_callbacks():
  """Clear all function callbacks, if any have been regisered."""
  atomic_function.FUNCTION_TRANSFORMS.clear()


@deprecation.deprecated(
    None, "Use `tf.config.run_functions_eagerly` instead of the experimental "
    "version.")
@tf_export("config.experimental_run_functions_eagerly")
def experimental_run_functions_eagerly(run_eagerly):
  """Enables / disables eager execution of `tf.function`s.

  Calling `tf.config.experimental_run_functions_eagerly(True)` will make all
  invocations of `tf.function` run eagerly instead of running as a traced graph
  function.

  See `tf.config.run_functions_eagerly` for an example.

  Note: This flag has no effect on functions passed into tf.data transformations
  as arguments. tf.data functions are never executed eagerly and are always
  executed as a compiled Tensorflow Graph.

  Args:
    run_eagerly: Boolean. Whether to run functions eagerly.

  Returns:
    None
  """
  return eager_function_run.run_functions_eagerly(run_eagerly)


@deprecation.deprecated(
    None,
    "Use tf.config.functions_run_eagerly instead of the experimental version.")
@tf_export("config.experimental_functions_run_eagerly")
def experimental_functions_run_eagerly():
  """Returns the value of the `experimental_run_functions_eagerly` setting."""
  return eager_function_run.functions_run_eagerly()
