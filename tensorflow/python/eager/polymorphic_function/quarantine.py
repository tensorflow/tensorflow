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
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


# TODO(b/258247871): Remove in favor of tf.function.
@tf_export("__internal__.function.defun_with_attributes", v1=[])
def defun_with_attributes(func=None,
                          input_signature=None,
                          attributes=None,
                          autograph=True,
                          experimental_autograph_options=None,
                          jit_compile=None,
                          reduce_retracing=False):
  """Compiles a Python function into a callable TensorFlow graph.

  This function supports adding extra function attributes. See detailed
  documentation in defun(). Currently this is not exposed in public API since we
  don't expect user to directly use attributes, and attribute won't work by
  itself. This assumption might change in future.

  Args:
    func: function to be compiled.
    input_signature: same as defun()'s input_signature.
    attributes: A dictionary of arguments which will be added to function def as
      attributes. Currently only support primitive types as value, and only
      allowlisted attribute name is allowed. Unallowlisted attribute name or
      unsupported value will result into ValueError. `func_name` is also one of
      the allowlisted argument which is a python string, and sets the name for
      this `ConcreteFunction` in the graph.
    autograph: same as defun()'s autograph.
    experimental_autograph_options: same as defun()'s
      experimental_autograph_options.
    jit_compile: same as defun()'s jit_compile.
    reduce_retracing: same as defun()'s reduce_retracing

  Returns:
    Same as the return value of defun, with attributes added to the function in
    graph.
  """

  # TODO(apassos): deal with captured global state. Deal with control flow.
  def decorated(function):
    try:
      if attributes:
        name = attributes.pop("func_name", function.__name__)
      else:
        name = function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        function,
        polymorphic_function.Function(
            function,
            name,
            input_signature=input_signature,
            autograph=autograph,
            jit_compile=jit_compile,
            reduce_retracing=reduce_retracing,
            experimental_autograph_options=experimental_autograph_options,
            experimental_attributes=attributes,
        ),
    )

  # This code path is for the `foo = tfe.defun(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tfe.defun(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tfe.defun(...)(foo)`
  return decorated


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
