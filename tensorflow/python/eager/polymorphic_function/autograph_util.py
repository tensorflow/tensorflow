# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unidiomatic-typecheck
"""Autograph utility functions for polymorphic_function."""

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.util import tf_decorator


def py_func_from_autograph(
    python_func,
    autograph_options=None,
):
  """Compile a python function using autograph, for use with FuncGraph.

  Args:
    python_func: the Python function to compile.
    autograph_options: additional knobs to control when `autograph=True`.
      See https://www.tensorflow.org/guide/autograph for more information.
  Returns:
    python_func, converted using autograph.
  """
  _, original_func = tf_decorator.unwrap(python_func)

  def autograph_handler(*args, **kwargs):
    """Calls a converted version of original_func."""
    try:
      return api.converted_call(
          original_func,
          args,
          kwargs,
          options=converter.ConversionOptions(
              recursive=True,
              optional_features=autograph_options,
              user_requested=True,
          ))
    except Exception as e:  # pylint:disable=broad-except
      if hasattr(e, "ag_error_metadata"):
        raise e.ag_error_metadata.to_exception(e)
      else:
        raise

  # Wrapping around a decorator allows checks like tf_inspect.getargspec
  # to be accurate.
  converted_func = tf_decorator.make_decorator(original_func, autograph_handler)
  return tf_decorator.rewrap(python_func, original_func, converted_func)
