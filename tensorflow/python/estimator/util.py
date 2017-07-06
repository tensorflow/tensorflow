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

"""Utility to retrieve function args.."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def fn_args(fn):
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  _, fn = tf_decorator.unwrap(fn)

  # Handle callables.
  if hasattr(fn, '__call__') and tf_inspect.ismethod(fn.__call__):
    return tuple(tf_inspect.getargspec(fn.__call__).args)

  # Handle functools.partial and similar objects.
  if hasattr(fn, 'func') and hasattr(fn, 'keywords') and hasattr(fn, 'args'):
    # Handle nested partial.
    original_args = fn_args(fn.func)
    if not original_args:
      return tuple()

    return tuple([
        arg for arg in original_args[len(fn.args):]
        if arg not in set((fn.keywords or {}).keys())
    ])

  # Handle function.
  return tuple(tf_inspect.getargspec(fn).args)
