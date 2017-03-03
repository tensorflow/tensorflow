# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Keyword args functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.util import decorator_utils


def keyword_args_only(func):
  """Decorator for marking specific function accepting keyword args only.

  This decorator raises a `ValueError` if the input `func` is called with any
  non-keyword args. This prevents the caller from providing the arguments in
  wrong order.

  Args:
    func: The function or method needed to be decorated.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If `func` is not callable.
  """

  decorator_utils.validate_callable(func, "keyword_args_only")
  @functools.wraps(func)
  def new_func(*args, **kwargs):
    """Keyword args only wrapper."""
    if args:
      raise ValueError(
          "Must use keyword args to call {}.".format(func.__name__))
    return func(**kwargs)
  return new_func
