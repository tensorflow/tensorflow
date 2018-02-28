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
"""Utilities for programmable docstrings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six


def expand_docstring(**kwargs):
  """Decorator to programmatically expand the docstring.

  Args:
    **kwargs: Keyword arguments to set. For each key-value pair `k` and `v`,
      the key is found as `@{k}` in the docstring and replaced with `v`.

  Returns:
    Decorated function.
  """
  def _fn_wrapped(fn):
    """Original function with modified `__doc__` attribute."""
    doc = _trim(fn.__doc__)
    for k, v in six.iteritems(kwargs):
      # Capture each @{k} reference to replace with v.
      # We wrap the replacement in a function so no backslash escapes
      # are processed.
      pattern = r'@\{' + str(k) + r'\}'
      doc = re.sub(pattern, lambda match: v, doc)  # pylint: disable=cell-var-from-loop
    fn.__doc__ = doc
    return fn
  return _fn_wrapped


def _trim(docstring):
  """Trims docstring indentation.

  In general, multi-line docstrings carry their level of indentation when
  defined under a function or class method. This function standardizes
  indentation levels by removing them. Taken from PEP 257 docs.

  Args:
    docstring: Python string to trim indentation.

  Returns:
    Trimmed docstring.
  """
  if not docstring:
    return ''
  # Convert tabs to spaces (following the normal Python rules)
  # and split into a list of lines:
  lines = docstring.expandtabs().splitlines()
  # Determine minimum indentation (first line doesn't count):
  indent = None
  for line in lines[1:]:
    stripped = line.lstrip()
    if stripped:
      if indent is None:
        indent = len(line) - len(stripped)
      else:
        indent = min(indent, len(line) - len(stripped))
  # Remove indentation (first line is special):
  trimmed = [lines[0].strip()]
  if indent is not None:
    for line in lines[1:]:
      trimmed.append(line[indent:].rstrip())
  # Strip off trailing and leading blank lines:
  while trimmed and not trimmed[-1]:
    trimmed.pop()
  while trimmed and not trimmed[0]:
    trimmed.pop(0)
  # Return a single string:
  return '\n'.join(trimmed)
