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

"""Functions for Python 2 vs. 3 compatibility.

## Conversion routines
In addition to the functions below, `as_str` converts an object to a `str`.

@@as_bytes
@@as_text
@@as_str_any

## Types
The compatibility module also provides the following types:

* `bytes_or_text_types`
* `complex_types`
* `integral_types`
* `real_types`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers as _numbers

import numpy as _np
import six as _six

from tensorflow.python.util.all_util import remove_undocumented


def as_bytes(bytes_or_text, encoding='utf-8'):
  """Converts either bytes or unicode to `bytes`, using utf-8 encoding for text.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

  Returns:
    A `bytes` object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  if isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text.encode(encoding)
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    (bytes_or_text,))


def as_text(bytes_or_text, encoding='utf-8'):
  """Returns the given argument as a unicode string.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  if isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text.decode(encoding)
  else:
    raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)


# Convert an object to a `str` in both Python 2 and 3.
if _six.PY2:
  as_str = as_bytes
else:
  as_str = as_text


def as_str_any(value):
  """Converts to `str` as `str(value)`, but use `as_str` for `bytes`.

  Args:
    value: A object that can be converted to `str`.

  Returns:
    A `str` object.
  """
  if isinstance(value, bytes):
    return as_str(value)
  else:
    return str(value)


# Numpy 1.8 scalars don't inherit from numbers.Integral in Python 3, so we
# need to check them specifically.  The same goes from Real and Complex.
integral_types = (_numbers.Integral, _np.integer)
real_types = (_numbers.Real, _np.integer, _np.floating)
complex_types = (_numbers.Complex, _np.number)


# Either bytes or text.
bytes_or_text_types = (bytes, _six.text_type)


_allowed_symbols = [
    'as_str',
    'bytes_or_text_types',
    'complex_types',
    'integral_types',
    'real_types',
]

remove_undocumented(__name__, _allowed_symbols)
