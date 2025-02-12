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
"""Compatibility functions.

The `tf.compat` module contains two sets of compatibility functions.

## Tensorflow 1.x and 2.x APIs

The `compat.v1` and `compat.v2` submodules provide a complete copy of both the
`v1` and `v2` APIs for backwards and forwards compatibility across TensorFlow
versions 1.x and 2.x. See the
[migration guide](https://www.tensorflow.org/guide/migrate) for details.

## Utilities for writing compatible code

Aside from the `compat.v1` and `compat.v2` submodules, `tf.compat` also contains
a set of helper functions for writing code that works in both:

* TensorFlow 1.x and 2.x
* Python 2 and 3


## Type collections

The compatibility module also provides the following aliases for common
sets of python types:

* `bytes_or_text_types`
* `complex_types`
* `integral_types`
* `real_types`

API docstring: tensorflow.compat
"""

import codecs
import collections.abc as collections_abc  # pylint: disable=unused-import
import numbers as _numbers

import numpy as _np

from tensorflow.python.util.tf_export import tf_export


def as_bytes(bytes_or_text, encoding='utf-8'):
  """Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

  Uses utf-8 encoding for text by default.

  Args:
    bytes_or_text: A `bytearray`, `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

  Returns:
    A `bytes` object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  # Validate encoding, a LookupError will be raised if invalid.
  encoding = codecs.lookup(encoding).name
  if isinstance(bytes_or_text, bytearray):
    return bytes(bytes_or_text)
  elif isinstance(bytes_or_text, str):
    return bytes_or_text.encode(encoding)
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    (bytes_or_text,))


def as_text(bytes_or_text, encoding='utf-8'):
  """Converts any string-like python input types to unicode.

  Returns the input as a unicode string. Uses utf-8 encoding for text
  by default.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  # Validate encoding, a LookupError will be raised if invalid.
  encoding = codecs.lookup(encoding).name
  if isinstance(bytes_or_text, str):
    return bytes_or_text
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text.decode(encoding)
  else:
    raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)


def as_str(bytes_or_text, encoding='utf-8'):
  """Acts as an alias for the `as_text` function..

  Args:
    bytes_or_text: The input value to be converted. A bytes or unicode object.
    encoding: Optional string. The encoding to use if bytes_or_text is a bytes
      object. Defaults to 'utf-8'.

  Returns:
    A unicode string.

  Raises:
    TypeError: If bytes_or_text is not a bytes or unicode object.
    UnicodeDecodeError: If bytes_or_text is a bytes object and cannot be
                        decoded using the specified encoding.
  """
  return as_text(bytes_or_text, encoding)

tf_export('compat.as_text')(as_text)
tf_export('compat.as_bytes')(as_bytes)
tf_export('compat.as_str')(as_str)


@tf_export('compat.as_str_any')
def as_str_any(value, encoding='utf-8'):
  """Converts input to `str` type.

     Uses `str(value)`, except for `bytes` typed inputs, which are converted
     using `as_str`.

  Args:
    value: A object that can be converted to `str`.
    encoding: Encoding for `bytes` typed inputs.

  Returns:
    A `str` object.
  """
  if isinstance(value, bytes):
    return as_str(value, encoding=encoding)
  else:
    return str(value)


@tf_export('compat.path_to_str')
def path_to_str(path):
  r"""Converts input which is a `PathLike` object to `str` type.

  Converts from any python constant representation of a `PathLike` object to
  a string. If the input is not a `PathLike` object, simply returns the input.

  Args:
    path: An object that can be converted to path representation.

  Returns:
    A `str` object.

  Usage:
    In case a simplified `str` version of the path is needed from an
    `os.PathLike` object.

  Examples:
  ```python
  $ tf.compat.path_to_str('C:\XYZ\tensorflow\./.././tensorflow')
  'C:\XYZ\tensorflow\./.././tensorflow' # Windows OS
  $ tf.compat.path_to_str(Path('C:\XYZ\tensorflow\./.././tensorflow'))
  'C:\XYZ\tensorflow\..\tensorflow' # Windows OS
  $ tf.compat.path_to_str(Path('./corpus'))
  'corpus' # Linux OS
  $ tf.compat.path_to_str('./.././Corpus')
  './.././Corpus' # Linux OS
  $ tf.compat.path_to_str(Path('./.././Corpus'))
  '../Corpus' # Linux OS
  $ tf.compat.path_to_str(Path('./..////../'))
  '../..' # Linux OS

  ```
  """
  if hasattr(path, '__fspath__'):
    path = as_str_any(path.__fspath__())
  return path


def path_to_bytes(path):
  r"""Converts input which is a `PathLike` object to `bytes`.

  Converts from any python constant representation of a `PathLike` object
  or `str` to bytes.

  Args:
    path: An object that can be converted to path representation.

  Returns:
    A `bytes` object.

  Usage:
    In case a simplified `bytes` version of the path is needed from an
    `os.PathLike` object.
  """
  if hasattr(path, '__fspath__'):
    path = path.__fspath__()
  return as_bytes(path)


# Numpy 1.8 scalars don't inherit from numbers.Integral in Python 3, so we
# need to check them specifically.  The same goes from Real and Complex.
integral_types = (_numbers.Integral, _np.integer)
tf_export('compat.integral_types').export_constant(__name__, 'integral_types')
real_types = (_numbers.Real, _np.integer, _np.floating)
tf_export('compat.real_types').export_constant(__name__, 'real_types')
complex_types = (_numbers.Complex, _np.number)
tf_export('compat.complex_types').export_constant(__name__, 'complex_types')

# Either bytes or text.
bytes_or_text_types = (bytes, str)
tf_export('compat.bytes_or_text_types').export_constant(__name__,
                                                        'bytes_or_text_types')
