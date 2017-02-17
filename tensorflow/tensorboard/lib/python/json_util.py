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

"""A module providing a function for serializing JSON values with Infinity.

Python provides no way to override how json.dumps serializes
Infinity/-Infinity/NaN; if allow_nan is true, it encodes them as
Infinity/-Infinity/NaN, in violation of the JSON spec and in violation of what
JSON.parse accepts. If it's false, it throws a ValueError, Neither subclassing
JSONEncoder nor passing a function in the |default| keyword argument overrides
this.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.util import compat

_INFINITY = float('inf')
_NEGATIVE_INFINITY = float('-inf')


def Cleanse(obj, encoding='utf-8'):
  """Makes Python object appropriate for JSON serialization.

  - Replaces instances of Infinity/-Infinity/NaN with strings.
  - Turns byte strings into unicode strings.
  - Turns sets into sorted lists.
  - Turns tuples into lists.

  Args:
    obj: Python data structure.
    encoding: Charset used to decode byte strings.

  Returns:
    Unicode JSON data structure.
  """
  if isinstance(obj, int):
    return obj
  elif isinstance(obj, float):
    if obj == _INFINITY:
      return 'Infinity'
    elif obj == _NEGATIVE_INFINITY:
      return '-Infinity'
    elif math.isnan(obj):
      return 'NaN'
    else:
      return obj
  elif isinstance(obj, bytes):
    return compat.as_text(obj, encoding)
  elif isinstance(obj, list) or isinstance(obj, tuple):
    return [Cleanse(i, encoding) for i in obj]
  elif isinstance(obj, set):
    return [Cleanse(i, encoding) for i in sorted(obj)]
  elif isinstance(obj, dict):
    return {Cleanse(k, encoding): Cleanse(v, encoding) for k, v in obj.items()}
  else:
    return obj
