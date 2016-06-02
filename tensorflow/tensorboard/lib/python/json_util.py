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


def WrapSpecialFloats(obj):
  """Replaces all instances of Infinity/-Infinity/NaN with strings."""
  if obj == float('inf'):
    return 'Infinity'
  elif obj == float('-inf'):
    return '-Infinity'
  elif isinstance(obj, float) and math.isnan(obj):
    return 'NaN'
  elif isinstance(obj, list) or isinstance(obj, tuple):
    return list(map(WrapSpecialFloats, obj))
  elif isinstance(obj, dict):
    return {
        WrapSpecialFloats(k): WrapSpecialFloats(v)
        for k, v in obj.items()
    }
  else:
    return obj
