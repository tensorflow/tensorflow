"""A module providing a function for serializing JSON values with Infinity.

Python provides no way to override how json.dumps serializes
Infinity/-Infinity/NaN; if allow_nan is true, it encodes them as
Infinity/-Infinity/NaN, in violation of the JSON spec and in violation of what
JSON.parse accepts. If it's false, it throws a ValueError, Neither subclassing
JSONEncoder nor passing a function in the |default| keyword argument overrides
this.
"""

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
    return map(WrapSpecialFloats, obj)
  elif isinstance(obj, dict):
    return {
        WrapSpecialFloats(k): WrapSpecialFloats(v)
        for k, v in obj.items()
    }
  else:
    return obj
