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

"""Global registry for OpDefs."""

import threading

from tensorflow.core.framework import op_def_pb2
# pylint: disable=invalid-import-order,g-bad-import-order, wildcard-import, unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import _op_def_registry

# The cache amortizes ProtoBuf serialization/deserialization overhead
# on the language boundary. If an OpDef has been looked up, its Python
# representation is cached.
_cache = {}
_cache_lock = threading.Lock()


def get(name):
  """Returns an OpDef for a given `name` or None if the lookup fails."""
  try:
    return _cache[name]
  except KeyError:
    pass

  with _cache_lock:
    try:
      # Return if another thread has already populated the cache.
      return _cache[name]
    except KeyError:
      pass

    serialized_op_def = _op_def_registry.get(name)
    if serialized_op_def is None:
      return None

    op_def = op_def_pb2.OpDef()
    op_def.ParseFromString(serialized_op_def)
    _cache[name] = op_def
    return op_def


# TODO(b/141354889): Remove once there are no callers.
def sync():
  """No-op. Used to synchronize the contents of the Python registry with C++."""
