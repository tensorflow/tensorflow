# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Bring in all of the public TensorFlow interface into this module (virtual pip).
"""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

try:
  from tensorflow_core import *
  # These need to be explicitly imported
  from tensorflow_core import __path__ as _new_path
  from tensorflow_core import keras
  __path__ = __path__ + _new_path
except ImportError as e:
  pass

try:
  from tensorflow_estimator import __path__ as _new_path
  __path__ = __path__ + _new_path
except ImportError as e:
  pass
