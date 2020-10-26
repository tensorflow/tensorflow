# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Interface that provides access to Keras dependencies.

This library is a common interface that contains Keras functions needed by
TensorFlow and TensorFlow Lite and is required as per the dependency inversion
principle (https://en.wikipedia.org/wiki/Dependency_inversion_principle). As per
this principle, high-level modules (eg: TensorFlow and TensorFlow Lite) should
not depend on low-level modules (eg: Keras) and instead both should depend on a
common interface such as this file.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


_KERAS_CALL_CONTEXT_FUNCTION = None


def register_call_context_function(func):
  global _KERAS_CALL_CONTEXT_FUNCTION
  # TODO(scottzhu): Disable duplicated inject once keras is moved to
  # third_party/py/keras.
  _KERAS_CALL_CONTEXT_FUNCTION = func


def get_call_context_function():
  global _KERAS_CALL_CONTEXT_FUNCTION
  return _KERAS_CALL_CONTEXT_FUNCTION
