# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A mixin class that delegates another Trackable to be used when saving.

This is intended to be used with wrapper classes that cannot directly proxy the
wrapped object (e.g. with wrapt.ObjectProxy), because there are inner attributes
that cannot be exposed.

The Wrapper class itself cannot contain any Trackable children, as only the
delegated Trackable will be saved to checkpoint and SavedModel.

This class will "disappear" and be replaced with the wrapped inner Trackable
after a cycle of SavedModel saving and loading, unless the object is registered
and loaded with Keras.
"""


# TODO(kathywu): Delete this file after all imports have been moved to the path
# below.
from tensorflow.python.trackable import base_delegate
from tensorflow.python.util import deprecation

__getattr__ = deprecation.deprecate_moved_module(
    __name__, base_delegate, "2.11")
