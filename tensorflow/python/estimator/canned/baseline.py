# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""baseline python module.

Importing from tensorflow.python.estimator is unsupported
and will soon break!
"""
# pylint: disable=unused-import,g-bad-import-order,g-import-not-at-top,wildcard-import

from tensorflow_estimator.python.estimator.canned import baseline

# Include attrs that start with single underscore.
_HAS_DYNAMIC_ATTRIBUTES = True
baseline.__all__ = [s for s in dir(baseline) if not s.startswith('__')]

from tensorflow_estimator.python.estimator.canned.baseline import *
