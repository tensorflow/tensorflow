# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Keras built-in regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Regularizer functions / callable classes.
from tensorflow.python.keras._impl.keras.regularizers import L1L2
from tensorflow.python.keras._impl.keras.regularizers import Regularizer

# Functional interface.
# pylint: disable=g-bad-import-order
from tensorflow.python.keras._impl.keras.regularizers import l1
from tensorflow.python.keras._impl.keras.regularizers import l2
from tensorflow.python.keras._impl.keras.regularizers import l1_l2

# Auxiliary utils.
from tensorflow.python.keras._impl.keras.regularizers import deserialize
from tensorflow.python.keras._impl.keras.regularizers import serialize
from tensorflow.python.keras._impl.keras.regularizers import get

del absolute_import
del division
del print_function
