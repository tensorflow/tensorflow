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
"""Keras built-in constraints functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Constraints functions / callable classes.
from tensorflow.python.keras._impl.keras.constraints import Constraint
from tensorflow.python.keras._impl.keras.constraints import max_norm
from tensorflow.python.keras._impl.keras.constraints import MaxNorm
from tensorflow.python.keras._impl.keras.constraints import min_max_norm
from tensorflow.python.keras._impl.keras.constraints import MinMaxNorm
from tensorflow.python.keras._impl.keras.constraints import non_neg
from tensorflow.python.keras._impl.keras.constraints import NonNeg
from tensorflow.python.keras._impl.keras.constraints import unit_norm
from tensorflow.python.keras._impl.keras.constraints import UnitNorm

# Auxiliary utils.
# pylint: disable=g-bad-import-order
from tensorflow.python.keras._impl.keras.constraints import deserialize
from tensorflow.python.keras._impl.keras.constraints import serialize
from tensorflow.python.keras._impl.keras.constraints import get

del absolute_import
del division
del print_function
