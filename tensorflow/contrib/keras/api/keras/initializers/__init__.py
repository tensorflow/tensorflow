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
"""Keras built-in initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Initializer functions / callable classes.
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.initializers import Identity
from tensorflow.python.keras.initializers import Initializer
from tensorflow.python.keras.initializers import Ones
from tensorflow.python.keras.initializers import Orthogonal
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.initializers import Zeros

# Functional interface.
# pylint: disable=g-bad-import-order
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.initializers import he_uniform
from tensorflow.python.keras.initializers import lecun_normal
from tensorflow.python.keras.initializers import lecun_uniform

# Auxiliary utils.
from tensorflow.python.keras.initializers import deserialize
from tensorflow.python.keras.initializers import serialize
from tensorflow.python.keras.initializers import get

del absolute_import
del division
del print_function
