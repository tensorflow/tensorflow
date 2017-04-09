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

# pylint: disable=line-too-long
"""This library provides a set of high-level neural networks layers.

@@dense
@@dropout
@@conv1d
@@conv2d
@@conv3d
@@separable_conv2d
@@conv2d_transpose
@@average_pooling1d
@@max_pooling1d
@@average_pooling2d
@@max_pooling2d
@@average_pooling3d
@@max_pooling3d
@@batch_normalization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=g-bad-import-order,unused-import

# Core layers.
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout

# Convolutional layers.
from tensorflow.python.layers.convolutional import conv1d
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.convolutional import conv3d
from tensorflow.python.layers.convolutional import separable_conv2d
from tensorflow.python.layers.convolutional import conv2d_transpose

# Pooling layers.
from tensorflow.python.layers.pooling import average_pooling1d
from tensorflow.python.layers.pooling import max_pooling1d
from tensorflow.python.layers.pooling import average_pooling2d
from tensorflow.python.layers.pooling import max_pooling2d
from tensorflow.python.layers.pooling import average_pooling3d
from tensorflow.python.layers.pooling import max_pooling3d

# Normalization layers.
from tensorflow.python.layers.normalization import batch_normalization

# pylint: enable=g-bad-import-order,unused-import

_allowed_symbols = []

remove_undocumented(__name__, _allowed_symbols)
