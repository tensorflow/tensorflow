# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TFGAN features module.

This module includes support for virtual batch normalization, buffer replay,
conditioning, etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse features into a single namespace.
# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.gan.python.features.python import clip_weights
from tensorflow.contrib.gan.python.features.python import conditioning_utils
from tensorflow.contrib.gan.python.features.python import random_tensor_pool
from tensorflow.contrib.gan.python.features.python import spectral_normalization
from tensorflow.contrib.gan.python.features.python import virtual_batchnorm

from tensorflow.contrib.gan.python.features.python.clip_weights import *
from tensorflow.contrib.gan.python.features.python.conditioning_utils import *
from tensorflow.contrib.gan.python.features.python.random_tensor_pool import *
from tensorflow.contrib.gan.python.features.python.spectral_normalization import *
from tensorflow.contrib.gan.python.features.python.virtual_batchnorm import *
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = clip_weights.__all__
_allowed_symbols += conditioning_utils.__all__
_allowed_symbols += random_tensor_pool.__all__
_allowed_symbols += spectral_normalization.__all__
_allowed_symbols += virtual_batchnorm.__all__
remove_undocumented(__name__, _allowed_symbols)
