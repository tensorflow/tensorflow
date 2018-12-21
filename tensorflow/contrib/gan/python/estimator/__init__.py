# Copyright 2016 Google Inc. All Rights Reserved.
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
"""TFGAN estimator module.

GANEstimator provides all the infrastructure support of a TensorFlow Estimator
with the feature support of TFGAN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Collapse `estimator` into a single namespace.
# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.gan.python.estimator.python import gan_estimator
from tensorflow.contrib.gan.python.estimator.python import head
from tensorflow.contrib.gan.python.estimator.python import stargan_estimator
from tensorflow.contrib.gan.python.estimator.python import tpu_gan_estimator

from tensorflow.contrib.gan.python.estimator.python.gan_estimator import *
from tensorflow.contrib.gan.python.estimator.python.head import *
from tensorflow.contrib.gan.python.estimator.python.stargan_estimator import *
from tensorflow.contrib.gan.python.estimator.python.tpu_gan_estimator import *
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ([
    'gan_estimator',
    'stargan_estimator',
    'tpu_gan_estimator',
    'head',
] + gan_estimator.__all__ + stargan_estimator.__all__ + head.__all__ +
                    tpu_gan_estimator.__all__)
remove_undocumented(__name__, _allowed_symbols)
