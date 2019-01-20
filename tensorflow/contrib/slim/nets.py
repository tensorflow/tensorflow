# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""TF-Slim Nets.

## Standard Networks.
@@alexnet_v2
@@inception_v1
@@inception_v1_base
@@inception_v2
@@inception_v2_base
@@inception_v3
@@inception_v3_base
@@overfeat
@@vgg_a
@@vgg_16
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,
# Collapse nets into a single namespace.
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import overfeat
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.python.util.all_util import make_all
# pylint: enable=unused-import

__all__ = make_all(__name__)
