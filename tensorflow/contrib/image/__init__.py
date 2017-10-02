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
"""Ops for image manipulation.

### API

This module provides functions for image manipulation; currently, chrominance
transformas (including changing saturation and hue) in YIQ space and
projective transforms (including rotation) are supported.

@@angles_to_projective_transforms
@@compose_transforms
@@adjust_yiq_hsv
@@random_yiq_hsv
@@rotate
@@transform
@@bipartite_match
@@single_image_random_dot_stereograms
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=line-too-long
from tensorflow.contrib.image.python.ops.distort_image_ops import adjust_hsv_in_yiq
from tensorflow.contrib.image.python.ops.distort_image_ops import random_hsv_in_yiq

from tensorflow.contrib.image.python.ops.image_ops import angles_to_projective_transforms
from tensorflow.contrib.image.python.ops.image_ops import compose_transforms
from tensorflow.contrib.image.python.ops.image_ops import rotate
from tensorflow.contrib.image.python.ops.image_ops import transform
from tensorflow.contrib.image.python.ops.single_image_random_dot_stereograms import single_image_random_dot_stereograms

from tensorflow.python.util.all_util import remove_undocumented

# pylint: enable=line-too-long

remove_undocumented(__name__)
