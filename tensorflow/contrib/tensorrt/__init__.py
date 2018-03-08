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
# =============================================================================
"""Exposes the python wrapper for TensorRT graph transforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import errors

# pylint: disable=unused-import,wildcard-import,g-import-not-at-top
try:
  from tensorflow.contrib.tensorrt.python import *
except errors.NotFoundError as e:
  no_trt_message = (
      '**** Failed to initialize TensorRT. This is either because the TensorRT'
      ' installation path is not in LD_LIBRARY_PATH, or because you do not have'
      ' it installed. If not installed, please go to'
      ' https://developer.nvidia.com/tensorrt to download and install'
      ' TensorRT ****')
  raise e(no_trt_message)
# pylint: enable=unused-import,wildcard-import,g-import-not-at-top
