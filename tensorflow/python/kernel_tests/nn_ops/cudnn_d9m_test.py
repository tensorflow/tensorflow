# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TF_CUDNN_DETERMINISTIC=1."""

import os

from tensorflow.python.kernel_tests.nn_ops import cudnn_deterministic_base
from tensorflow.python.platform import test

ConvolutionTest = cudnn_deterministic_base.ConvolutionTest

if __name__ == '__main__':
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require another test file to be added.
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  test.main()
