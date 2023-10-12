# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests VLOG printing in TensorFlow."""


# Must set the VLOG environment variables before importing TensorFlow.
import os
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# pylint: disable=g-import-not-at-top
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class VlogTest(test.TestCase):

  # Runs a simple conv graph to check if VLOG crashes.
  def test_simple_conv(self):
    height, width = 7, 9
    images = random_ops.random_uniform((5, height, width, 3))
    w = random_ops.random_normal([5, 5, 3, 32], mean=0, stddev=1)
    nn_ops.conv2d(images, w, strides=[1, 1, 1, 1], padding="SAME")


if __name__ == "__main__":
  test.main()
