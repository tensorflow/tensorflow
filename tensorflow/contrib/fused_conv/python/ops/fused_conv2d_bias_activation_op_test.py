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

"""Tests for fused convolutions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.fused_conv.python.ops import fused_conv2d_bias_activation_op_test_base as test_base
from tensorflow.python.platform import test


# Instantiate three test suites from test_base, mixing in test.TestCase as
# the test framework.
class FusedConv2DBiasActivationTest(test_base.FusedConv2DBiasActivationTest,
                                    test.TestCase):
  pass


class FusedConvInt8CPUTests(test_base.FusedConvInt8CPUTests, test.TestCase):
  pass


class FusedConvInt8CorrespondenceTests(
    test_base.FusedConvInt8CorrespondenceTests, test.TestCase):
  pass


if __name__ == '__main__':
  test.main()
