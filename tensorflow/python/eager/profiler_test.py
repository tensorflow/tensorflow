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
# ==============================================================================
"""Tests for eager profiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.profiler import trace_events_pb2
from tensorflow.python.eager import profiler
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util


class ProfilerTest(test_util.TensorFlowTestCase):

  def test_profile(self):
    profiler.start()
    three = constant_op.constant(3)
    five = constant_op.constant(5)
    product = three * five
    self.assertAllEqual(15, product)
    profile_result = profiler.stop()
    profile_pb = trace_events_pb2.Trace()
    profile_pb.ParseFromString(profile_result)
    profile_pb_str = '%s' % profile_pb
    self.assertTrue('Mul' in profile_pb_str)


if __name__ == '__main__':
  test.main()
