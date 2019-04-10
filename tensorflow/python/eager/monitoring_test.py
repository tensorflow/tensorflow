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
"""Tests for monitoring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


class MonitoringTest(test_util.TensorFlowTestCase):

  def test_monitoring(self):
    # These methods should not throw any exception.
    monitoring.gauge('test/gauge', 'label', 1)
    monitoring.counter('test/counter', 'label', 1)
    monitoring.sampler('test/sampler', 'label', 1.0)


if __name__ == '__main__':
  test.main()
