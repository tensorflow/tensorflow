# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for model_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries import model_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ModelUtilsTest(test.TestCase):

  def test_parameter_switching(self):
    parameter = array_ops.constant(5)
    overridden_parameter = array_ops.constant(3)
    with self.test_session():
      getter = model_utils.parameter_switch({overridden_parameter: 4})
      self.assertEqual(5, getter(parameter))
      self.assertEqual(4, getter(overridden_parameter))


if __name__ == "__main__":
  test.main()
