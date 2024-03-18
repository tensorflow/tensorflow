# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.python.platform import test


class TestingTest(test.TestCase):

  def test_parameter_combinations(self):
    """Tests that parameter_combinations returns correct combinations."""
    test_parameters = [{
        'shapes': [
            [3, 3],
            [3, None],
        ],
        'has_bias': [True, False],
    }]
    combinations = testing.parameter_combinations(test_parameters)

    self.assertLen(combinations, 4)
    self.assertIn({'shapes': [3, 3], 'has_bias': True}, combinations)
    self.assertIn({'shapes': [3, 3], 'has_bias': False}, combinations)
    self.assertIn({'shapes': [3, None], 'has_bias': True}, combinations)
    self.assertIn({'shapes': [3, None], 'has_bias': False}, combinations)


if __name__ == '__main__':
  test.main()
