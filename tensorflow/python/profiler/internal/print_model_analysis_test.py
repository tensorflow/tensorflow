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
"""print_model_analysis test."""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
TEST_OPTIONS = {
    'max_depth': 10000,
    'min_bytes': 0,
    'min_micros': 0,
    'min_params': 0,
    'min_float_ops': 0,
    'order_by': 'name',
    'account_type_regexes': ['.*'],
    'start_name_regexes': ['.*'],
    'trim_name_regexes': [],
    'show_name_regexes': ['.*'],
    'hide_name_regexes': [],
    'account_displayed_op_only': True,
    'select': ['params'],
    'output': 'stdout',
}

# pylint: enable=bad-whitespace
# pylint: enable=bad-continuation


class PrintModelAnalysisTest(test.TestCase):

  def _BuildSmallModel(self):
    image = array_ops.zeros([2, 6, 6, 3])
    kernel = variable_scope.get_variable(
        'DW', [6, 6, 3, 6],
        dtypes.float32,
        initializer=init_ops.random_normal_initializer(stddev=0.001))
    x = nn_ops.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
    return x


if __name__ == '__main__':
  test.main()
