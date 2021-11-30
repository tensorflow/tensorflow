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
"""Tests for tf.strings.reduce_join."""
from absl.testing import parameterized
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class StringsReduceJoinOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def test_rank_one(self):
    input_array = [b'this', b'is', b'a', b'test']
    truth = b'thisisatest'
    truth_shape = []
    with self.cached_session():
      output = ragged_string_ops.reduce_join(
          inputs=input_array, axis=-1, keepdims=False, separator='')
      output_array = self.evaluate(output)
    self.assertAllEqual(truth, output_array)
    self.assertAllEqual(truth_shape, output.get_shape())

  @parameterized.parameters([
      {
          'input_array': [[
              b'this', b'is', b'a', b'test', b'for', b'ragged', b'tensors'
          ], [b'please', b'do', b'not', b'panic', b'!']],
          'axis': 0,
          'keepdims': False,
          'truth': [
              b'thisplease', b'isdo', b'anot', b'testpanic', b'for!', b'ragged',
              b'tensors'
          ],
          'truth_shape': [7],
      },
      {
          'input_array': [[
              b'this', b'is', b'a', b'test', b'for', b'ragged', b'tensors'
          ], [b'please', b'do', b'not', b'panic', b'!']],
          'axis': 1,
          'keepdims': False,
          'truth': [b'thisisatestforraggedtensors', b'pleasedonotpanic!'],
          'truth_shape': [2],
      },
      {
          'input_array': [[
              b'this', b'is', b'a', b'test', b'for', b'ragged', b'tensors'
          ], [b'please', b'do', b'not', b'panic', b'!']],
          'axis': 1,
          'keepdims': False,
          'truth': [
              b'this|is|a|test|for|ragged|tensors', b'please|do|not|panic|!'
          ],
          'truth_shape': [2],
          'separator': '|',
      },
      {
          'input_array': [[[b'a', b'b'], [b'b', b'c']], [[b'dd', b'ee']]],
          'axis': -1,
          'keepdims': False,
          'truth': [[b'a|b', b'b|c'], [b'dd|ee']],
          'truth_shape': [2, None],
          'separator': '|',
      },
      {
          'input_array': [[[[b'a', b'b', b'c'], [b'dd', b'ee']]],
                          [[[b'f', b'g', b'h'], [b'ii', b'jj']]]],
          'axis': -2,
          'keepdims': False,
          'truth': [[[b'a|dd', b'b|ee', b'c']], [[b'f|ii', b'g|jj', b'h']]],
          'truth_shape': [2, None, None],
          'separator': '|',
      },
      {
          'input_array': [[[b't', b'h', b'i', b's'], [b'i', b's'], [b'a'],
                           [b't', b'e', b's', b't']],
                          [[b'p', b'l', b'e', b'a', b's', b'e'],
                           [b'p', b'a', b'n', b'i', b'c']]],
          'axis': -1,
          'keepdims': False,
          'truth': [[b'this', b'is', b'a', b'test'], [b'please', b'panic']],
          'truth_shape': [2, None],
          'separator': '',
      },
      {
          'input_array': [[[[b't'], [b'h'], [b'i'], [b's']], [[b'i', b's']],
                           [[b'a', b'n']], [[b'e'], [b'r'], [b'r']]],
                          [[[b'p'], [b'l'], [b'e'], [b'a'], [b's'], [b'e']],
                           [[b'p'], [b'a'], [b'n'], [b'i'], [b'c']]]],
          'axis': -1,
          'keepdims': False,
          'truth': [[[b't', b'h', b'i', b's'], [b'is'], [b'an'],
                     [b'e', b'r', b'r']],
                    [[b'p', b'l', b'e', b'a', b's', b'e'],
                     [b'p', b'a', b'n', b'i', b'c']]],
          'truth_shape': [2, None, None],
          'separator': '',
      },
  ])
  def test_different_ranks(self,
                           input_array,
                           axis,
                           keepdims,
                           truth,
                           truth_shape,
                           separator=''):
    with self.cached_session():
      input_tensor = ragged_factory_ops.constant(input_array)
      output = ragged_string_ops.reduce_join(
          inputs=input_tensor,
          axis=axis,
          keepdims=keepdims,
          separator=separator)
      output_array = self.evaluate(output)
    self.assertAllEqual(truth, output_array)
    if all(isinstance(s, tensor_shape.Dimension) for s in output.shape):
      output_shape = [dim.value for dim in output.shape]
    else:
      output_shape = output.shape
    self.assertAllEqual(truth_shape, output_shape)


if __name__ == '__main__':
  googletest.main()
