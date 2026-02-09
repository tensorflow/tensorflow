# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tests for ops to pack model inputs."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test
from tensorflow_text.python.ops import pad_model_inputs_ops


class ModelInputPackerTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      # Test out padding out when sequence length < max_seq_length.
      dict(
          pack_inputs=[
              [101, 1, 2, 102, 10, 20, 102],
              [101, 3, 4, 102, 30, 40, 50, 60],
              [101, 5, 6, 7, 8, 9, 102, 70],
          ],
          max_seq_length=10,
          expected=[[101, 1, 2, 102, 10, 20, 102, 0, 0, 0],
                    [101, 3, 4, 102, 30, 40, 50, 60, 0, 0],
                    [101, 5, 6, 7, 8, 9, 102, 70, 0, 0]],
          expected_mask=[[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
      ),
      dict(
          pack_inputs=[
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 1],
          ],
          expected=[
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          ],
          expected_mask=[
              [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          ],
          max_seq_length=10,
      ),
      # Test out truncation when sequence length > max_seq_length.
      dict(
          pack_inputs=[
              [0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 1],
          ],
          expected=[
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0],
          ],
          expected_mask=[
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
          ],
          max_seq_length=5,
      ),
  ])
  def testPadModelInputs(self,
                         pack_inputs,
                         expected,
                         expected_mask,
                         max_seq_length=10):
    # Pack everything as a RaggedTensor.
    pack_inputs = ragged_factory_ops.constant(pack_inputs)

    # Pad to max_seq_length and construct input_mask
    actual_padded, actual_mask = pad_model_inputs_ops.pad_model_inputs(
        pack_inputs, max_seq_length=max_seq_length, pad_value=0)

    # Verify the contents of all the padded (and maybe truncated) values as well
    # as the mask.
    self.assertAllEqual(expected, actual_padded)
    self.assertAllEqual(expected_mask, actual_mask)

  @parameterized.parameters([
      # Test out padding out when sequence length < max_seq_length.
      dict(
          pack_inputs=[
              [101, 1, 2, 102, 10, 20, 102],
              [101, 3, 4, 102, 30, 40, 50],
              [101, 5, 6, 700, 80, 90, 102],
          ],
          max_seq_length=10,
          expected=[[101, 1, 2, 102, 10, 20, 102, 0, 0, 0],
                    [101, 3, 4, 102, 30, 40, 50, 0, 0, 0],
                    [101, 5, 6, 700, 80, 90, 102, 0, 0, 0]],
          expected_mask=[[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
      ),
      dict(
          pack_inputs=[
              [0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 1],
          ],
          expected=[
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          ],
          expected_mask=[
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          ],
          max_seq_length=10,
      ),
      # Test out truncation when sequence length > max_seq_length.
      dict(
          pack_inputs=[
              [0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 1],
          ],
          expected=[
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0],
          ],
          expected_mask=[
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
          ],
          max_seq_length=5,
      ),
      # Test out single dimension < max_seq_length.
      dict(
          pack_inputs=[101, 1, 2, 102, 10, 20, 102],
          max_seq_length=10,
          expected=[101, 1, 2, 102, 10, 20, 102, 0, 0, 0],
          expected_mask=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
      ),
      # Test out single dimension when sequence length > max_seq_length.
      dict(
          pack_inputs=[101, 1, 2, 102, 10, 20, 102],
          max_seq_length=4,
          expected=[101, 1, 2, 102],
          expected_mask=[1, 1, 1, 1],
      ),
  ])
  def testPadModelInputsTensor(self,
                               pack_inputs,
                               expected,
                               expected_mask,
                               max_seq_length=10):
    # Pack everything as a RaggedTensor.
    pack_inputs = constant_op.constant(pack_inputs)

    # Pad to max_seq_length and construct input_mask
    actual_padded, actual_mask = pad_model_inputs_ops.pad_model_inputs(
        pack_inputs, max_seq_length=max_seq_length, pad_value=0)

    # Verify the contents of all the padded (and maybe truncated) values as well
    # as the mask.
    self.assertAllEqual(expected, actual_padded)
    self.assertAllEqual(expected_mask, actual_mask)

  @parameterized.named_parameters([
      ("PythonInt", lambda l: l),
      ("NpInt32", lambda l: np.array(l, np.int32)),
      ("NpInt64", lambda l: np.array(l, np.int64)),
      ("TfInt32", lambda l: constant_op.constant(l, dtypes.int32)),
      ("TfInt64", lambda l: constant_op.constant(l, dtypes.int64)),
  ])
  def testLengthType(self, length_fn):
    """Tests types beyond Python int for the max_seq_length argument."""
    pack_inputs = ragged_factory_ops.constant([[1, 2, 3, 4, 5],
                                               [8, 9]], dtypes.int32)
    max_seq_length = length_fn(3)
    expected_padded = [[1, 2, 3], [8, 9, 0]]
    expected_mask = [[1, 1, 1], [1, 1, 0]]
    actual_padded, actual_mask = pad_model_inputs_ops.pad_model_inputs(
        pack_inputs, max_seq_length=max_seq_length, pad_value=0)
    self.assertAllEqual(expected_padded, actual_padded)
    self.assertAllEqual(expected_mask, actual_mask)


if __name__ == "__main__":
  test.main()
