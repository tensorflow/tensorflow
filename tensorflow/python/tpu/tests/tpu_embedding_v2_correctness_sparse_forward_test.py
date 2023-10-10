# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.platform import test
from tensorflow.python.tpu.tests import tpu_embedding_v2_correctness_base_test


class TPUEmbeddingCorrectnessTest(
    tpu_embedding_v2_correctness_base_test.TPUEmbeddingCorrectnessBaseTest):

  @parameterized.parameters(
      ['sgd', 'adagrad', 'adam', 'ftrl', 'adagrad_momentum'])
  def test_embedding(self, optimizer_name):
    if optimizer_name != 'sgd':
      self.skip_if_oss()
    self._test_embedding(
        optimizer_name, training=False, sparse=True, is_high_dimensional=False)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
