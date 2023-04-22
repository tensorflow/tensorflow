# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU Embeddings mid level API utils on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2_utils


class TPUEmbeddingOptimizerTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters(tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_grad_clip_with_accumulation_off(self, optimizer):
    with self.assertRaisesRegex(ValueError, 'accumulation'):
      optimizer(use_gradient_accumulation=False, clipvalue=0.)
    with self.assertRaisesRegex(ValueError, 'accumulation'):
      optimizer(use_gradient_accumulation=False, clipvalue=(None, 1.))

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_grad_clip_with_tuple(self, optimizer):
    opt = optimizer(clipvalue=(-1., 1.))
    self.assertEqual(-1., opt.clip_gradient_min)
    self.assertEqual(1., opt.clip_gradient_max)

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_grad_clip_with_single_value(self, optimizer):
    opt = optimizer(clipvalue=1.)
    self.assertEqual(-1., opt.clip_gradient_min)
    self.assertEqual(1., opt.clip_gradient_max)

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_grad_clip_with_tuple_and_none(self, optimizer):
    opt = optimizer(clipvalue=(None, 1))
    self.assertIsNone(opt.clip_gradient_min)
    self.assertEqual(1., opt.clip_gradient_max)


class ConfigTest(test.TestCase):

  def test_table_config_repr(self):
    table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=2, dim=4, initializer=None,
        combiner='sum', name='table')

    self.assertEqual(
        repr(table),
        'TableConfig(vocabulary_size=2, dim=4, initializer=None, '
        'optimizer=None, combiner=\'sum\', name=\'table\')')

  def test_feature_config_repr(self):
    table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=2, dim=4, initializer=None,
        combiner='sum', name='table')

    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=table, name='feature')

    self.assertEqual(
        repr(feature_config),
        'FeatureConfig(table=TableConfig(vocabulary_size=2, dim=4, '
        'initializer=None, optimizer=None, combiner=\'sum\', name=\'table\'), '
        'max_sequence_length=0, validate_weights_and_indices=True, '
        'name=\'feature\')')


class TPUEmbeddingConfigurationTest(test.TestCase):

  def test_no_truncate(self):
    truncate_length = 14937  # Experimentally maximum string length loggable.

    config = tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration()
    for i in range(500):
      td = config.table_descriptor.add()
      td.name = 'table_{}'.format(i)
      td.vocabulary_size = i
    config.num_hosts = 2
    config.num_tensor_cores = 4
    config.batch_size_per_tensor_core = 128

    self.assertGreater(
        len(str(config)), truncate_length,
        'Test sanity check: generated config should be of truncating length.')

    with self.assertLogs() as logs:
      tpu_embedding_v2_utils.log_tpu_embedding_configuration(config)

    self.assertIn('table_499', ''.join(logs.output))
    for line in logs.output:
      self.assertLess(
          len(line), truncate_length,
          'Logging function lines should not be of truncating length.')


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
