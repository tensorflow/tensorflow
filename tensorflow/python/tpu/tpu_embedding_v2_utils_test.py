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

from absl.testing import parameterized

from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
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

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_equal_and_hash_function(self, optimizer):
    opt1 = optimizer(0.1)
    opt2 = optimizer(0.1)
    opt3 = optimizer(0.2)
    self.assertEqual(opt1, opt2)
    self.assertEqual(hash(opt1), hash(opt2))
    self.assertNotEqual(opt1, opt3)
    self.assertNotEqual(hash(opt1), hash(opt3))


class TPUEmbeddingCustomCombinerTest(parameterized.TestCase, test.TestCase):

  def get_sum_combiner(self):

    @def_function.function
    def sum_combiner(valency, vectors):
      max_valency = vectors.shape[0]
      valid_mask = array_ops.range(max_valency) < valency
      vectors_masked = array_ops.where(
          array_ops.expand_dims(valid_mask, axis=-1),
          vectors,
          array_ops.zeros_like(vectors),
      )
      return math_ops.reduce_sum(vectors_masked, axis=0)

    return sum_combiner

  def get_positional_weight_combiner(self):

    @def_function.function
    def positional_weight_combiner(valency, vectors, weights):
      max_valency = vectors.shape[0]
      valid_mask = array_ops.range(max_valency) < valency
      vectors_masked = array_ops.where(
          array_ops.expand_dims(valid_mask, axis=-1),
          vectors,
          array_ops.zeros_like(vectors),
      )
      return math_ops.matvec(vectors_masked, weights, transpose_a=True)

    return positional_weight_combiner

  def test_zero_num_weights_combiner_has_no_slots(self):
    combiner = tpu_embedding_v2_utils.CustomCombiner(
        self.get_sum_combiner(),
        max_valency=16,
        num_weights=0,
    )
    self.assertEmpty(combiner._slot_names())
    self.assertEmpty(combiner._slot_initializers())

  def test_name_starts_with_custom_combiner(self):
    combiner = tpu_embedding_v2_utils.CustomCombiner(
        self.get_sum_combiner(),
        max_valency=16,
    )
    self.assertStartsWith(str(combiner), 'custom_combiner')

  def test_non_zero_weights_requires_initializer(self):
    with self.assertRaisesRegex(ValueError, '`initializer` must be set'):
      tpu_embedding_v2_utils.CustomCombiner(
          self.get_positional_weight_combiner(),
          max_valency=16,
          num_weights=16,
      )

  def test_non_zero_weights_has_one_slot_variable(self):
    combiner = tpu_embedding_v2_utils.CustomCombiner(
        self.get_positional_weight_combiner(),
        max_valency=16,
        num_weights=16,
        initializer=init_ops_v2.zeros_initializer,
    )
    self.assertLen(combiner._slot_names(), 1)
    self.assertLen(combiner._slot_initializers(), 1)
    self.assertStartsWith(combiner._slot_names()[0], 'custom_combiner')


class ConfigTest(test.TestCase):

  def test_table_config_repr(self):
    table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=2, dim=4,
        combiner='sum', name='table')

    self.assertEqual(
        repr(table),
        'TableConfig(vocabulary_size=2, dim=4, initializer=None, '
        'optimizer=None, combiner=\'sum\', name=\'table\', '
        'quantization_config=None)')

  def test_feature_config_repr(self):
    table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=2, dim=4, initializer=None,
        combiner='sum', name='table')

    feature_config = tpu_embedding_v2_utils.FeatureConfig(
        table=table, output_shape=[16, 4], name='feature')

    self.assertEqual(
        repr(feature_config),
        'FeatureConfig(table=TableConfig(vocabulary_size=2, dim=4, '
        'initializer=None, optimizer=None, combiner=\'sum\', '
        'name=\'table\', quantization_config=None), max_sequence_length=0, '
        'validate_weights_and_indices=True, output_shape=TensorShape([16, 4]), '
        'name=\'feature\')')

  def test_quantization_config_num_buckets(self):
    with self.assertRaisesRegex(ValueError, 'num_buckets'):
      tpu_embedding_v2_utils.QuantizationConfig(0, -1, 1)

  def test_quantization_config_repr(self):
    quantization_config = tpu_embedding_v2_utils.QuantizationConfig(
        num_buckets=10, lower=-1.0, upper=1.0)

    self.assertEqual(
        repr(quantization_config),
        'QuantizationConfig(num_buckets=10, lower=-1.0, upper=1.0)')


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


class TPUEmbeddingUtilityFunctionTest(test.TestCase):

  def test_sort_device_spec_strings(self):
    device_spec_strings = []
    for task in [2, 3, 0, 1]:  # Intentionally permuted
      for device in range(8):
        device_spec_strings.append(
            f'/job:trainer/replica:0/task:{task}/device:TPU:{device}'
        )
    sorted_specs = tpu_embedding_v2_utils._sort_device_spec_strings(
        device_spec_strings
    )

    self.assertEqual(sorted_specs, sorted(device_spec_strings))


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
