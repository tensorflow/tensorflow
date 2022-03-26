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
"""Tests for TPUEmbeddingV0 mid level API on TPU."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v1
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import util


class TPUEmbeddingCheckpointTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def _get_strategy(self):
    # We can cache the strategy as TPUEmbeddingV0 doesn't require
    # reconfiguration to the tpu.
    if hasattr(self, 'strategy'):
      return self.strategy
    return super(TPUEmbeddingCheckpointTest, self)._get_strategy()

  def _create_mid_level(self, optimizer=None):
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    return tpu_embedding_v1.TPUEmbeddingV0(
        feature_config=self.feature_config, optimizer=optimizer)

  def test_checkpoint_save_and_restore(self):
    strategy = self._get_strategy()
    with strategy.scope():
      first_mid_level_contents = np.ones((4, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=4,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v1.TPUEmbeddingV0(
          feature_config, first_mid_level_optimizer)
      first_mid_level.build()

    first_checkpoint = util.Checkpoint(model=first_mid_level)
    first_checkpoint.save(self._get_tmpdir('restore', 'save'))

    with strategy.scope():
      second_mid_level_contents = np.ones((4, 4)) * 2
      second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(second_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=4,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)
      second_mid_level = tpu_embedding_v1.TPUEmbeddingV0(
          feature_config, second_mid_level_optimizer)
      second_mid_level.build()
    # We restore the checkpoint of our first model into our second model.
    second_checkpoint = util.Checkpoint(model=second_mid_level)
    second_checkpoint.restore(self._get_tmpdir('restore', 'save-1'))

    self.assertAllClose(
        first_mid_level_contents,
        second_mid_level._variables['table']['parameters'].numpy(),
        msg='Second mid level api should have restored the first model values.')

  def test_model_export_cpu(self):
    strategy = self._get_strategy()

    with strategy.scope():
      first_mid_level_contents = np.ones((4, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=4,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v1.TPUEmbeddingV0(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build()

    cpu_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config, cpu_mid_level_optimizer)

    cpu_mid_level.build()

    tpu_checkpoint = util.Checkpoint(model=first_mid_level)
    tpu_checkpoint.save(self._get_tmpdir('export_cpu', 'save'))

    # We restore the checkpoint of our tpu mid level onto our cpu mid level.
    cpu_checkpoint = util.Checkpoint(model=cpu_mid_level)
    cpu_checkpoint.restore(self._get_tmpdir('export_cpu', 'save-1'))

    @def_function.function
    def serve_tensors(features):
      features = tpu_embedding_for_serving.cpu_embedding_lookup(
          features, None, cpu_mid_level.embedding_tables,
          cpu_mid_level._feature_config)
      return features[0]

    signatures = {
        'serving_default':
            serve_tensors.get_concrete_function((tensor_spec.TensorSpec(
                shape=(2,), dtype=dtypes.int32, name='feature'),))
    }
    save.save(
        cpu_mid_level,
        export_dir=self._get_tmpdir('export_cpu', 'exported_model'),
        signatures=signatures)

    imported = load.load(self._get_tmpdir('export_cpu', 'exported_model'))
    predict_fn = imported.signatures['serving_default']

    input_feature_value = np.array([1, 0])
    input_batch = (constant_op.constant(
        input_feature_value, dtype=dtypes.int32),)
    prediction = predict_fn(*input_batch)['output_0']
    self.assertAllClose(prediction.numpy(),
                        first_mid_level_contents[input_feature_value])

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam,
                            tpu_embedding_v2_utils.FTRL)
  def test_check_checkpoint_variable_names_are_same_on_cpu_and_tpu(
      self, optimizer):
    # Reinitialize the TPU so that we can re-initialize the embeddings with the
    # given optimizer.
    if optimizer != tpu_embedding_v2_utils.SGD:
      self.skip_if_oss()
    strategy = self._get_strategy()

    with strategy.scope():
      first_mid_level_contents = np.ones((4, 4))
      first_mid_level_optimizer = optimizer(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=4,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v1.TPUEmbeddingV0(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build()

    cpu_mid_level_optimizer = optimizer(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(
        feature_config, cpu_mid_level_optimizer)
    cpu_mid_level.build()

    tpu_checkpoint = util.Checkpoint(model=first_mid_level)
    tpu_checkpoint.save(self._get_tmpdir('save-tpu', 'save'))
    tpu_variables = checkpoint_utils.list_variables(
        self._get_tmpdir('save-tpu'))

    cpu_checkpoint = util.Checkpoint(model=cpu_mid_level)
    cpu_checkpoint.save(self._get_tmpdir('save-cpu', 'save'))
    cpu_variables = checkpoint_utils.list_variables(
        self._get_tmpdir('save-cpu'))

    self.assertAllEqual(tpu_variables, cpu_variables)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
