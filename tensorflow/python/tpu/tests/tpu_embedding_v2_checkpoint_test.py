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
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu.tests import tpu_embedding_base_test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import util


class TPUEmbeddingCheckpointTest(tpu_embedding_base_test.TPUEmbeddingBaseTest):

  def make_checkpoint_and_get_embedding(self, name, model, num_rows):
    """Saves model to checkpoint name, retrieves embedding variables."""
    checkpoint = util.Checkpoint(model=model)
    checkpoint.save(self._get_tmpdir(name, 'save'))

    # Get the name of the table video variable which should be the only
    # [8, 4] shaped tensor in the checkpoint. Note that we do this
    # as the key can change.
    variables = checkpoint_utils.list_variables(self._get_tmpdir(name))
    variables = [name for name, size in variables if size == [num_rows, 4]]
    if len(variables) != 1:
      raise RuntimeError('Found {} copies of the parameter variable in the '
                         'checkpoint. Exactly one copy exported.'.format(
                             len(variables)))
    return checkpoint_utils.load_variable(self._get_tmpdir(name), variables[0])

  def test_checkpoint_save_retrieves(self):
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)
      first_mid_level.build(64)

    # Ensure that the variables from the first model are loaded.
    first_mid_level._load_variables()

    self.assertAllClose(
        first_mid_level_contents,
        self.make_checkpoint_and_get_embedding('before_load', first_mid_level,
                                               num_rows),
        msg='Checkpoint should contain values from the first api object.')

    # Reinitialize the tpu.
    tpu_strategy_util.initialize_tpu_system(self.resolver)

    with strategy.scope():
      second_mid_level_contents = np.ones((num_rows, 4)) * 2
      second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(second_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)
      second_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, second_mid_level_optimizer)
      second_mid_level.build(64)

    second_mid_level._load_variables()

    # When we load the variables from the second mid level API object to the TPU
    # we expect that checkpointing the first mid level API object will now
    # retrieve the values from the TPU which are now different from the current
    # variables in the first mid level.
    self.assertAllClose(
        second_mid_level_contents,
        self.make_checkpoint_and_get_embedding('after_load', first_mid_level,
                                               num_rows),
        msg='Checkpoint should contain values from the second api object.')

  def test_checkpoint_restore_loads(self):
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    def get_values(mid):
      return ops.convert_to_tensor(
          mid._variables['table']['parameters'].variables[0])

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)
      first_mid_level.build(64)

    first_mid_level._load_variables()

    first_checkpoint = util.Checkpoint(model=first_mid_level)
    first_checkpoint.save(self._get_tmpdir('restore', 'save'))

    tpu_strategy_util.initialize_tpu_system(self.resolver)

    with strategy.scope():
      second_mid_level_contents = np.ones((num_rows, 4)) * 2
      second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(second_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)
      second_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, second_mid_level_optimizer)
      second_mid_level.build(64)

    second_mid_level._load_variables()

    self.assertAllClose(
        second_mid_level_contents,
        get_values(second_mid_level),
        msg='Second mid level api should contain its initial values.',
    )
    # We restore the checkpoint of our first model into our second model.
    # This should load the first mid level API object onto the TPU.
    second_checkpoint = util.Checkpoint(model=second_mid_level)
    second_checkpoint.restore(self._get_tmpdir('restore', 'save-1'))

    # Call retrieve here as a way to check what the TPU contains.
    # Calling the retrieve ops directly might make for a cleaner separation of
    # test and module, though.
    second_mid_level._retrieve_variables()

    self.assertAllClose(
        first_mid_level_contents,
        get_values(second_mid_level),
        msg='Second mid level api should have retrieved the first model values.'
    )

  def test_checkpoint_restore_before_variable_creation(self):
    self.skip_if_oss()

    class TestModule(module.Module):

      def __init__(self, initializer, rows):
        self._initializer = initializer
        self._rows = rows

        table = tpu_embedding_v2_utils.TableConfig(
            vocabulary_size=self._rows,
            dim=4,
            initializer=self._initializer,
            combiner='sum',
            name='table')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(
            table=table, name='feature'),)
        optimizer = tpu_embedding_v2_utils.SGD()

        self.tpu_embedding = tpu_embedding_v2.TPUEmbedding(
            feature_config, optimizer)

      def create_embedding(self):
        # We aren't training so batch_size here doesn't matter.
        self.tpu_embedding.build(64)

    strategy = self._get_strategy()
    with strategy.scope():
      module1 = TestModule(init_ops_v2.Ones(),
                           strategy.num_replicas_in_sync * 2)
      module1.create_embedding()

    checkpoint = util.Checkpoint(test_module=module1)
    checkpoint.save(self._get_tmpdir('restore_before_create', 'save'))

    # Reinitialize the tpu
    strategy = self._get_strategy()

    with strategy.scope():
      module2 = TestModule(init_ops_v2.Zeros(),
                           strategy.num_replicas_in_sync * 2)

    checkpoint = util.Checkpoint(test_module=module2)
    checkpoint.restore(self._get_tmpdir('restore_before_create', 'save-1'))

    with strategy.scope():
      module2.create_embedding()

    def get_values(mid):
      return mid._variables['table']['parameters'].variables[0].numpy()

    self.assertAllClose(
        np.ones((strategy.num_replicas_in_sync * 2, 4)),
        get_values(module2.tpu_embedding))

    # Fetch the values from the TPU to check that they are the same.
    module2.tpu_embedding._retrieve_variables()

    self.assertAllClose(
        np.ones((strategy.num_replicas_in_sync * 2, 4)),
        get_values(module2.tpu_embedding))

  def test_model_export_cpu(self):
    strategy = self._get_strategy()
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build(64)

    cpu_mid_level_optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_v2.TPUEmbedding(feature_config,
                                                  cpu_mid_level_optimizer)

    cpu_mid_level.build(64)

    first_mid_level._load_variables()

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
    num_rows = strategy.num_replicas_in_sync

    with strategy.scope():
      first_mid_level_contents = np.ones((num_rows, 4))
      first_mid_level_optimizer = optimizer(learning_rate=0.1)
      initializer = init_ops_v2.Constant(first_mid_level_contents)

      table = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_rows,
          dim=4,
          initializer=initializer,
          combiner='sum',
          name='table')
      feature_config = (tpu_embedding_v2_utils.FeatureConfig(
          table=table, name='feature'),)

      first_mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config, first_mid_level_optimizer)

      first_mid_level.build(64)

    cpu_mid_level_optimizer = optimizer(learning_rate=0.1)
    cpu_mid_level = tpu_embedding_v2.TPUEmbedding(feature_config,
                                                  cpu_mid_level_optimizer)
    cpu_mid_level.build(64)

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
