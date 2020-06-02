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
"""Tests for TPU Embeddings mid level API on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import os

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_test_lib
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest


FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')
flags.DEFINE_string('model_dir', os.environ.get('TEST_TMPDIR'),
                    'A temporary directory.')


class TPUEmbeddingCheckpointTest(parameterized.TestCase, test.TestCase):

  def setUp(self):
    super(TPUEmbeddingCheckpointTest, self).setUp()
    self.resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    remote.connect_to_cluster(self.resolver)
    tpu_strategy_util.initialize_tpu_system(self.resolver)
    self.strategy = tpu_strategy.TPUStrategy(self.resolver)
    self.num_rows = self.strategy.num_replicas_in_sync

    # These tests use two mid level API objects, initialized with different
    # values. These have the same sizes.
    with self.strategy.scope():
      self.first_mid_level_contents = np.ones((self.num_rows, 4))
      self.first_mid_level_optimizer = tpu_embedding_v2_utils.SGD(
          learning_rate=0.1)
      self.first_mid_level = self.build_mid_level(
          self.first_mid_level_contents, self.first_mid_level_optimizer)

      self.second_mid_level_contents = np.ones((self.num_rows, 4)) * 2
      self.second_mid_level_optimizer = tpu_embedding_v2_utils.SGD(
          learning_rate=0.1)
      self.second_mid_level = self.build_mid_level(
          self.second_mid_level_contents, self.second_mid_level_optimizer,
          initialize_tpu_embedding=False)

    self.cpu_mid_level_optimizer = tpu_embedding_v2_utils.SGD(
        learning_rate=0.1)
    self.cpu_mid_level = self.build_mid_level(
        self.second_mid_level_contents, self.cpu_mid_level_optimizer)

  def tearDown(self):
    tpu_strategy_util.shutdown_tpu_system(self.resolver)
    super(TPUEmbeddingCheckpointTest, self).tearDown()

  def test_checkpoint_save_retrieves(self):
    # Ensure that the variables from the first model are loaded.
    self.first_mid_level._load_variables()

    self.assertAllClose(
        self.first_mid_level_contents,
        self.make_checkpoint_and_get_embedding('before_load',
                                               self.first_mid_level),
        msg='Checkpoint should contain values from the first api object.')

    self.second_mid_level._load_variables()

    # When we load the variables from the second mid level API object to the TPU
    # we expect that checkpointing the first mid level API object will now
    # retrieve the values from the TPU which are now different from the current
    # variables in the first mid level.
    self.assertAllClose(
        self.second_mid_level_contents,
        self.make_checkpoint_and_get_embedding('after_load',
                                               self.first_mid_level),
        msg='Checkpoint should contain values from the second api object.')

  def test_checkpoint_restore_loads(self):

    def get_values(mid):
      return ops.convert_to_tensor(
          mid._variables['table']['parameters'].variables[0])

    self.first_mid_level._load_variables()

    first_checkpoint = util.Checkpoint(model=self.first_mid_level)
    first_checkpoint.save(_get_tmpdir('restore', 'save'))

    # Checkpoint now has values from first_mid_level. See first assert in
    # test_checkpoint_save_retrieves.

    self.second_mid_level._load_variables()

    self.assertAllClose(
        self.second_mid_level_contents,
        get_values(self.second_mid_level),
        msg='Second mid level api should contain its initial values.',
    )

    # We restore the checkpoint of our first model into our second model.
    # This should load the first mid level API object onto the TPU.
    second_checkpoint = util.Checkpoint(model=self.second_mid_level)
    second_checkpoint.restore(_get_tmpdir('restore', 'save-1'))

    # Call retrieve here as a way to check what the TPU contains contains.
    # Calling the retrieve ops directly might make for a cleaner separation of
    # test and module, though.
    self.second_mid_level._retrieve_variables()

    self.assertAllClose(
        self.first_mid_level_contents,
        get_values(self.second_mid_level),
        msg='Second mid level api should have retrieved the first model values.'
    )

  def build_mid_level(self, embedding_values, optimizer,
                      initialize_tpu_embedding=True):
    """Creates an embedding api object initialized to embedding_values."""
    initializer = init_ops_v2.Constant(embedding_values)

    table = tpu_embedding_v2_utils.TableConfig(
        vocabulary_size=self.num_rows, dim=4, initializer=initializer,
        combiner='sum', name='table')
    feature_config = (tpu_embedding_v2_utils.FeatureConfig(
        table=table, name='feature'),)

    # batch_size here does not matter as we aren't traininig in any of these
    # tests.
    return tpu_embedding_v2.TPUEmbedding(
        feature_config, 64, optimizer,
        initialize_tpu_embedding=initialize_tpu_embedding)

  def make_checkpoint_and_get_embedding(self, name, model):
    """Saves model to checkpoint name, retrieves embedding variables."""
    checkpoint = util.Checkpoint(model=model)
    checkpoint.save(_get_tmpdir(name, 'save'))

    # Get the name of the parameters variable which should be the only
    # [self.num_rows, 4] shaped tensor in the checkpoint. Note that we do this
    # as the key can change.
    variables = checkpoint_utils.list_variables(_get_tmpdir(name))
    variables = [name for name, size in variables if size == [self.num_rows, 4]]
    if len(variables) != 1:
      raise RuntimeError('Found {} copies of the parameter variable in the '
                         'checkpoint. Exactly one copy exported.'.format(
                             len(variables)))
    return checkpoint_utils.load_variable(_get_tmpdir(name), variables[0])

  def test_model_export_cpu(self):
    self.first_mid_level._load_variables()

    tpu_checkpoint = util.Checkpoint(model=self.first_mid_level)
    tpu_checkpoint.save(_get_tmpdir('export_cpu', 'save'))

    # We restore the checkpoint of our tpu mid level onto our cpu mid level.
    cpu_checkpoint = util.Checkpoint(model=self.cpu_mid_level)
    cpu_checkpoint.restore(_get_tmpdir('export_cpu', 'save-1'))

    @def_function.function
    def serve_tensors(features):
      features = tpu_embedding_v2.cpu_embedding_lookup(
          features, None, self.cpu_mid_level.embedding_tables,
          self.cpu_mid_level._feature_config)
      return features[0]

    signatures = {
        'serving_default':
            serve_tensors.get_concrete_function(
                (tensor_spec.TensorSpec(
                    shape=(2,), dtype=dtypes.int32, name='feature'),))}
    save.save(self.cpu_mid_level,
              export_dir=_get_tmpdir('export_cpu', 'exported_model'),
              signatures=signatures)

    imported = load.load(_get_tmpdir('export_cpu', 'exported_model'))
    predict_fn = imported.signatures['serving_default']

    input_feature_value = np.array([1, 0])
    input_batch = (constant_op.constant(input_feature_value,
                                        dtype=dtypes.int32),)
    prediction = predict_fn(*input_batch)['output_0']
    self.assertAllClose(prediction.numpy(),
                        self.first_mid_level_contents[input_feature_value])

  @parameterized.parameters(tpu_embedding_v2_utils.SGD,
                            tpu_embedding_v2_utils.Adagrad,
                            tpu_embedding_v2_utils.Adam)
  def test_check_checkpoint_variable_names_are_same_on_cpu_and_tpu(self,
                                                                   optimizer):
    # Reinitialize the TPU so that we can re-initialize the embeddings with the
    # given optimizer.
    tpu_strategy_util.initialize_tpu_system(self.resolver)
    optimizer = optimizer(learning_rate=0.1)

    with self.strategy.scope():
      tpu_mid_level = self.build_mid_level(
          self.first_mid_level_contents, optimizer)

    tpu_checkpoint = util.Checkpoint(model=tpu_mid_level)
    tpu_checkpoint.save(_get_tmpdir('save-tpu', 'save'))
    tpu_variables = checkpoint_utils.list_variables(_get_tmpdir('save-tpu'))

    cpu_mid_level = self.build_mid_level(
        self.first_mid_level_contents, optimizer)

    cpu_checkpoint = util.Checkpoint(model=cpu_mid_level)
    cpu_checkpoint.save(_get_tmpdir('save-cpu', 'save'))
    cpu_variables = checkpoint_utils.list_variables(_get_tmpdir('save-cpu'))

    self.assertAllEqual(tpu_variables, cpu_variables)


class TPUEmbeddingTest(parameterized.TestCase,
                       tpu_embedding_v2_test_lib.EmbeddingTestBase):

  def setUp(self):
    super(TPUEmbeddingTest, self).setUp()
    self._create_initial_data()
    self.resolver = None

  def tearDown(self):
    if self.resolver:
      tpu_strategy_util.shutdown_tpu_system(self.resolver)
    super(TPUEmbeddingTest, self).tearDown()

  def test_tables_with_same_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Multiple tables with name table found.'):
      with self._get_strategy().scope():
        tpu_embedding_v2.TPUEmbedding(
            (tpu_embedding_v2_utils.FeatureConfig(
                table=tpu_embedding_v2_utils.TableConfig(
                    name='table',
                    vocabulary_size=4,
                    dim=2,
                    initializer=self.initializer,),
                name='watched'),
             tpu_embedding_v2_utils.FeatureConfig(
                 table=tpu_embedding_v2_utils.TableConfig(
                     name='table',
                     vocabulary_size=4,
                     dim=2,
                     initializer=self.initializer),
                 name='favorited')),
            self.batch_size,
            tpu_embedding_v2_utils.SGD(learning_rate=0.1))

  def test_unsupported_optimizer(self):
    with self.assertRaisesRegex(
        ValueError, 'is an unsupported optimizer class.'):
      with self._get_strategy().scope():
        tpu_embedding_v2.TPUEmbedding(
            self.feature_config, self.batch_size,
            tpu_embedding.AdagradParameters(learning_rate=0.1))

  def test_pass_non_tensor_to_apply_gradients(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    @def_function.function
    def test_apply():
      mid_level_api.apply_gradients((1, 2, 3))

    with self.assertRaisesRegex(ValueError, 'Expected Tensor.'):
      strategy.run(test_apply)

  def test_pass_different_structure_to_apply_gradients(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    @def_function.function
    def test_apply():
      # This should be a tuple as feature_config is a tuple of 3 configs.
      mid_level_api.apply_gradients([1, 2, 3])

    with self.assertRaisesRegex(
        TypeError,
        'The two structures don\'t have the same nested structure.'):
      strategy.run(test_apply)

  def test_pass_none_to_apply_gradients(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    data = next(iter(strategy.experimental_distribute_dataset(dataset)))

    @def_function.function
    def embedding_and_set_gradients(data):
      mid_level_api.enqueue(data)
      def tpu_fn():
        results = mid_level_api.dequeue()
        mid_level_api.apply_gradients((None, None,
                                       array_ops.ones_like(results[2])))
        return results
      return strategy.run(tpu_fn)

    @def_function.function
    def embedding_only(data):
      mid_level_api.enqueue(data, training=False)
      def tpu_fn():
        return mid_level_api.dequeue()
      return strategy.run(tpu_fn)

    first = self._get_replica_numpy(
        embedding_and_set_gradients(data), strategy, 0)
    second = self._get_replica_numpy(embedding_only(data), strategy, 0)

    # First two features should be the same as None gradient was applied.
    # Third feature had gradient of 1 passed in from each core.
    # Each core received the same ids per core and returned the following batch:
    # [ row 3, row 0 + row 1 + row 2 ]
    # so gradient update was (learning rate = 0.1):
    #   row 0: -1/3*0.1
    #   row 1: -1/3*0.1
    #   row 2: -1/3*0.1
    #   row 3: -1*0.1
    # There is a factor of num_replicas because each replica gave an update.

    num_replicas = strategy.num_replicas_in_sync
    update = ([[0.0]], [[0.0]],
              [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
    golden = tuple([feature-np.array(up) for feature, up in zip(first, update)])

    self.assertAllClose(golden, second)

  def _get_strategy(self):
    self.resolver = tpu_cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    remote.connect_to_cluster(self.resolver)
    tpu_strategy_util.initialize_tpu_system(self.resolver)
    return tpu_strategy.TPUStrategy(self.resolver)

  def test_dequeue_on_cpu(self):
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.dequeue()

  def test_enqueue_on_cpu(self):
    mid_level_api = self._create_mid_level()
    features = {
        'watched': sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices,
            values=self.feature_watched_values,
            dense_shape=[2, 2])}
    with self.assertRaises(RuntimeError):
      mid_level_api.enqueue(features)

  def test_apply_gradients_on_cpu(self):
    mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.enqueue(None)

  def test_get_embedding_tables_on_cpu(self):
    mid_level_api = self._create_mid_level()
    self.assertEqual(
        set(mid_level_api.embedding_tables.keys()),
        set([self.table_video, self.table_user]))

  def test_get_embedding_tables_on_tpu(self):
    with self._get_strategy().scope():
      mid_level_api = self._create_mid_level()
    with self.assertRaises(RuntimeError):
      mid_level_api.embedding_tables()

  def test_enqueue_weight_for_dense_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    input_fn = self._create_dense_input_fn(strategy, include_weights=True)
    dist = strategy.experimental_distribute_datasets_from_function(input_fn)
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      features, weights = next(dist_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(ValueError, 'Weight specified for dense input'):
      test_fn()

  def test_enqueue_wrong_weight_type_for_sparse_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    ragged = self._create_ragged_dataset(strategy, include_weights=True)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))
    ragged_iter = iter(strategy.experimental_distribute_dataset(ragged))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      features = next(sparse_iter)
      _, weights = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is SparseTensor.'):
      test_fn()

  def test_enqueue_wrong_weight_type_for_ragged_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    ragged = self._create_ragged_dataset(strategy)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))
    ragged_iter = iter(strategy.experimental_distribute_dataset(ragged))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      _, weights = next(sparse_iter)
      features = next(ragged_iter)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'which does not match type input which is RaggedTensor.'):
      test_fn()

  def test_enqueue_sparse_and_ragged(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    ragged = self._create_ragged_dataset(strategy)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))
    ragged_iter = iter(strategy.experimental_distribute_dataset(ragged))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      sparse_features = next(sparse_iter)
      ragged_features = next(ragged_iter)
      features = (sparse_features[0], ragged_features[1], sparse_features[2])
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    with self.assertRaisesRegex(
        ValueError, 'Found both SparseTensors and RaggedTensors'):
      test_fn()

  def test_enqueue_incorrect_structure_for_features(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      features = next(sparse_iter)
      features = (features[0],)
      mid_level_api.enqueue(features, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_fn()

  def test_enqueue_incorrect_structure_for_weights(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy, include_weights=True)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      features, weights = next(sparse_iter)
      weights = (weights[0],)
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(step)

    # The error here is raised from nest.assert_same_structure
    with self.assertRaises(ValueError):
      test_fn()

  def test_enqueue_ragged_tensor(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    sparse = self._create_sparse_dataset(strategy)
    ragged = self._create_ragged_dataset(strategy)
    sparse_iter = iter(strategy.experimental_distribute_dataset(sparse))
    ragged_iter = iter(strategy.experimental_distribute_dataset(ragged))

    @def_function.function
    def test_fn():
      def get_activations():
        return mid_level_api.dequeue()

      sparse_features = next(sparse_iter)
      ragged_features = next(ragged_iter)
      mid_level_api.enqueue(sparse_features, training=False)
      sparse_activations = strategy.run(get_activations)
      mid_level_api.enqueue(ragged_features, training=False)
      ragged_activations = strategy.run(get_activations)
      return sparse_activations, ragged_activations

    sparse_activations, ragged_activations = test_fn()

    # Extact per core numpy arrays and check that both sparse and ragged have
    # the same results.
    sparse0 = self._get_replica_numpy(sparse_activations, strategy, 0)
    ragged0 = self._get_replica_numpy(ragged_activations, strategy, 0)
    self.assertAllClose(sparse0, ragged0)

  @parameterized.parameters(True, False)
  def test_enqueue_with_weights(self, ragged):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    weight = 0.5
    if ragged:
      dataset = self._create_ragged_dataset(strategy, include_weights=True,
                                            weight=weight)
    else:
      dataset = self._create_sparse_dataset(strategy, include_weights=True,
                                            weight=weight)

    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def enqueue_and_get(features, weights):
      def get_activations():
        return mid_level_api.dequeue()
      mid_level_api.enqueue(features, weights=weights, training=False)
      return strategy.run(get_activations)

    features, weights = next(dataset_iter)
    # Replace the weight for the second feature by None to test.
    weights = (weights[0], None, weights[2])

    no_weights_activations = enqueue_and_get(features, weights=None)
    weights_activations = enqueue_and_get(features, weights=weights)

    # Extact per core numpy arrays.
    no_weights0 = self._get_replica_numpy(no_weights_activations, strategy, 0)
    weights0 = self._get_replica_numpy(weights_activations, strategy, 0)
    # videos table has sum combiner and users table has mean combiner.
    # i.e. users table lookups isn't affected by the weights as all the weights
    # are the same.
    # Tuple entry 0 and 1 are the watched and favorited features from the videos
    # table and entry 2 is the friends feature from the users table.
    # Note that None was passed as a weight for entry 1 so weight should have no
    # effect.
    weight = (0.5, 1.0, 1.0)
    golden = tuple([no_weight * w for no_weight, w in zip(no_weights0, weight)])

    self.assertAllClose(golden, weights0)

  def test_enqueue_with_outside_compilation(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def enqueue_with_outside_compilation(data):
      def get_activations(features):
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(data,))

    @def_function.function
    def enqueue_without_outside_compilation(data):
      def get_activations():
        return mid_level_api.dequeue()
      mid_level_api.enqueue(data, training=False)
      return strategy.run(get_activations)

    features = next(dataset_iter)

    activations_oc = enqueue_with_outside_compilation(features)
    activations = enqueue_without_outside_compilation(features)

    # Extact per core numpy arrays.
    activations_oc0 = self._get_replica_numpy(activations_oc, strategy, 0)
    activations0 = self._get_replica_numpy(activations, strategy, 0)

    self.assertAllClose(activations_oc0, activations0)

  def test_enqueue_with_outside_compilation_in_control_flow(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))

    # This is one way to force the enqueue in some control flow. @tf.functions
    # aren't inlined in the calling tf.function. An alternative would be to
    # place the enqueue in a switch_v2 or something similar.
    @def_function.function
    def enqueue_fn(features):
      mid_level_api.enqueue(features, training=False)

    @def_function.function
    def enqueue_with_outside_compilation():
      def get_activations(features):
        enqueue_fn(features)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(next(dataset_iter),))

    with self.assertRaisesRegex(
        RuntimeError,
        'does not match graph which contains TPUReplicateContext'):
      enqueue_with_outside_compilation()

  def test_enqueue_with_outside_compilation_non_direct_input(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def enqueue_with_outside_compilation():
      def get_activations(features):
        # This inserts a mul operation on the TPU to trigger the direct input
        # error.
        features = (features[0]*2, features[1]*2, features[2]*2)
        mid_level_api.enqueue(features, training=False)
        return mid_level_api.dequeue()
      return strategy.run(get_activations, args=(next(dataset_iter),))

    with self.assertRaisesRegex(
        ValueError, 'which does not have the `_tpu_input_identity` attr'):
      enqueue_with_outside_compilation()

  def test_enqueue_with_outside_compilation_auto_mode(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')
    dataset = self._create_sparse_dataset(strategy)
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def enqueue_with_no_gradient_apply(data):
      def get_activations(features):
        # Note the lack of setting training=False, so training defaults to true
        # here even though we don't have apply gradients.
        # We detect the correct mode based on which ops exist that share the
        # same 'name'.
        mid_level_api.enqueue(features, name='call1')
        return mid_level_api.dequeue(name='call1')
      return strategy.run(get_activations, args=(data,))

    @def_function.function
    def enqueue_with_gradient_apply(data):
      def get_activations(features):
        mid_level_api.enqueue(features, name='call2')
        activations = mid_level_api.dequeue(name='call2')
        # Apply an all ones gradient
        gradients = nest.map_structure(array_ops.ones_like, activations)
        mid_level_api.apply_gradients(gradients, name='call2')
        return activations
      return strategy.run(get_activations, args=(data,))

    data = next(dataset_iter)
    before_gradient_apply = enqueue_with_gradient_apply(data)
    after_gradient_apply = enqueue_with_no_gradient_apply(data)
    before_gradient_apply0 = self._get_replica_numpy(before_gradient_apply,
                                                     strategy, 0)
    after_gradient_apply0 = self._get_replica_numpy(after_gradient_apply,
                                                    strategy, 0)

    num_replicas = strategy.num_replicas_in_sync
    # We are passing a gradient of 1 for all lookups, optimizer is SGD with a
    # learning rate of 0.1. Feature 0 and 1 are looked up with a sum combiner
    # with the following ids:
    # Feature 0: [0, 0, 1], [0, 1, 1], ... repeated over num_replicas
    # Feature 1: [0, 1, 1], [0, 0, 1], ... repeated over num_replicas
    # i.e. Row 0 and 1 were looked up 3*num_replicas times over all cores and as
    # the gradient is 1, the accumulated gradient is 3*num_replicas for each
    # position in row 0 and 1 in table.
    #
    # See comments in test_pass_none_to_apply_gradients for the update to
    # Feature 2 and its table.
    # The *2 in the next tests are because those rows have 2 lookups vs
    # the 1 lookup in the other row.
    update = ([[0.3 * num_replicas], [0.3 * num_replicas * 2]],
              [[0.3 * num_replicas * 2], [0.3 * num_replicas]],
              [[0.1 * num_replicas], [0.1 / 3 * num_replicas]])
    golden = tuple([before - np.array(up) for before, up in
                    zip(before_gradient_apply0, update)])

    self.assertAllClose(golden, after_gradient_apply0)

  def _create_strategy_and_mid_level(self, optimizer_name):
    strategy = self._get_strategy()

    with strategy.scope():
      if optimizer_name == 'sgd':
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
      elif optimizer_name == 'adagrad':
        optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
      elif optimizer_name == 'adam':
        optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
      else:
        raise ValueError('optimizer is not recognized: ', optimizer_name)
      mid_level_api = self._create_mid_level(optimizer=optimizer)

    return strategy, mid_level_api, optimizer

  @parameterized.parameters(
      *itertools.product(
          ['sgd', 'adagrad', 'adam'],
          [True, False]))
  def test_embedding(self, optimizer_name, training):
    strategy, mid_level_api, optimizer = (
        self._create_strategy_and_mid_level(optimizer_name))

    dataset = self._create_sparse_dataset(strategy)
    dist = strategy.experimental_distribute_dataset(dataset)
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():

      def step():
        """Create and run computation that returns the embedding activations."""
        if not training:
          activations = mid_level_api.dequeue()
          total_loss = _get_total_loss_tensor(activations)
          ret_val = [total_loss] + list(activations)
          return ret_val
        else:
          with backprop.GradientTape() as tape:
            activations = mid_level_api.dequeue()
            tape.watch(activations)
            total_loss = _get_total_loss_tensor(activations)
            loss_per_replica = total_loss / strategy.num_replicas_in_sync
          gradients = tape.gradient(loss_per_replica, activations)
          mid_level_api.apply_gradients(gradients)
        ret_val = [total_loss] + list(activations)
        return ret_val

      mid_level_api.enqueue(next(dist_iter), training=training)
      result = strategy.run(step)
      return result

    # Run model.
    shard_out_val = test_fn()

    # Retrieve TPU weights to CPU.
    mid_level_api._retrieve_variables()

    # Compute sparse tensors for global batch.
    input_data = next(iter(self._create_sparse_dataset(strategy)))

    # Check results.
    self._check_results(strategy, shard_out_val, training, input_data,
                        mid_level_api._variables,
                        optimizer)

  def _create_mid_level(self, optimizer=None):
    # Create `TPUEmbedding` object.
    if optimizer is None:
      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)

    num_replicas = (
        distribution_strategy_context.get_strategy().num_replicas_in_sync)
    return tpu_embedding_v2.TPUEmbedding(
        feature_config=self.feature_config,
        batch_size=self.batch_size * num_replicas,
        optimizer=optimizer)

  def _create_sparse_dataset(self, strategy, include_weights=False, weight=0.5):
    # Create dataset for enqueue operation
    sparse_features = (
        sparse_tensor.SparseTensor(
            indices=self.feature_watched_indices,
            values=self.feature_watched_values,
            dense_shape=[self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_favorited_indices,
            values=self.feature_favorited_values,
            dense_shape=[self.data_batch_size, 2]),
        sparse_tensor.SparseTensor(
            indices=self.feature_friends_indices,
            values=self.feature_friends_values,
            dense_shape=[self.data_batch_size, 3]))
    if include_weights:
      weights = []
      for sparse in sparse_features:
        values = (
            array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight)
        weights.append(sparse_tensor.SparseTensor(
            indices=sparse.indices,
            values=values,
            dense_shape=sparse.dense_shape))
      sparse_features = (sparse_features, tuple(weights))

    dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
    # Create dataset for enqueue operation
    ragged_features = (
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_watched_row_lengths,
            values=self.feature_watched_values),
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_favorited_row_lengths,
            values=self.feature_favorited_values),
        ragged_tensor.RaggedTensor.from_row_lengths(
            row_lengths=self.feature_friends_row_lengths,
            values=self.feature_friends_values))
    if include_weights:
      weights = []
      for ragged in ragged_features:
        weights.append(ragged.with_values(
            array_ops.ones_like(ragged.values, dtype=dtypes.float32) * weight))
      ragged_features = (ragged_features, tuple(weights))

    dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)

    # Data is batched to self.data_batch_size, rebatch to global batch size.
    return dataset.unbatch().repeat().batch(
        self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

  def _create_dense_input_fn(self, strategy, include_weights=False, weight=0.5):

    def input_fn(ctx):
      del ctx
      features = (
          constant_op.constant(self.feature_watched_values[-2:],
                               dtype=dtypes.int32),
          constant_op.constant(self.feature_favorited_values[-2:],
                               dtype=dtypes.int32),
          constant_op.constant(self.feature_friends_values[-2:],
                               dtype=dtypes.int32))
      if include_weights:
        weights = [array_ops.ones_like(t, dtype=dtypes.float32) * weight
                   for t in features]
        features = (features, tuple(weights))
      return dataset_ops.DatasetV2.from_tensors(features).repeat()

    return input_fn

  def _check_results(self, strategy, shard_out_val, training, input_data,
                     table_to_variable, optimizer):
    num_replicas = strategy.num_replicas_in_sync

    # Unpack the values `strategy.run()` returns.
    loss = _unpack(strategy, shard_out_val[0])
    activation_watched = _unpack(strategy, shard_out_val[1])
    activation_favorited = _unpack(strategy, shard_out_val[2])
    activation_friends = _unpack(strategy, shard_out_val[3])

    # Core 0:
    # Calculate the values of embedding activations.
    activation_watched_gold0 = np.array([[0, 1, 2, 3], [4, 6, 8, 10]])
    activation_favorited_gold0 = np.array([[4, 6, 8, 10], [4, 5, 6, 7]])
    # Second row of `activation_friends_gold0` is the mean of the following.
    # row 0: 0 1
    # row 1: 2 3
    # row 2: 4 5
    activation_friends_gold0 = np.array([[6, 7], [2, 3]])

    loss_gold0 = _compute_loss(activation_watched_gold0,
                               activation_favorited_gold0,
                               activation_friends_gold0)

    # Add on values from other cores:
    # Activations for watched are an alternating sequence of
    # activation_watched_gold0 and activation_favorited_gold0.
    # For favorited it is the same but in the opposite order.
    activation_watched_gold = np.concatenate(
        (np.concatenate((np.expand_dims(activation_watched_gold0, axis=0),) *
                        (num_replicas // 2)),
         np.concatenate((np.expand_dims(activation_favorited_gold0, axis=0),) *
                        (num_replicas // 2))),
        axis=1).reshape([self.batch_size * num_replicas, 4])
    activation_favorited_gold = np.concatenate(
        (activation_watched_gold[self.batch_size:,],
         activation_watched_gold[0:self.batch_size,]))
    activation_friends_gold = np.concatenate(
        (activation_friends_gold0,) * num_replicas)

    loss_gold = [loss_gold0] * num_replicas

    # Test values.
    self.assertAllClose(activation_watched_gold, activation_watched)
    self.assertAllClose(activation_favorited_gold, activation_favorited)
    self.assertAllClose(activation_friends_gold, activation_friends)

    self.assertAllClose(loss_gold, loss)

    embedding_table_video_before = np.copy(
        np.reshape(self.embedding_values, [8, 4]))
    embedding_table_user_before = np.copy(
        np.reshape(self.embedding_values, [16, 2]))

    global_batch_size = self.batch_size * num_replicas
    if training:
      gradient_wrt_watched_gold = (2 * activation_watched_gold /
                                   global_batch_size)
      gradient_wrt_favorited_gold = (2 * activation_favorited_gold /
                                     global_batch_size)
      gradient_wrt_friends_gold = (2 * activation_friends_gold /
                                   global_batch_size)

      # Calculate gradients wrt embedding tables.
      gradients_wrt_user = (
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_friends_gold,
              embedding_table_user_before, input_data[2].indices.numpy(),
              input_data[2].values.numpy(), self.table_user.combiner))
      gradients_wrt_video = (
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_favorited_gold,
              embedding_table_video_before, input_data[1].indices.numpy(),
              input_data[1].values.numpy(), self.table_video.combiner) +
          _compute_gradients_wrt_embedding_table(
              global_batch_size, gradient_wrt_watched_gold,
              embedding_table_video_before, input_data[0].indices.numpy(),
              input_data[0].values.numpy(), self.table_video.combiner))

      self._check_embedding_and_slot_variables(embedding_table_user_before,
                                               gradients_wrt_user,
                                               embedding_table_video_before,
                                               gradients_wrt_video,
                                               optimizer,
                                               table_to_variable)

  def _check_embedding_and_slot_variables(self, embedding_table_user_before,
                                          gradients_wrt_user,
                                          embedding_table_video_before,
                                          gradients_wrt_video,
                                          optimizer,
                                          table_to_variable):
    if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
      check_fn = self._check_embedding_and_slot_variables_for_sgd
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
      check_fn = self._check_embedding_and_slot_variables_for_adagrad
    elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
      check_fn = self._check_embedding_and_slot_variables_for_adam
    else:
      raise ValueError('optimizer is not recognized: ', type(optimizer))
    check_fn(embedding_table_user_before, gradients_wrt_user,
             optimizer, table_to_variable[self.table_user.name])
    check_fn(embedding_table_video_before, gradients_wrt_video,
             optimizer, table_to_variable[self.table_video.name])

  def _check_embedding_and_slot_variables_for_sgd(self, embedding_table_before,
                                                  gradients,
                                                  optimizer,
                                                  variables):
    embedding_table = np.copy(embedding_table_before)
    embedding_table -= optimizer.learning_rate * np.sum(gradients, axis=0)
    self.assertAllClose(_get_variable(variables['parameters']).numpy(),
                        embedding_table)

  def _check_embedding_and_slot_variables_for_adagrad(self,
                                                      embedding_table_before,
                                                      gradients,
                                                      optimizer,
                                                      variable):
    embedding_table = np.copy(embedding_table_before)
    accumulator = (
        optimizer.initial_accumulator_value + np.sum(gradients, axis=0)**2)
    embedding_table -= (
        optimizer.learning_rate * np.sum(gradients, axis=0) /
        np.sqrt(accumulator))
    self.assertAllClose(_get_variable(variable['parameters']).numpy(),
                        embedding_table)
    self.assertAllClose(_get_variable(variable['accumulators']).numpy(),
                        accumulator)

  def _check_embedding_and_slot_variables_for_adam(self, embedding_table_before,
                                                   gradients,
                                                   optimizer,
                                                   variable):
    embedding_table = np.copy(embedding_table_before)
    g = np.sum(gradients, axis=0)
    v = g**2 * (1 - optimizer.beta_2)
    m = g * (1 - optimizer.beta_1)
    epsilon = optimizer.epsilon
    # TPU Embeddings don't have the LR decay factor for Adam.
    lr_modifier = 1
    embedding_table -= (
        m * optimizer.learning_rate * lr_modifier / (np.sqrt(v) + epsilon))
    self.assertAllClose(_get_variable(variable['parameters']).numpy(),
                        embedding_table, rtol=1e-4)
    self.assertAllClose(_get_variable(variable['momenta']).numpy(),
                        m, rtol=1e-4)
    self.assertAllClose(_get_variable(variable['velocities']).numpy(),
                        v, rtol=1e-4)

  def _get_replica_numpy(self, structured, strategy, replica_id):
    def select_replica(x):
      x = strategy.experimental_local_results(x)
      if len(x) == 1:
        return x.numpy()
      return x[replica_id].numpy()
    return nest.map_structure(select_replica, structured)

  def test_dense_lookup(self):
    strategy, mid_level_api, _ = self._create_strategy_and_mid_level('sgd')

    input_fn = self._create_dense_input_fn(strategy)
    dist = strategy.experimental_distribute_datasets_from_function(input_fn)
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      def step():
        return mid_level_api.dequeue()

      mid_level_api.enqueue(next(dist_iter), training=False)
      return strategy.run(step)

    # Run model.
    shard0 = self._get_replica_numpy(test_fn(), strategy, 0)

    # embedding_values is a linear list, so we reshape to match the correct
    # shape of the corresponding table before performing the lookup.
    numpy_videos = np.reshape(self.embedding_values, (8, 4))
    numpy_users = np.reshape(self.embedding_values, (16, 2))
    golden = ((numpy_videos[self.feature_watched_values[-2:]],
               numpy_videos[self.feature_favorited_values[-2:]],
               numpy_users[self.feature_friends_values[-2:]]))
    self.assertAllClose(shard0, golden)

  def test_variable_learning_rate(self):
    num_steps = 10
    num_steps_float = float(num_steps)
    starting_lr = 1.0
    ending_lr = 0.5

    strategy = self._get_strategy()
    num_replicas = strategy.num_replicas_in_sync

    # Create model with Keras.
    with strategy.scope():
      step_counter = tf_variables.Variable(0.0, dtypes.float32)

      def lr_function():
        return gen_math_ops.maximum(
            ending_lr,
            starting_lr + ((ending_lr - starting_lr) * step_counter) /
            num_steps_float)

      optimizer = tpu_embedding_v2_utils.SGD(learning_rate=lr_function)
      table_config = tpu_embedding_v2_utils.TableConfig(
          vocabulary_size=num_replicas,
          dim=4,
          initializer=init_ops_v2.Constant(np.zeros((num_replicas, 4))),
          combiner='sum', name='table')
      mid_level_api = tpu_embedding_v2.TPUEmbedding(
          feature_config={
              'feature': tpu_embedding_v2_utils.FeatureConfig(
                  table=table_config, name='feature')},
          batch_size=num_replicas,
          optimizer=optimizer)

    feature = {'feature': constant_op.constant([0], dtype=dtypes.int32)}

    def input_fn(ctx):
      del ctx
      return dataset_ops.DatasetV2.from_tensors(feature).repeat()
    dist = strategy.experimental_distribute_datasets_from_function(input_fn)
    dist_iter = iter(dist)

    @def_function.function
    def test_fn():
      def step():
        with backprop.GradientTape() as tape:
          activations = mid_level_api.dequeue()
          tape.watch(activations)
          result = math_ops.reduce_sum(activations['feature'])
          loss = result / num_replicas
        grads = tape.gradient(loss, activations)
        mid_level_api.apply_gradients(grads)
        return activations['feature']

      mid_level_api.enqueue(next(dist_iter), training=True)
      return strategy.run(step)

    # Run model.
    results = []
    for _ in range(num_steps):
      result = test_fn()
      results.append(_unpack(strategy, result))
      step_counter.assign_add(1.0)

    # Table is 2 elements wide, per-replica batch size of 1, with id 0.
    # Loss for the gradient is the sum of the entries divided by the number of
    # replicas. Thus the per replica gradient is 1/#of replicas for row 0 and no
    # other updates. The reduced gradient is therefore 1.
    # Learning rate schedule over num_steps steps:
    # 1.0 0.95 0.9 0.85 0.8 ...
    # Since use SGD and the gradient is one, the first row of the table is
    # [0, 0] [-1.0, -1.0] [-1.95, -1.95] [-2.85, -2.85] ... (the negative
    # partial sums of the above).

    learning_rates = [starting_lr - (starting_lr - ending_lr) / num_steps * j
                      for j in range(num_steps)]
    cumsum = [sum(learning_rates[0:j]) for j in range(num_steps)]
    goldens = [[[-cumsum[i]] * table_config.dim] * num_replicas
               for i in range(10)]
    self.assertAllClose(results, goldens)

  @parameterized.parameters([True, False])
  def test_optimizer_with_slot_creation_fn(self, use_tpu):
    def slot_creation_fn(table, slot_names):
      slots = {}
      for slot in slot_names:
        slots[slot] = tf_variables.Variable(
            name='{}_{}'.format(table.name, slot),
            initial_value=functools.partial(
                init_ops_v2.Zeros(), shape=table.shape, dtype=dtypes.float32),
            trainable=False)
      return slots
    optimizer = tpu_embedding_v2_utils.Adagrad(
        learning_rate=0.1,
        slot_variable_creation_fn=slot_creation_fn)
    if use_tpu:
      strategy = self._get_strategy()
    else:
      strategy = distribution_strategy_context.get_strategy()
    num_replicas = strategy.num_replicas_in_sync
    with strategy.scope():
      mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config=self.feature_config,
          batch_size=self.batch_size * num_replicas,
          optimizer=optimizer)
    video_accumulator = mid_level._variables['video']['accumulators']
    user_accumulator = mid_level._variables['user']['accumulators']
    if use_tpu:
      # To check the table contents (ensure that it is zero rather than the
      # normal initial accumulator value specified to in the optimizer config),
      # we need to select the underlying table variable on TPU.
      # We only have one shard on Forge.
      video_accumulator = video_accumulator.variables[0]
      user_accumulator = user_accumulator.variables[0]

    self.assertAllClose(video_accumulator.numpy(),
                        np.zeros((self.table_video.vocabulary_size,
                                  self.table_video.dim)))
    self.assertAllClose(user_accumulator.numpy(),
                        np.zeros((self.table_user.vocabulary_size,
                                  self.table_user.dim)))

  def test_optimizer_with_slot_creation_fn_non_partial(self):
    def slot_creation_fn(table, slot_names):
      slots = {}
      for slot in slot_names:
        # Note that we don't pass functools.partial here, so on TPU we can't
        # extract the shape. We expect the error below.
        slots[slot] = tf_variables.Variable(
            name='{}_{}'.format(table.name, slot),
            initial_value=init_ops_v2.Zeros()(shape=table.shape,
                                              dtype=dtypes.float32),
            trainable=False)
      return slots
    optimizer = tpu_embedding_v2_utils.Adagrad(
        learning_rate=0.1,
        slot_variable_creation_fn=slot_creation_fn)
    strategy = self._get_strategy()
    num_replicas = strategy.num_replicas_in_sync
    with strategy.scope():
      with self.assertRaisesRegex(ValueError,
                                  'Unable to extract initializer function'):
        tpu_embedding_v2.TPUEmbedding(
            feature_config=self.feature_config,
            batch_size=self.batch_size*num_replicas,
            optimizer=optimizer)

  def test_sequence_embeddings(self):
    feature_config = (
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='watched',
            max_sequence_length=2),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_video, name='favorited',
            max_sequence_length=2),
        tpu_embedding_v2_utils.FeatureConfig(
            table=self.table_user, name='friends',
            max_sequence_length=3))
    optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
    strategy = self._get_strategy()
    num_replicas = strategy.num_replicas_in_sync
    with strategy.scope():
      mid_level = tpu_embedding_v2.TPUEmbedding(
          feature_config=feature_config,
          batch_size=self.batch_size * num_replicas,
          optimizer=optimizer)

    dataset = self._create_sparse_dataset(strategy)
    data = next(iter(strategy.experimental_distribute_dataset(dataset)))

    @def_function.function
    def embedding_and_set_gradients(data):
      def tpu_fn():
        activations = mid_level.dequeue()
        mid_level.apply_gradients(nest.map_structure(array_ops.ones_like,
                                                     activations))
        return activations
      mid_level.enqueue(data)
      return strategy.run(tpu_fn)

    @def_function.function
    def embedding_only(data):
      def tpu_fn():
        return mid_level.dequeue()
      mid_level.enqueue(data)
      return strategy.run(tpu_fn)

    # Only check core 0.
    before_update = self._get_replica_numpy(
        embedding_and_set_gradients(data), strategy, 0)
    after_update = self._get_replica_numpy(embedding_only(data), strategy, 0)

    # For videos table, row 0 and row 1 are looked up 3*num_replicas times as
    # they occur 3 times per replica (considering the features 0 and 1 which are
    # both looked up in the videos table).
    # Feature 0 has ids [0, 0, 1], [0, 1, 1], ... repeated over num_replicas
    # Feature 1 has ids [0, 1, 1], [0, 0, 1], ... repeated over num_replicas
    # This means that both rows 0 and 1 get a -0.1*3*num_replicas update
    # For users table, each row is looked up twice:
    # Feature 2 has ids [3, 0, 1, 2], .. repeated over num_replicas
    # This means that we get a -0.1*num_replicas update to the third feature.

    # In general this means that after the update, if we lookup feature 0 and 1
    # the values will be 0.3*num_replicas lower per entry and for feature 2 they
    # will be 0.1*num_replicas lower.
    # The one issue that that these lookups contain padding values.
    # For core 0, we get the first 2 elements of the 4 element batch.
    # For feature 0, the indices are [[0, 0], [1, 0], [1, 1]] with max sequence
    # length of 2, which means that [0, 1] will be 0s.
    # For feature 1, the indices are [[0, 0], [0, 1], [1, 0]] with max sequence
    # length of 2, which means that [1, 1] will be 0s.
    # For feature 2, the indices are [[0, 0], [1, 0], [1, 1], [1, 2]] with max
    # sequence length of 3, which means that [0, 1], [0, 2] will be 0s.
    # The following masks represent that so that we only apply the above updates
    # to the non-padding rows:
    masks = (
        np.array([[[1], [0]], [[1], [1]]]),
        np.array([[[1], [1]], [[1], [0]]]),
        np.array([[[1], [0], [0]], [[1], [1], [1]]]))

    per_row_update = (0.3 * num_replicas,
                      0.3 * num_replicas,
                      0.1 * num_replicas)
    golden = tuple([before - update * mask for before, update, mask in
                    zip(before_update, per_row_update, masks)])
    self.assertAllClose(golden, after_update)


def _compute_gradients_wrt_embedding_table(batch_size,
                                           gradient_wrt_activation,
                                           embedding_table,
                                           feature_indices,
                                           feature_values,
                                           combiner,
                                           max_sequence_length=0):
  """Compute gradients wrt embedding_table.

  Args:
    batch_size: `int`, batch size.
    gradient_wrt_activation: `np.array` with shape `batch_size` by
      embedding `dimension`.
    embedding_table: `np.array` with shape `vocabulary_size` by embedding
      `dimension`.
    feature_indices: `indices` as used to construct `SparseTensor`.
    feature_values: `values` as used to construct `SparseTensor`.
    combiner: `String`, 'mean' or 'sum'.
    max_sequence_length: If non-zero, a sequence feature with the given length.

  Returns:
    Gradients wrt `embedding_table`, an `np.array`s with shape
      `batch_size` by `vocabulary_size` by
      embedding `dimension`.

  Raises:
    ValueError: if `combiner` is not one of 'mean' or 'sum'.
  """
  if combiner not in ('mean', 'sum'):
    raise ValueError('`combiner` must be mean or sum; got {}.'.format(combiner))
  grads = []
  for i in range(batch_size):
    grad = np.zeros_like(embedding_table)
    count = 0
    for (batch_i, seq_index), vocabulary_id in zip(feature_indices,
                                                   feature_values):
      if batch_i == i:
        count += 1
        if max_sequence_length > 0:
          if seq_index < max_sequence_length:
            grad[vocabulary_id, :] += gradient_wrt_activation[i, seq_index, :]
        else:
          grad[vocabulary_id, :] += gradient_wrt_activation[i, :]
    if combiner == 'mean' and not max_sequence_length:
      grad = grad / count
    grads.append(grad)
  return np.stack(grads)


def _unpack(strategy, per_replica_output):
  per_replica_output = strategy.experimental_local_results(per_replica_output)
  per_replica_output = array_ops.concat(per_replica_output, axis=0).numpy()
  return per_replica_output


def _get_total_loss_tensor(activations):
  losses = []
  for activation in activations:
    losses.append(
        math_ops.reduce_mean(
            math_ops.reduce_sum(
                gen_math_ops.squared_difference(activation, 0), 1)))
  total_loss = array_ops.expand_dims_v2(sum(losses), 0)
  return total_loss


def _compute_loss(activation_watched, activation_favorited, activation_friends):
  watched_loss = np.mean(np.sum(activation_watched**2, axis=1))
  if len(activation_favorited.shape) == 2:
    favorited_loss = np.mean(np.sum(activation_favorited**2, axis=1))
  else:
    favorited_loss = np.mean(np.sum(activation_favorited**2, axis=(1, 2)))
  if len(activation_friends.shape) == 2:
    friends_loss = np.mean(np.sum(activation_friends**2, axis=1))
  else:
    friends_loss = np.mean(np.sum(activation_friends**2, axis=(1, 2)))
  loss = watched_loss + favorited_loss + friends_loss
  return loss


def _get_tmpdir(name, subdir=''):
  segments = [FLAGS.model_dir, name] + ([subdir] if subdir else [])
  return os.path.join(*segments)


def _get_variable(variable):
  if isinstance(variable, tpu_embedding_v2.TPUShardedVariable):
    return variable.variables[0]
  return variable


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
