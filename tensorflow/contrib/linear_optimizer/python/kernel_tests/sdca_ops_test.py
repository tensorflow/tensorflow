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
"""Tests for SdcaModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading

from tensorflow.contrib.linear_optimizer.python.ops.sdca_ops import SdcaModel
from tensorflow.contrib.linear_optimizer.python.ops.sparse_feature_column import SparseFeatureColumn
from tensorflow.core.example import example_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import googletest

_MAX_ITERATIONS = 100
_SHARD_NUMBERS = [None, 1, 3]
_NUM_LOSS_PARTITIONS = [4]


def make_example_proto(feature_dict, target, value=1.0):
  e = example_pb2.Example()
  features = e.features

  features.feature['target'].float_list.value.append(target)

  for key, values in feature_dict.items():
    features.feature[key + '_indices'].int64_list.value.extend(values)
    features.feature[key + '_values'].float_list.value.extend([value] *
                                                              len(values))

  return e


def make_example_dict(example_protos, example_weights):

  def parse_examples(example_protos):
    features = {
        'target':
            parsing_ops.FixedLenFeature(
                shape=[1], dtype=dtypes.float32, default_value=0),
        'age_indices':
            parsing_ops.VarLenFeature(dtype=dtypes.int64),
        'age_values':
            parsing_ops.VarLenFeature(dtype=dtypes.float32),
        'gender_indices':
            parsing_ops.VarLenFeature(dtype=dtypes.int64),
        'gender_values':
            parsing_ops.VarLenFeature(dtype=dtypes.float32)
    }
    return parsing_ops.parse_example(
        [e.SerializeToString() for e in example_protos], features)

  parsed = parse_examples(example_protos)
  sparse_features = [
      SparseFeatureColumn(
          array_ops.reshape(
              array_ops.split(
                  value=parsed['age_indices'].indices,
                  num_or_size_splits=2,
                  axis=1)[0], [-1]),
          array_ops.reshape(parsed['age_indices'].values, [-1]),
          array_ops.reshape(parsed['age_values'].values, [-1])),
      SparseFeatureColumn(
          array_ops.reshape(
              array_ops.split(
                  value=parsed['gender_indices'].indices,
                  num_or_size_splits=2,
                  axis=1)[0], [-1]),
          array_ops.reshape(parsed['gender_indices'].values, [-1]),
          array_ops.reshape(parsed['gender_values'].values, [-1]))
  ]
  return dict(
      sparse_features=sparse_features,
      dense_features=[],
      example_weights=example_weights,
      example_labels=array_ops.reshape(parsed['target'], [-1]),
      example_ids=['%d' % i for i in range(0, len(example_protos))])


def make_random_examples_and_variables_dicts(num_examples, dim, num_non_zero):
  random.seed(1)

  sparse_features = [
      SparseFeatureColumn(
          [i for i in range(num_examples) for _ in range(num_non_zero)], [
              i for _ in range(num_examples)
              for i in random.sample(range(dim), num_non_zero)
          ],
          [num_non_zero**(-0.5) for _ in range(num_examples * num_non_zero)])
  ]
  examples_dict = dict(
      sparse_features=sparse_features,
      dense_features=[],
      example_weights=[random.random() for _ in range(num_examples)],
      example_labels=[
          1. if random.random() > 0.5 else 0. for _ in range(num_examples)
      ],
      example_ids=[str(i) for i in range(num_examples)])

  weights = variables_lib.Variable(
      array_ops.zeros([dim], dtype=dtypes.float32))
  variables_dict = dict(
      sparse_features_weights=[weights],
      dense_features_weights=[])

  return examples_dict, variables_dict


def make_variable_dict(max_age, max_gender, partitioned=False):
  # TODO(sibyl-toe9oF2e):  Figure out how to derive max_age & max_gender from
  # examples_dict.
  partitioner = None
  if partitioned:
    partitioner = partitioned_variables.fixed_size_partitioner(num_shards=2,
                                                               axis=0)
  with variable_scope.variable_scope(
      name_or_scope='variables',
      partitioner=partitioner):
    age_weights = variables_lib.Variable(
        array_ops.zeros(
            [max_age + 1], dtype=dtypes.float32))
    gender_weights = variables_lib.Variable(
        array_ops.zeros(
            [max_gender + 1], dtype=dtypes.float32))
  return dict(
      sparse_features_weights=[age_weights, gender_weights],
      dense_features_weights=[])


def make_dense_examples_and_variables_dicts(dense_features_values, weights,
                                            labels):
  """Creates examples and variables dictionaries for dense features.

  Variables shapes are inferred from the list of dense feature values passed as
  argument.

  Args:
    dense_features_values: The values of the dense features
    weights: The example weights.
    labels: The example labels.
  Returns:
    One dictionary for the examples and one for the variables.
  """
  dense_tensors = []
  dense_weights = []
  for dense_feature in dense_features_values:
    dense_tensor = ops.convert_to_tensor(dense_feature, dtype=dtypes.float32)
    check_shape_op = control_flow_ops.Assert(
        math_ops.less_equal(array_ops.rank(dense_tensor), 2),
        ['dense_tensor shape must be [batch_size, dimension] or [batch_size]'])
    # Reshape to [batch_size, dense_column_dimension].
    with ops.control_dependencies([check_shape_op]):
      dense_tensor = array_ops.reshape(
          dense_tensor, [dense_tensor.get_shape().as_list()[0], -1])
    dense_tensors.append(dense_tensor)
    # Add variables of shape [feature_column_dimension].
    dense_weights.append(
        variables_lib.Variable(
            array_ops.zeros(
                [dense_tensor.get_shape().as_list()[1]], dtype=dtypes.float32)))

  examples_dict = dict(
      sparse_features=[],
      dense_features=dense_tensors,
      example_weights=weights,
      example_labels=labels,
      example_ids=['%d' % i for i in range(0, len(labels))])
  variables_dict = dict(
      sparse_features_weights=[], dense_features_weights=dense_weights)

  return examples_dict, variables_dict


def get_binary_predictions_for_logistic(predictions, cutoff=0.5):
  return math_ops.cast(
      math_ops.greater_equal(predictions,
                             array_ops.ones_like(predictions) * cutoff),
      dtype=dtypes.int32)


def get_binary_predictions_for_hinge(predictions):
  return math_ops.cast(
      math_ops.greater_equal(predictions, array_ops.zeros_like(predictions)),
      dtype=dtypes.int32)


# TODO(sibyl-Mooth6ku): Add tests that exercise L1 and Shrinking.
# TODO(sibyl-vie3Poto): Refactor tests to avoid repetition of boilerplate code.
class SdcaModelTest(TensorFlowTestCase):
  """Base SDCA optimizer test class for any loss type."""

  def _single_threaded_test_session(self):
    config = config_pb2.ConfigProto(
        inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    return self.test_session(use_gpu=False, config=config)


class SdcaWithLogisticLossTest(SdcaModelTest):
  """SDCA optimizer test class for logistic loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()
        # The high tolerance in unregularized_loss comparisons is due to the
        # fact that it's possible to trade off unregularized_loss vs.
        # regularization and still have a sum that is quite close to the
        # optimal regularized_loss value.  SDCA's duality gap only ensures that
        # the regularized_loss is within 0.01 of optimal.
        # 0.525457 is the optimal regularized_loss.
        # 0.411608 is the unregularized_loss at that optimum.
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(
            0.01, lr.approximate_duality_gap().eval(), rtol=1e-2, atol=1e-2)

  def testPartitionedPrimals(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1, partitioned=True)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()
        # The high tolerance in unregularized_loss comparisons is due to the
        # fact that it's possible to trade off unregularized_loss vs.
        # regularization and still have a sum that is quite close to the
        # optimal regularized_loss value.  SDCA's duality gap only ensures that
        # the regularized_loss is within 0.01 of optimal.
        # 0.525457 is the optimal regularized_loss.
        # 0.411608 is the unregularized_loss at that optimum.
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(
            0.01, lr.approximate_duality_gap().eval(), rtol=1e-2, atol=1e-2)

  def testSparseRandom(self):
    dim = 20
    num_examples = 1000
    # Number of non-zero features per example.
    non_zeros = 10
    # Setup test data.
    with self._single_threaded_test_session():
      examples, variables = make_random_examples_and_variables_dicts(
          num_examples, dim, non_zeros)
      options = dict(
          symmetric_l2_regularization=.1,
          symmetric_l1_regularization=0,
          num_table_shards=1,
          adaptive=False,
          loss_type='logistic_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      train_op = lr.minimize()
      for _ in range(4):
        train_op.run()
      lr.update_weights(train_op).run()
      # Duality gap is 1.4e-5.
      # It would be 0.01 without shuffling and 0.02 with adaptive sampling.
      self.assertNear(0.0, lr.approximate_duality_gap().eval(), err=1e-3)

  def testSparseDuplicate(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0] * 5,
            'gender': [0] * 5
        }, 0),
        make_example_proto({
            'age': [1] * 5,
            'gender': [1] * 5
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1,
          symmetric_l1_regularization=0,
          loss_type='logistic_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      train_op = lr.minimize()
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   'Duplicate'):
        train_op.run()

  def testDistributedSimple(self):
    # Distributed SDCA may not converge if the workers update concurrently the
    # same example. In this test the examples are partitioned across workers.
    # The examples are the same for all workers, just the example_ids are
    # different.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    examples = make_example_dict(example_protos, example_weights)
    example_ids = array_ops.placeholder(
        dtypes.string, shape=(len(example_weights),))
    examples['example_ids'] = example_ids
    variables = make_variable_dict(1, 1)
    for num_shards in _SHARD_NUMBERS:
      for num_loss_partitions in _NUM_LOSS_PARTITIONS:
        with self._single_threaded_test_session():
          options = dict(
              # Keep the same solution as for TestSimple: since the number of
              # examples is multplied by num_loss_partitions, multiply also
              # L2 by the same value.
              symmetric_l2_regularization=num_loss_partitions,
              symmetric_l1_regularization=0,
              loss_type='logistic_loss',
              num_table_shards=num_shards,
              num_loss_partitions=num_loss_partitions)

          lr = SdcaModel(examples, variables, options)
          variables_lib.global_variables_initializer().run()
          unregularized_loss = lr.unregularized_loss(examples)
          loss = lr.regularized_loss(examples)
          predictions = lr.predictions(examples)
          self.assertAllClose(0.693147, unregularized_loss.eval())
          self.assertAllClose(0.693147, loss.eval())

          train_op = lr.minimize()

          def minimize(worker_id):
            with self._single_threaded_test_session():
              feed_dict = {example_ids: [
                  str(i + worker_id*len(example_weights)) for i in range(
                      len(example_weights))]}
              for _ in range(_MAX_ITERATIONS):
                train_op.run(feed_dict=feed_dict)  # pylint: disable=cell-var-from-loop

          threads = []
          for worker_id in range(num_loss_partitions):
            threads.append(threading.Thread(target=minimize, args=(worker_id,)))
            threads[-1].start()

          for t in threads:
            t.join()
          lr.update_weights(train_op).run(feed_dict={
              example_ids: [str(i) for i in range(len(example_weights))]})

          # Test only the unregularized loss because the optimal value of the
          # regularized loss depends on num_loss_partitions.
          self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.02)
          predicted_labels = get_binary_predictions_for_logistic(predictions)
          self.assertAllEqual([0, 1], predicted_labels.eval())
          self.assertNear(0.0, lr.approximate_duality_gap().eval(), 0.02)

  def testSimpleNoL2(self):
    # Same as test above (so comments from above apply) but without an L2.
    # The algorithm should behave as if we have an L2 of 1 in optimization but
    # 0 in regularized_loss.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=0,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        self.assertAllClose(0.693147, unregularized_loss.eval())
        self.assertAllClose(0.693147, loss.eval())
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()

        # There is neither L1 nor L2 loss, so regularized and unregularized
        # losses should be exactly the same.
        self.assertAllClose(0.40244, unregularized_loss.eval(), atol=0.01)
        self.assertAllClose(0.40244, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(
            0.01, lr.approximate_duality_gap().eval(), rtol=1e-2, atol=1e-2)

  def testSomeUnweightedExamples(self):
    # Setup test data with 4 examples, but should produce the same
    # results as testSimple.
    example_protos = [
        # Will be used.
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        # Will be ignored.
        make_example_proto({
            'age': [1],
            'gender': [0]
        }, 0),
        # Will be used.
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
        # Will be ignored.
        make_example_proto({
            'age': [1],
            'gender': [0]
        }, 1),
    ]
    example_weights = [1.0, 0.0, 1.0, 0.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        # Only use examples 0 and 2
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()

        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllClose([0, 1, 1, 1], predicted_labels.eval())
        self.assertAllClose(
            0.0, lr.approximate_duality_gap().eval(), rtol=1e-2, atol=1e-2)

  def testFractionalExampleLabel(self):
    # Setup test data with 1 positive, and 1 mostly-negative example.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0.1),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 0.9),
    ]
    example_weights = [1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        with self.assertRaisesOpError(
            'Only labels of 0.0 or 1.0 are supported right now.'):
          lr.minimize().run()

  def testImbalanced(self):
    # Setup test data with 1 positive, and 3 negative examples.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [2],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [3],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(3, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()

        self.assertAllClose(
            0.226487 + 0.102902, unregularized_loss.eval(), atol=0.08)
        self.assertAllClose(0.328394 + 0.131364, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 0, 0, 1], predicted_labels.eval())
        self.assertAllClose(
            0.0, lr.approximate_duality_gap().eval(), rtol=2e-2, atol=1e-2)

  def testImbalancedWithExampleWeights(self):
    # Setup test data with 1 positive, and 1 negative example.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [3.0, 1.0]
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()

        self.assertAllClose(0.284860, unregularized_loss.eval(), atol=0.08)
        self.assertAllClose(0.408044, loss.eval(), atol=0.012)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 1], predicted_labels.eval())
        self.assertAllClose(
            0.0, lr.approximate_duality_gap().eval(), rtol=2e-2, atol=1e-2)

  def testInstancesOfOneClassOnly(self):
    # Setup test data with 1 positive (ignored), and 1 negative example.
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [0]
        }, 1),  # Shares gender with the instance above.
    ]
    example_weights = [1.0, 0.0]  # Second example "omitted" from training.
    for num_shards in _SHARD_NUMBERS:
      with self._single_threaded_test_session():
        examples = make_example_dict(example_protos, example_weights)
        variables = make_variable_dict(1, 1)
        options = dict(
            symmetric_l2_regularization=1,
            symmetric_l1_regularization=0,
            num_table_shards=num_shards,
            loss_type='logistic_loss')

        lr = SdcaModel(examples, variables, options)
        variables_lib.global_variables_initializer().run()
        unregularized_loss = lr.unregularized_loss(examples)
        loss = lr.regularized_loss(examples)
        predictions = lr.predictions(examples)
        train_op = lr.minimize()
        for _ in range(_MAX_ITERATIONS):
          train_op.run()
        lr.update_weights(train_op).run()
        self.assertAllClose(0.411608, unregularized_loss.eval(), atol=0.05)
        self.assertAllClose(0.525457, loss.eval(), atol=0.01)
        predicted_labels = get_binary_predictions_for_logistic(predictions)
        self.assertAllEqual([0, 0], predicted_labels.eval())
        self.assertAllClose(
            0.01, lr.approximate_duality_gap().eval(), rtol=1e-2, atol=1e-2)

  def testOutOfRangeSparseFeatures(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(0, 0)
      options = dict(
          symmetric_l2_regularization=1,
          symmetric_l1_regularization=0,
          loss_type='logistic_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      train_op = lr.minimize()
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   'indices.*'):
        train_op.run()

  def testOutOfRangeDenseFeatures(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.0], [0.0, 1.0]]],
          weights=[20.0, 10.0],
          labels=[1.0, 0.0])
      # Replace with a variable of size 1 instead of 2.
      variables['dense_features_weights'] = [
          variables_lib.Variable(array_ops.zeros(
              [1], dtype=dtypes.float32))
      ]
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='logistic_loss')
      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      train_op = lr.minimize()
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          'More dense features than we have parameters for.*'):
        train_op.run()

  # TODO(katsiaspis): add a test for the case when examples at the end of an
  # epoch are repeated, since example id may be duplicated.


class SdcaWithLinearLossTest(SdcaModelTest):
  """SDCA optimizer test class for linear (squared) loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, -10.0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 14.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1,
          symmetric_l1_regularization=0,
          loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = lr.predictions(examples)
      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # Predictions should be 2/3 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2
      self.assertAllClose(
          [-20.0 / 3.0, 28.0 / 3.0], predictions.eval(), rtol=0.005)
      # Approximate gap should be very close to 0.0. (In fact, because the gap
      # is only approximate, it is likely that upon convergence the duality gap
      # can have a tiny negative value).
      self.assertAllClose(0.0, lr.approximate_duality_gap().eval(), atol=1e-2)

  def testL2Regularization(self):
    # Setup test data
    example_protos = [
        # 2 identical examples
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, -10.0),
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, -10.0),
        # 2 more identical examples
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 14.0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 14.0),
    ]
    example_weights = [1.0, 1.0, 1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=16,
          symmetric_l1_regularization=0,
          loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # Predictions should be 1/5 of label due to minimizing regularized loss:
      #   (label - 2 * weight)^2 + L2 * 16 * weight^2
      optimal1 = -10.0 / 5.0
      optimal2 = 14.0 / 5.0
      self.assertAllClose(
          [optimal1, optimal1, optimal2, optimal2],
          predictions.eval(),
          rtol=0.01)

  def testL1Regularization(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, -10.0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 14.0),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=4.0,
          loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      prediction = lr.predictions(examples)
      loss = lr.regularized_loss(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # Predictions should be -4.0, 48/5 due to minimizing regularized loss:
      #   (label - 2 * weight)^2 / 2 + L2 * 2 * weight^2 + L1 * 4 * weight
      self.assertAllClose([-4.0, 20.0 / 3.0], prediction.eval(), rtol=0.08)

      # Loss should be the sum of the regularized loss value from above per
      # example after plugging in the optimal weights.
      self.assertAllClose(308.0 / 6.0, loss.eval(), atol=0.01)

  def testFeatureValues(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, -10.0, -2.0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 14.0, 2.0),
    ]
    example_weights = [5.0, 3.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)

      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1,
          symmetric_l1_regularization=0,
          loss_type='squared_loss')

      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # There are 4 (sparse) variable weights to be learned. 2 for age and 2 for
      # gender. Let w_1, w_2 be age weights, w_3, w_4 be gender weights, y_1,
      # y_2 be the labels for examples 1 and 2 respectively and s_1, s_2 the
      # corresponding *example* weights. With the given feature values, the loss
      # function is given by:
      # s_1/2(y_1 + 2w_1 + 2w_3)^2 + s_2/2(y_2 - 2w_2 - 2w_4)^2
      # + \lambda/2 (w_1^2 + w_2^2 + w_3^2 + w_4^2). Solving for the optimal, it
      # can be verified that:
      # w_1* = w_3* = -2.0 s_1 y_1/(\lambda + 8 s_1) and
      # w_2* = w_4* = 2 \cdot s_2 y_2/(\lambda + 8 s_2). Equivalently, due to
      # regularization and example weights, the predictions are within:
      # 8 \cdot s_i /(\lambda + 8 \cdot s_i) of the labels.
      self.assertAllClose(
          [-10 * 40.0 / 41.0, 14.0 * 24 / 25.0], predictions.eval(), atol=0.01)

  def testDenseFeaturesWithDefaultWeights(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0], [0.0]], [0.0, 1.0]],
          weights=[1.0, 1.0],
          labels=[10.0, -5.0])
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # The loss function for these particular features is given by:
      # 1/2(label_1-w_1)^2 + 1/2(label_2-w_2)^2 + \lambda/2 (w_1^2 + w_2^2). So,
      # differentiating wrt to w_1, w_2 yields the following optimal values:
      # w_1* = label_1/(\lambda + 1)= 10/2, w_2* =label_2/(\lambda + 1)= -5/2.
      # In this case the (unnormalized regularized) loss will be:
      # 1/2(10-5)^2 + 1/2(5-5/2)^2 + 1/2(5^2 + (5/2)^2) = 125.0/4. The actual
      # loss should be further normalized by the sum of example weights.
      self.assertAllClose([5.0, -2.5], predictions.eval(), rtol=0.01)
      loss = lr.regularized_loss(examples)
      self.assertAllClose(125.0 / 8.0, loss.eval(), atol=0.01)

  def testDenseFeaturesWithArbitraryWeights(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.0], [0.0, 1.0]]],
          weights=[20.0, 10.0],
          labels=[10.0, -5.0])
      options = dict(
          symmetric_l2_regularization=5.0,
          symmetric_l1_regularization=0,
          loss_type='squared_loss')
      lr = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = lr.predictions(examples)

      train_op = lr.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      lr.update_weights(train_op).run()

      # The loss function for these particular features is given by:
      # 1/2 s_1 (label_1-w_1)^2 + 1/2 s_2(label_2-w_2)^2 +
      # \lambda/2 (w_1^2 + w_2^2) where s_1, s_2 are the *example weights. It
      # turns out that the optimal (variable) weights are given by:
      # w_1* = label_1 \cdot s_1/(\lambda + s_1)= 8.0 and
      # w_2* =label_2 \cdot s_2/(\lambda + s_2)= -10/3.
      # In this case the (unnormalized regularized) loss will be:
      # s_1/2(8-10)^2 + s_2/2(5-10/3)^2 + 5.0/2(8^2 + (10/3)^2) = 2175.0/9. The
      # actual loss should be further normalized by the sum of example weights.
      self.assertAllClose([8.0, -10.0 / 3], predictions.eval(), rtol=0.01)
      loss = lr.regularized_loss(examples)
      self.assertAllClose(2175.0 / 270.0, loss.eval(), atol=0.01)


class SdcaWithHingeLossTest(SdcaModelTest):
  """SDCA optimizer test class for hinge loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). Solving the system w1 + w3 = 1.0, w2 + w4 = -1.0 and minimizing
      # wrt to \|\vec{w}\|_2, gives w1=w3=1/2 and w2=w4=-1/2. This gives 0.0
      # unregularized loss and 0.25 L2 loss.
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllEqual([-1.0, 1.0], predictions.eval())
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.0, unregularized_loss.eval())
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.05)

  def testDenseFeaturesPerfectlySeparable(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[1.0, 1.0], [1.0, -1.0]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      self.assertAllClose([1.0, -1.0], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())

      # (1.0, 1.0) and (1.0, -1.0) are perfectly separable by x-axis (that is,
      # the SVM's functional margin >=1), so the unregularized loss is ~0.0.
      # There is only loss due to l2-regularization. For these datapoints, it
      # turns out that w_1~=0.0 and w_2~=1.0 which means that l2 loss is ~0.25.
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.0, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.25, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesSeparableWithinMargins(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0, 0.5], [1.0, -0.5]]],
          weights=[1.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)

      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      # (1.0, 0.5) and (1.0, -0.5) are separable by x-axis but the datapoints
      # are within the margins so there is unregularized loss (1/2 per example).
      # For these datapoints, optimal weights are w_1~=0.0 and w_2~=1.0 which
      # gives an L2 loss of ~0.25.
      self.assertAllClose([0.5, -0.5], predictions.eval(), rtol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.5, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.75, regularized_loss.eval(), atol=0.02)

  def testDenseFeaturesWeightedExamples(self):
    with self._single_threaded_test_session():
      examples, variables = make_dense_examples_and_variables_dicts(
          dense_features_values=[[[1.0], [1.0]], [[0.5], [-0.5]]],
          weights=[3.0, 1.0],
          labels=[1.0, 0.0])
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='hinge_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()
      predictions = model.predictions(examples)
      binary_predictions = get_binary_predictions_for_hinge(predictions)
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      # Point (1.0, 0.5) has higher weight than (1.0, -0.5) so the model will
      # try to increase the margin from (1.0, 0.5). Due to regularization,
      # (1.0, -0.5) will be within the margin. For these points and example
      # weights, the optimal weights are w_1~=0.4 and w_2~=1.2 which give an L2
      # loss of 0.5 * 0.25 * 0.25 * 1.6 = 0.2. The binary predictions will be
      # correct, but the boundary will be much closer to the 2nd point than the
      # first one.
      self.assertAllClose([1.0, -0.2], predictions.eval(), atol=0.05)
      self.assertAllEqual([1, 0], binary_predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(0.2, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.4, regularized_loss.eval(), atol=0.02)


class SdcaWithSmoothHingeLossTest(SdcaModelTest):
  """SDCA optimizer test class for smooth hinge loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 1),
    ]
    example_weights = [1.0, 1.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='smooth_hinge_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 0.5 * (1+1) = 1.0.
      predictions = model.predictions(examples)
      self.assertAllClose([0.0, 0.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # After minimization, the model separates perfectly the data points. There
      # are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender (say w3
      # and w4). The minimization leads to w1=w3=1/3 and w2=w4=-1/3. This gives
      # an unregularized hinge loss of 0.33 and a 0.11 L2 loss
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      binary_predictions = get_binary_predictions_for_hinge(predictions)
      self.assertAllClose([-0.67, 0.67], predictions.eval(), atol=0.05)
      self.assertAllEqual([0, 1], binary_predictions.eval())
      self.assertAllClose(0.33, unregularized_loss.eval(), atol=0.02)
      self.assertAllClose(0.44, regularized_loss.eval(), atol=0.02)

class SdcaWithPoissonLossTest(SdcaModelTest):
  """SDCA optimizer test class for poisson loss."""

  def testSimple(self):
    # Setup test data
    example_protos = [
        make_example_proto({
            'age': [0],
            'gender': [0]
        }, 0),
        make_example_proto({
            'age': [1],
            'gender': [1]
        }, 2),
    ]
    example_weights = [100.0, 100.0]
    with self._single_threaded_test_session():
      examples = make_example_dict(example_protos, example_weights)
      variables = make_variable_dict(1, 1)
      options = dict(
          symmetric_l2_regularization=1.0,
          symmetric_l1_regularization=0,
          loss_type='poisson_loss')
      model = SdcaModel(examples, variables, options)
      variables_lib.global_variables_initializer().run()

      # Before minimization, the weights default to zero. There is no loss due
      # to regularization, only unregularized loss which is 1 for each example.
      predictions = model.predictions(examples)
      self.assertAllClose([1.0, 1.0], predictions.eval())
      unregularized_loss = model.unregularized_loss(examples)
      regularized_loss = model.regularized_loss(examples)
      approximate_duality_gap = model.approximate_duality_gap()
      self.assertAllClose(1.0, unregularized_loss.eval())
      self.assertAllClose(1.0, regularized_loss.eval())

      # There are 4 sparse weights: 2 for age (say w1, w2) and 2 for gender
      # (say w3 and w4). The minimization leads to:
      # w1=w3=-1.96487, argmin of 100*(exp(2*w)-2*w*0)+w**2.
      # w2=w4=0.345708, argmin of 100*(exp(2*w)-2*w*2)+w**2.
      # This gives an unregularized loss of .3167 and .3366 with regularization.
      train_op = model.minimize()
      for _ in range(_MAX_ITERATIONS):
        train_op.run()
      model.update_weights(train_op).run()

      self.assertAllClose([0.0196, 1.9965], predictions.eval(), atol=1e-4)
      self.assertAllClose(0.3167, unregularized_loss.eval(), atol=1e-4)
      self.assertAllClose(0.3366, regularized_loss.eval(), atol=1e-4)
      self.assertAllClose(0., approximate_duality_gap.eval(), atol=1e-6)


class SdcaFprintTest(SdcaModelTest):
  """Tests for the SdcaFprint op.

  This is one way of enforcing the platform-agnostic nature of SdcaFprint.
  Basically we are checking against exact values and this test could be running
  across different platforms. Note that it is fine for expected values to change
  in the future, if the implementation of SdcaFprint changes (ie this is *not* a
  frozen test).
  """

  def testFprint(self):
    with self._single_threaded_test_session():
      in_data = constant_op.constant(['abc', 'very looooooong string', 'def'])
      out_data = gen_sdca_ops.sdca_fprint(in_data)
      self.assertAllEqual([[4143508125394299908, -6879828354153669051],
                           [5849691694103072671, -4874542629849009556],
                           [603227410218889250, 8762207001949257490]],
                          out_data.eval())


if __name__ == '__main__':
  googletest.main()
