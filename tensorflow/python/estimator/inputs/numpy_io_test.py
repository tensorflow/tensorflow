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
"""Tests for numpy_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.client import session as session_lib
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.feature_column.feature_column import _LinearModel
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl


class NumpyIoTest(test.TestCase):

  def testNumpyInputFn(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -28)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      session.run([features, target])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithVeryLargeBatchSizeAndMultipleEpochs(self):
    a = np.arange(2) * 1.0
    b = np.arange(32, 34)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -30)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=128, shuffle=False, num_epochs=2)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1, 0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33, 32, 33])
      self.assertAllEqual(res[1], [-32, -31, -32, -31])

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithZeroEpochs(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -28)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=0)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithBatchSizeNotDividedByDataSize(self):
    batch_size = 2
    a = np.arange(5) * 1.0
    b = np.arange(32, 37)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -27)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=batch_size, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [2, 3])
      self.assertAllEqual(res[0]['b'], [34, 35])
      self.assertAllEqual(res[1], [-30, -29])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [4])
      self.assertAllEqual(res[0]['b'], [36])
      self.assertAllEqual(res[1], [-28])

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithBatchSizeNotDividedByDataSizeAndMultipleEpochs(self):
    batch_size = 2
    a = np.arange(3) * 1.0
    b = np.arange(32, 35)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -29)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=batch_size, shuffle=False, num_epochs=3)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [2, 0])
      self.assertAllEqual(res[0]['b'], [34, 32])
      self.assertAllEqual(res[1], [-30, -32])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [1, 2])
      self.assertAllEqual(res[0]['b'], [33, 34])
      self.assertAllEqual(res[1], [-31, -30])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1])
      self.assertAllEqual(res[0]['b'], [32, 33])
      self.assertAllEqual(res[1], [-32, -31])

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [2])
      self.assertAllEqual(res[0]['b'], [34])
      self.assertAllEqual(res[1], [-30])

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithBatchSizeLargerThanDataSize(self):
    batch_size = 10
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = np.arange(-32, -28)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=batch_size, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [0, 1, 2, 3])
      self.assertAllEqual(res[0]['b'], [32, 33, 34, 35])
      self.assertAllEqual(res[1], [-32, -31, -30, -29])

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithDifferentDimensionsOfFeatures(self):
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    x = {'a': a, 'b': b}
    y = np.arange(-32, -30)

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], [[1, 2], [3, 4]])
      self.assertAllEqual(res[0]['b'], [5, 6])
      self.assertAllEqual(res[1], [-32, -31])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithXAsNonDict(self):
    x = list(range(32, 36))
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x must be a dict or array'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x, y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testNumpyInputFnWithXIsEmptyDict(self):
    x = {}
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, 'x cannot be an empty'):
        failing_input_fn = numpy_io.numpy_input_fn(x, y, shuffle=False)
        failing_input_fn()

  def testNumpyInputFnWithXIsEmptyArray(self):
    x = np.array([[], []])
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, 'x cannot be an empty'):
        failing_input_fn = numpy_io.numpy_input_fn(x, y, shuffle=False)
        failing_input_fn()

  def testNumpyInputFnWithYIsNone(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = None

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features_tensor = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      feature = session.run(features_tensor)
      self.assertEqual(len(feature), 2)
      self.assertAllEqual(feature['a'], [0, 1])
      self.assertAllEqual(feature['b'], [32, 33])

      session.run([features_tensor])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features_tensor])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithNonBoolShuffle(self):
    x = np.arange(32, 36)
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(ValueError,
                                   'shuffle must be provided and explicitly '
                                   'set as boolean'):
        # Default shuffle is None.
        numpy_io.numpy_input_fn(x, y)

  def testNumpyInputFnWithTargetKeyAlreadyInX(self):
    array = np.arange(32, 36)
    x = {'__target_key__': array}
    y = np.arange(4)

    with self.test_session():
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      input_fn()
      self.assertAllEqual(x['__target_key__'], array)
      # The input x should not be mutated.
      self.assertItemsEqual(x.keys(), ['__target_key__'])

  def testNumpyInputFnWithMismatchLengthOfInputs(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    x_mismatch_length = {'a': np.arange(1), 'b': b}
    y_longer_length = np.arange(10)

    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, 'Length of tensors in x and y is mismatched.'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x, y_longer_length, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

      with self.assertRaisesRegexp(
          ValueError, 'Length of tensors in x and y is mismatched.'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x=x_mismatch_length,
            y=None,
            batch_size=2,
            shuffle=False,
            num_epochs=1)
        failing_input_fn()

  def testNumpyInputFnWithYAsDict(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = {'y1': np.arange(-32, -28), 'y2': np.arange(32, 28, -1)}

    with self.test_session() as session:
      input_fn = numpy_io.numpy_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)
      features_tensor, targets_tensor = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      features, targets = session.run([features_tensor, targets_tensor])
      self.assertEqual(len(features), 2)
      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertEqual(len(targets), 2)
      self.assertAllEqual(targets['y1'], [-32, -31])
      self.assertAllEqual(targets['y2'], [32, 31])

      session.run([features_tensor, targets_tensor])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features_tensor, targets_tensor])

      coord.request_stop()
      coord.join(threads)

  def testNumpyInputFnWithYIsEmptyDict(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = {}
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, 'y cannot be empty'):
        failing_input_fn = numpy_io.numpy_input_fn(x, y, shuffle=False)
        failing_input_fn()

  def testNumpyInputFnWithDuplicateKeysInXAndY(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x = {'a': a, 'b': b}
    y = {'y1': np.arange(-32, -28), 'a': a, 'y2': np.arange(32, 28, -1), 'b': b}
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, '2 duplicate keys are found in both x and y'):
        failing_input_fn = numpy_io.numpy_input_fn(x, y, shuffle=False)
        failing_input_fn()

  def testNumpyInputFnWithXIsArray(self):
    x = np.arange(4) * 1.0
    y = np.arange(-32, -28)

    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
    features, target = input_fn()

    with monitored_session.MonitoredSession() as session:
      res = session.run([features, target])
      self.assertAllEqual(res[0], [0, 1])
      self.assertAllEqual(res[1], [-32, -31])

      session.run([features, target])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

  def testNumpyInputFnWithXIsNDArray(self):
    x = np.arange(16).reshape(4, 2, 2) * 1.0
    y = np.arange(-48, -32).reshape(4, 2, 2)

    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
    features, target = input_fn()

    with monitored_session.MonitoredSession() as session:
      res = session.run([features, target])
      self.assertAllEqual(res[0], [[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
      self.assertAllEqual(
          res[1], [[[-48, -47], [-46, -45]], [[-44, -43], [-42, -41]]])

      session.run([features, target])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

  def testNumpyInputFnWithXIsArrayYIsDict(self):
    x = np.arange(4) * 1.0
    y = {'y1': np.arange(-32, -28)}

    input_fn = numpy_io.numpy_input_fn(
        x, y, batch_size=2, shuffle=False, num_epochs=1)
    features_tensor, targets_tensor = input_fn()

    with monitored_session.MonitoredSession() as session:
      features, targets = session.run([features_tensor, targets_tensor])
      self.assertEqual(len(features), 2)
      self.assertAllEqual(features, [0, 1])
      self.assertEqual(len(targets), 1)
      self.assertAllEqual(targets['y1'], [-32, -31])

      session.run([features_tensor, targets_tensor])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features_tensor, targets_tensor])

  def testArrayAndDictGiveSameOutput(self):
    a = np.arange(4) * 1.0
    b = np.arange(32, 36)
    x_arr = np.vstack((a, b))
    x_dict = {'feature1': x_arr}
    y = np.arange(-48, -40).reshape(2, 4)

    input_fn_arr = numpy_io.numpy_input_fn(
        x_arr, y, batch_size=2, shuffle=False, num_epochs=1)
    features_arr, targets_arr = input_fn_arr()

    input_fn_dict = numpy_io.numpy_input_fn(
        x_dict, y, batch_size=2, shuffle=False, num_epochs=1)
    features_dict, targets_dict = input_fn_dict()

    with monitored_session.MonitoredSession() as session:
      res_arr, res_dict = session.run([
          (features_arr, targets_arr), (features_dict, targets_dict)])

      self.assertAllEqual(res_arr[0], res_dict[0]['feature1'])
      self.assertAllEqual(res_arr[1], res_dict[1])


class FeatureColumnIntegrationTest(test.TestCase):

  def _initialized_session(self, config=None):
    sess = session_lib.Session(config=config)
    sess.run(variables_lib.global_variables_initializer())
    sess.run(lookup_ops.tables_initializer())
    return sess

  def _get_linear_model_bias(self, name='linear_model'):
    with variable_scope.variable_scope(name, reuse=True):
      return variable_scope.get_variable('bias_weights')

  def _get_linear_model_column_var(self, column, name='linear_model'):
    return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                              name + '/' + column.name)[0]

  def _get_keras_linear_model_predictions(
      self,
      features,
      feature_columns,
      units=1,
      sparse_combiner='sum',
      weight_collections=None,
      trainable=True,
      cols_to_vars=None):
    keras_linear_model = _LinearModel(
        feature_columns,
        units,
        sparse_combiner,
        weight_collections,
        trainable,
        name='linear_model')
    retval = keras_linear_model(features)  # pylint: disable=not-callable
    if cols_to_vars is not None:
      cols_to_vars.update(keras_linear_model.cols_to_vars())
    return retval

  def test_linear_model_numpy_input_fn(self):
    price = fc.numeric_column('price')
    price_buckets = fc.bucketized_column(price, boundaries=[0., 10., 100.,])
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    input_fn = numpy_io.numpy_input_fn(
        x={
            'price': np.array([-1., 2., 13., 104.]),
            'body-style': np.array(['sedan', 'hardtop', 'wagon', 'sedan']),
        },
        batch_size=2,
        shuffle=False)
    features = input_fn()
    net = fc.linear_model(features, [price_buckets, body_style])
    # self.assertEqual(1 + 3 + 5, net.shape[1])
    with self._initialized_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)

      bias = self._get_linear_model_bias()
      price_buckets_var = self._get_linear_model_column_var(price_buckets)
      body_style_var = self._get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [100 - 10 + 5.]], sess.run(net))

      coord.request_stop()
      coord.join(threads)

  def test_linear_model_impl_numpy_input_fn(self):
    price = fc.numeric_column('price')
    price_buckets = fc.bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    input_fn = numpy_io.numpy_input_fn(
        x={
            'price': np.array([-1., 2., 13., 104.]),
            'body-style': np.array(['sedan', 'hardtop', 'wagon', 'sedan']),
        },
        batch_size=2,
        shuffle=False)
    features = input_fn()
    net = self._get_keras_linear_model_predictions(
        features, [price_buckets, body_style])
    # self.assertEqual(1 + 3 + 5, net.shape[1])
    with self._initialized_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)

      bias = self._get_linear_model_bias()
      price_buckets_var = self._get_linear_model_column_var(price_buckets)
      body_style_var = self._get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [100 - 10 + 5.]], sess.run(net))

      coord.request_stop()
      coord.join(threads)

  def test_functional_input_layer_with_numpy_input_fn(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    # one_hot_body_style has 3 dims in input_layer.
    one_hot_body_style = fc.indicator_column(body_style)
    # embedded_body_style has 5 dims in input_layer.
    embedded_body_style = fc.embedding_column(body_style, dimension=5,
                                              initializer=_initializer)

    input_fn = numpy_io.numpy_input_fn(
        x={
            'price': np.array([11., 12., 13., 14.]),
            'body-style': np.array(['sedan', 'hardtop', 'wagon', 'sedan']),
        },
        batch_size=2,
        shuffle=False)
    features = input_fn()
    net = fc.input_layer(features,
                         [price, one_hot_body_style, embedded_body_style])
    self.assertEqual(1 + 3 + 5, net.shape[1])
    with self._initialized_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess, coord=coord)

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[11., 12., 13., 14., 15., 0., 0., 1., 11.],
           [1., 2., 3., 4., 5., 1., 0., 0., 12]],
          sess.run(net))

      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  test.main()
