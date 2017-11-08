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

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
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
    x = np.arange(32, 36)
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x must be dict'):
        failing_input_fn = numpy_io.numpy_input_fn(
            x, y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testNumpyInputFnWithXIsEmptyDict(self):
    x = {}
    y = np.arange(4)
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, 'x cannot be empty'):
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
      with self.assertRaisesRegexp(TypeError,
                                   'shuffle must be explicitly set as boolean'):
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
    y = {'y1': np.arange(-32, -28),
         'a': a,
         'y2': np.arange(32, 28, -1),
         'b': b}
    with self.test_session():
      with self.assertRaisesRegexp(
              ValueError, '2 duplicate keys are found in both x and y'):
        failing_input_fn = numpy_io.numpy_input_fn(x, y, shuffle=False)
        failing_input_fn()


if __name__ == '__main__':
  test.main()
