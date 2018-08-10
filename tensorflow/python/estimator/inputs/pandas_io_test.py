# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pandas_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.estimator.inputs import pandas_io
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

try:
  # pylint: disable=g-import-not-at-top
  import pandas as pd
  HAS_PANDAS = True
except IOError:
  # Pandas writes a temporary file during import. If it fails, don't use pandas.
  HAS_PANDAS = False
except ImportError:
  HAS_PANDAS = False


class PandasIoTest(test.TestCase):

  def makeTestDataFrame(self):
    index = np.arange(100, 104)
    a = np.arange(4)
    b = np.arange(32, 36)
    x = pd.DataFrame({'a': a, 'b': b}, index=index)
    y = pd.Series(np.arange(-32, -28), index=index)
    return x, y

  def makeTestDataFrameWithYAsDataFrame(self):
    index = np.arange(100, 104)
    a = np.arange(4)
    b = np.arange(32, 36)
    a_label = np.arange(10, 14)
    b_label = np.arange(50, 54)
    x = pd.DataFrame({'a': a, 'b': b}, index=index)
    y = pd.DataFrame({'a_target': a_label, 'b_target': b_label}, index=index)
    return x, y

  def callInputFnOnce(self, input_fn, session):
    results = input_fn()
    coord = coordinator.Coordinator()
    threads = queue_runner_impl.start_queue_runners(session, coord=coord)
    result_values = session.run(results)
    coord.request_stop()
    coord.join(threads)
    return result_values

  def testPandasInputFn_IndexMismatch(self):
    if not HAS_PANDAS:
      return
    x, _ = self.makeTestDataFrame()
    y_noindex = pd.Series(np.arange(-32, -28))
    with self.assertRaises(ValueError):
      pandas_io.pandas_input_fn(
          x, y_noindex, batch_size=2, shuffle=False, num_epochs=1)

  def testPandasInputFn_RaisesWhenTargetColumnIsAList(self):
    if not HAS_PANDAS:
      return

    x, y = self.makeTestDataFrame()

    with self.assertRaisesRegexp(TypeError,
                                 'target_column must be a string type'):
      pandas_io.pandas_input_fn(x, y, batch_size=2,
                                shuffle=False,
                                num_epochs=1,
                                target_column=['one', 'two'])

  def testPandasInputFn_NonBoolShuffle(self):
    if not HAS_PANDAS:
      return
    x, _ = self.makeTestDataFrame()
    y_noindex = pd.Series(np.arange(-32, -28))
    with self.assertRaisesRegexp(ValueError,
                                 'shuffle must be provided and explicitly '
                                 'set as boolean'):
      # Default shuffle is None
      pandas_io.pandas_input_fn(x, y_noindex)

  def testPandasInputFn_ProducesExpectedOutputs(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      features, target = self.callInputFnOnce(input_fn, session)

      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertAllEqual(target, [-32, -31])

  def testPandasInputFnWhenYIsDataFrame_ProducesExpectedOutput(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrameWithYAsDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      features, targets = self.callInputFnOnce(input_fn, session)

      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertAllEqual(targets['a_target'], [10, 11])
      self.assertAllEqual(targets['b_target'], [50, 51])

  def testPandasInputFnYIsDataFrame_HandlesOverlappingColumns(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrameWithYAsDataFrame()
      y = y.rename(columns={'a_target': 'a', 'b_target': 'b'})
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      features, targets = self.callInputFnOnce(input_fn, session)

      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertAllEqual(targets['a'], [10, 11])
      self.assertAllEqual(targets['b'], [50, 51])

  def testPandasInputFnYIsDataFrame_HandlesOverlappingColumnsInTargets(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrameWithYAsDataFrame()
      y = y.rename(columns={'a_target': 'a', 'b_target': 'a_n'})
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      features, targets = self.callInputFnOnce(input_fn, session)

      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertAllEqual(targets['a'], [10, 11])
      self.assertAllEqual(targets['a_n'], [50, 51])

  def testPandasInputFn_ProducesOutputsForLargeBatchAndMultipleEpochs(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      index = np.arange(100, 102)
      a = np.arange(2)
      b = np.arange(32, 34)
      x = pd.DataFrame({'a': a, 'b': b}, index=index)
      y = pd.Series(np.arange(-32, -30), index=index)
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=128, shuffle=False, num_epochs=2)

      results = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      features, target = session.run(results)
      self.assertAllEqual(features['a'], [0, 1, 0, 1])
      self.assertAllEqual(features['b'], [32, 33, 32, 33])
      self.assertAllEqual(target, [-32, -31, -32, -31])

      with self.assertRaises(errors.OutOfRangeError):
        session.run(results)

      coord.request_stop()
      coord.join(threads)

  def testPandasInputFn_ProducesOutputsWhenDataSizeNotDividedByBatchSize(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      index = np.arange(100, 105)
      a = np.arange(5)
      b = np.arange(32, 37)
      x = pd.DataFrame({'a': a, 'b': b}, index=index)
      y = pd.Series(np.arange(-32, -27), index=index)

      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      results = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      features, target = session.run(results)
      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])
      self.assertAllEqual(target, [-32, -31])

      features, target = session.run(results)
      self.assertAllEqual(features['a'], [2, 3])
      self.assertAllEqual(features['b'], [34, 35])
      self.assertAllEqual(target, [-30, -29])

      features, target = session.run(results)
      self.assertAllEqual(features['a'], [4])
      self.assertAllEqual(features['b'], [36])
      self.assertAllEqual(target, [-28])

      with self.assertRaises(errors.OutOfRangeError):
        session.run(results)

      coord.request_stop()
      coord.join(threads)

  def testPandasInputFn_OnlyX(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, _ = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y=None, batch_size=2, shuffle=False, num_epochs=1)

      features = self.callInputFnOnce(input_fn, session)

      self.assertAllEqual(features['a'], [0, 1])
      self.assertAllEqual(features['b'], [32, 33])

  def testPandasInputFn_ExcludesIndex(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)

      features, _ = self.callInputFnOnce(input_fn, session)

      self.assertFalse('index' in features)

  def assertInputsCallableNTimes(self, input_fn, session, n):
    inputs = input_fn()
    coord = coordinator.Coordinator()
    threads = queue_runner_impl.start_queue_runners(session, coord=coord)
    for _ in range(n):
      session.run(inputs)
    with self.assertRaises(errors.OutOfRangeError):
      session.run(inputs)
    coord.request_stop()
    coord.join(threads)

  def testPandasInputFn_RespectsEpoch_NoShuffle(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=4, shuffle=False, num_epochs=1)

      self.assertInputsCallableNTimes(input_fn, session, 1)

  def testPandasInputFn_RespectsEpoch_WithShuffle(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=4, shuffle=True, num_epochs=1)

      self.assertInputsCallableNTimes(input_fn, session, 1)

  def testPandasInputFn_RespectsEpoch_WithShuffleAutosize(self):
    if not HAS_PANDAS:
      return
    with self.test_session() as session:
      x, y = self.makeTestDataFrame()
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=True, queue_capacity=None, num_epochs=2)

      self.assertInputsCallableNTimes(input_fn, session, 4)

  def testPandasInputFn_RespectsEpochUnevenBatches(self):
    if not HAS_PANDAS:
      return
    x, y = self.makeTestDataFrame()
    with self.test_session() as session:
      input_fn = pandas_io.pandas_input_fn(
          x, y, batch_size=3, shuffle=False, num_epochs=1)

      # Before the last batch, only one element of the epoch should remain.
      self.assertInputsCallableNTimes(input_fn, session, 2)

  def testPandasInputFn_Idempotent(self):
    if not HAS_PANDAS:
      return
    x, y = self.makeTestDataFrame()
    for _ in range(2):
      pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=False, num_epochs=1)()
    for _ in range(2):
      pandas_io.pandas_input_fn(
          x, y, batch_size=2, shuffle=True, num_epochs=1)()


if __name__ == '__main__':
  test.main()
