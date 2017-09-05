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

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes

  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np
from tensorflow.contrib.learn.python.learn.learn_io import generator_io
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class GeneratorIoTest(test.TestCase):

  def testGeneratorInputFn(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones(1) * index,
            'b': np.ones(1) * index + 32,
            'label': np.ones(1) * index - 32
        }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator,
          target_key='label',
          batch_size=2,
          shuffle=False,
          num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], np.asarray([0, 1]).reshape(-1, 1))
      self.assertAllEqual(res[0]['b'], np.asarray([32, 33]).reshape(-1, 1))
      self.assertAllEqual(res[1], np.asarray([-32, -31]).reshape(-1, 1))

      session.run([features])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testGeneratorSingleInputFn(self):

    def generator():
      for index in range(2):
        yield {'a': np.ones(1) * index}

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator, target_key=None, batch_size=2, shuffle=False, num_epochs=1)
      features = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features])
      self.assertAllEqual(res[0]['a'], np.asarray([0, 1]).reshape(-1, 1))

      session.run([features])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features])

      coord.request_stop()
      coord.join(threads)

  def testGeneratorInputFnLabelDict(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones(1) * index,
            'b': np.ones(1) * index + 32,
            'label': np.ones(1) * index - 32,
            'label2': np.ones(1) * index - 64,
        }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator,
          target_key=['label', 'label2'],
          batch_size=2,
          shuffle=False,
          num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'], np.asarray([0, 1]).reshape(-1, 1))
      self.assertAllEqual(res[0]['b'], np.asarray([32, 33]).reshape(-1, 1))
      self.assertAllEqual(res[1]['label'], np.asarray([-32, -31]).reshape(
          -1, 1))
      self.assertAllEqual(res[1]['label2'],
                          np.asarray([-64, -63]).reshape(-1, 1))

      session.run([features])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features, target])

      coord.request_stop()
      coord.join(threads)

  def testGeneratorInputFnWithDifferentDimensionsOfFeatures(self):

    def generator():
      for index in range(100):
        yield {
            'a': np.ones((10, 10)) * index,
            'b': np.ones((5, 5)) * index + 32,
            'label': np.ones((3, 3)) * index - 32
        }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator,
          target_key='label',
          batch_size=2,
          shuffle=False,
          num_epochs=1)
      features, target = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run([features, target])
      self.assertAllEqual(res[0]['a'],
                          np.vstack((np.zeros((10, 10)), np.ones(
                              (10, 10)))).reshape(2, 10, 10))
      self.assertAllEqual(res[0]['b'],
                          np.vstack((np.zeros((5, 5)), np.ones(
                              (5, 5)))).reshape(2, 5, 5) + 32)
      self.assertAllEqual(res[1],
                          np.vstack((np.zeros((3, 3)), np.ones(
                              (3, 3)))).reshape(2, 3, 3) - 32)

      coord.request_stop()
      coord.join(threads)

  def testGeneratorInputFnWithXAsNonGeneratorFunction(self):
    x = np.arange(32, 36)
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x must be generator function'):
        failing_input_fn = generator_io.generator_input_fn(
            x, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFnWithXAsNonGenerator(self):

    def generator():
      return np.arange(32, 36)

    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x\(\) must be generator'):
        failing_input_fn = generator_io.generator_input_fn(
            generator, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFnWithXAsNonGeneratorYieldingDicts(self):

    def generator():
      yield np.arange(32, 36)

    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'x\(\) must yield dict'):
        failing_input_fn = generator_io.generator_input_fn(
            generator, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFNWithTargetLabelNotString(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones((10, 10)) * index,
            'b': np.ones((5, 5)) * index + 32,
            'label': np.ones((3, 3)) * index - 32
        }

    y = np.arange(32, 36)
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'target_key must be str or'
                                   ' Container of str'):
        failing_input_fn = generator_io.generator_input_fn(
            generator, target_key=y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFNWithTargetLabelListNotString(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones((10, 10)) * index,
            'b': np.ones((5, 5)) * index + 32,
            'label': np.ones((3, 3)) * index - 32
        }

    y = ['label', np.arange(10)]
    with self.test_session():
      with self.assertRaisesRegexp(TypeError, 'target_key must be str or'
                                   ' Container of str'):
        failing_input_fn = generator_io.generator_input_fn(
            generator, target_key=y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFNWithTargetLabelNotInDict(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones((10, 10)) * index,
            'b': np.ones((5, 5)) * index + 32,
            'label': np.ones((3, 3)) * index - 32
        }

    y = ['label', 'target']
    with self.test_session():
      with self.assertRaisesRegexp(KeyError, 'target_key not in yielded dict'):
        failing_input_fn = generator_io.generator_input_fn(
            generator, target_key=y, batch_size=2, shuffle=False, num_epochs=1)
        failing_input_fn()

  def testGeneratorInputFnWithNoTargetKey(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones(1) * index,
            'b': np.ones(1) * index + 32,
            'label': np.ones(1) * index - 32
        }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator, target_key=None, batch_size=2, shuffle=False, num_epochs=1)
      features = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run(features)
      self.assertAllEqual(res['a'], np.asarray([0, 1]).reshape(-1, 1))
      self.assertAllEqual(res['b'], np.asarray([32, 33]).reshape(-1, 1))
      self.assertAllEqual(res['label'], np.asarray([-32, -31]).reshape(-1, 1))

      session.run([features])
      with self.assertRaises(errors.OutOfRangeError):
        session.run([features])

      coord.request_stop()
      coord.join(threads)

  def testGeneratorInputFnWithBatchLargerthanData(self):

    def generator():
      for index in range(2):
        yield {
            'a': np.ones(1) * index,
            'b': np.ones(1) * index + 32,
            'label': np.ones(1) * index - 32
        }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator, target_key=None, batch_size=4, shuffle=False, num_epochs=1)
      features = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      res = session.run(features)
      self.assertAllEqual(res['a'], np.asarray([0, 1, 0, 1]).reshape(-1, 1))
      self.assertAllEqual(res['b'], np.asarray([32, 33, 32, 33]).reshape(-1, 1))
      self.assertAllEqual(res['label'],
                          np.asarray([-32, -31, -32, -31]).reshape(-1, 1))

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features])

      coord.request_stop()
      coord.join(threads)

  def testGeneratorInputFnWithMismatchinGeneratorKeys(self):

    def generator():
      index = 0
      yield {
          'a': np.ones(1) * index,
          'b': np.ones(1) * index + 32,
          'label': np.ones(1) * index - 32
      }
      index = 1
      yield {
          'a': np.ones(1) * index,
          'c': np.ones(1) * index + 32,
          'label': np.ones(1) * index - 32
      }

    with self.test_session() as session:
      input_fn = generator_io.generator_input_fn(
          generator, target_key=None, batch_size=2, shuffle=False, num_epochs=1)
      features = input_fn()

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      with self.assertRaises(errors.OutOfRangeError):
        session.run([features])

      with self.assertRaisesRegex(KeyError, 'key mismatch between dicts emitted'
                                  ' by GenFunExpected'):
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
  test.main()
