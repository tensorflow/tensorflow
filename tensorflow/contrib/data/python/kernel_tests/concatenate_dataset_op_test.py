# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.contrib.data.python.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class ConcatenateDatasetTest(test.TestCase):

  def testConcatenateDataset(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0]))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(concatenated.output_shapes, (tensor_shape.TensorShape(
        [20]), tensor_shape.TensorShape([15]), tensor_shape.TensorShape([])))

    iterator = concatenated.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(9):
        result = sess.run(get_next)
        if i < 4:
          for component, result_component in zip(input_components, result):
            self.assertAllEqual(component[i], result_component)
        else:
          for component, result_component in zip(to_concatenate_components,
                                                 result):
            self.assertAllEqual(component[i - 4], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testConcatenateDatasetDifferentShape(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    concatenated = input_dataset.concatenate(dataset_to_concatenate)
    self.assertEqual(
        [ts.as_list()
         for ts in nest.flatten(concatenated.output_shapes)], [[20], [None]])

    iterator = concatenated.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(9):
        result = sess.run(get_next)
        if i < 4:
          for component, result_component in zip(input_components, result):
            self.assertAllEqual(component[i], result_component)
        else:
          for component, result_component in zip(to_concatenate_components,
                                                 result):
            self.assertAllEqual(component[i - 4], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testConcatenateDatasetDifferentStructure(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1], [2], [3], [4], [5]]), 20),
        np.tile(np.array([[12], [13], [14], [15], [16]]), 15),
        np.array([37.0, 38.0, 39.0, 40.0, 41.0]))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegexp(ValueError,
                                 "don't have the same number of elements"):
      input_dataset.concatenate(dataset_to_concatenate)

  def testConcatenateDatasetDifferentType(self):
    input_components = (
        np.tile(np.array([[1], [2], [3], [4]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (
        np.tile(np.array([[1.0], [2.0], [3.0], [4.0]]), 5),
        np.tile(np.array([[12], [13], [14], [15]]), 15))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)

    with self.assertRaisesRegexp(TypeError, "have different types"):
      input_dataset.concatenate(dataset_to_concatenate)

  def _iterator_checkpoint_prefix(self):
    return os.path.join(self.get_temp_dir(), "iterator")

  def _build_graph(self, input_components, to_concatenate_components):
    input_dataset = dataset_ops.Dataset.from_tensor_slices(input_components)
    dataset_to_concatenate = dataset_ops.Dataset.from_tensor_slices(
        to_concatenate_components)
    iterator = input_dataset.concatenate(
        dataset_to_concatenate).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()
    saveable = iterator_ops.make_saveable_from_iterator(iterator)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    # TODO(shivaniagrawal) : non-intuitive way, add support in mata_graph
    for t in nest.flatten(get_next):
      ops.add_to_collection("get_next", t)
    return init_op, get_next

  def _testSaveRestoreUtility(self, start, break_range, stop):
    path = self._iterator_checkpoint_prefix()
    step = 0
    meta_filename = path + "-%d.meta" % step

    input_components = (np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
        np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (np.tile(
        np.array([[5], [6], [7], [8], [9]]), 20), np.tile(
            np.array([[16], [17], [18], [19], [20]]), 15))

    with ops.Graph().as_default() as g:
      init_op, get_next = self._build_graph(input_components,
                                            to_concatenate_components)
      saver = saver_lib.Saver()
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        for i in range(start, break_range):
          result = sess.run(get_next)
          if i < 4:
            for component, result_component in zip(input_components, result):
              self.assertAllEqual(component[i], result_component)
          else:
            for component, result_component in zip(to_concatenate_components,
                                                   result):
              self.assertAllEqual(component[i - 4], result_component)
        saver.save(sess, path, step)

    with ops.Graph().as_default() as g:
      saver = saver_lib.import_meta_graph(meta_filename)
      with self.test_session(graph=g) as sess:
        get_next = nest.pack_sequence_as(("a", "b"),
                                         ops.get_collection("get_next"))
        saver.restore(sess, saver_lib.latest_checkpoint(self.get_temp_dir()))
        for i in range(break_range, stop):
          result = sess.run(get_next)
          if i < 4:
            for component, result_component in zip(input_components, result):
              self.assertAllEqual(component[i], result_component)
          else:
            for component, result_component in zip(to_concatenate_components,
                                                   result):
              self.assertAllEqual(component[i - 4], result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)

  def testRestoreAtFirstDataset(self):
    start = 0
    stop = 9
    break_range = 3
    self._testSaveRestoreUtility(start, break_range, stop)

  def testRestoreAtSecondDataset(self):
    start = 0
    stop = 9
    break_range = 6
    self._testSaveRestoreUtility(start, break_range, stop)

  def testRestoreAtBetweenDatasets(self):
    start = 0
    stop = 9
    break_range = 4
    self._testSaveRestoreUtility(start, break_range, stop)

  def testRestoreExhaustedIterator(self):
    start = 0
    stop = 9
    break_range = 9
    self._testSaveRestoreUtility(start, break_range, stop)

  def testRestoreInModifiedGraph(self):
    start = 0
    stop = 9
    break_range = 6
    path = self._iterator_checkpoint_prefix()
    step = 0

    input_components = (np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
        np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (np.tile(
        np.array([[5], [6], [7], [8], [9]]), 20), np.tile(
            np.array([[16], [17], [18], [19], [20]]), 15))

    with ops.Graph().as_default() as g:
      init_op, get_next = self._build_graph(input_components,
                                            to_concatenate_components)
      saver = saver_lib.Saver(allow_empty=True)
      with self.test_session(graph=g) as sess:
        sess.run(init_op)
        for i in range(start, break_range):
          result = sess.run(get_next)
          if i < 4:
            for component, result_component in zip(input_components, result):
              self.assertAllEqual(component[i], result_component)
          else:
            for component, result_component in zip(to_concatenate_components,
                                                   result):
              self.assertAllEqual(component[i - 4], result_component)
        saver.save(sess, path, step)

    new_to_concatenate_components = (np.array([[5], [6], [7], [8], [9]]),
                                     np.array([[16], [17], [18], [19], [20]]))
    with ops.Graph().as_default() as g:
      init_op, get_next = self._build_graph(input_components,
                                            new_to_concatenate_components)
      saver = saver_lib.Saver()
      with self.test_session(graph=g) as sess:
        saver.restore(sess, saver_lib.latest_checkpoint(self.get_temp_dir()))
        for i in range(break_range, stop):
          result = sess.run(get_next)
          for component, result_component in zip(to_concatenate_components,
                                                 result):
            self.assertAllEqual(component[i - 4], result_component)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(get_next)


if __name__ == "__main__":
  test.main()
