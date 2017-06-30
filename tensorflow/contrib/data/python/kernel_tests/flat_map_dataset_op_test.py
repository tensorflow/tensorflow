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

import random

import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class FlatMapDatasetTest(test.TestCase):

  # pylint: disable=g-long-lambda
  def testFlatMapDataset(self):
    repeats = [1, 2, 3, 4, 5, 0, 1]
    components = np.array(repeats, dtype=np.int64)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .flat_map(lambda x: dataset_ops.Dataset.from_tensors([x]).repeat(x))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in repeats:
        for _ in range(i):
          self.assertEqual(i, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .flat_map(lambda x: dataset_ops.Dataset.from_tensor_slices(x)
                  .flat_map(lambda y: dataset_ops.Dataset.from_tensors(y)
                            .repeat(y))).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for row in repeats:
        for i in row:
          for _ in range(i):
            self.assertEqual(i, sess.run(get_next))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSharedResourceNestedFlatMapDataset(self):
    repeats = [[1, 2], [3, 4], [5, 0], [1, 7]]
    components = np.array(repeats, dtype=np.int64)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .flat_map(lambda x: dataset_ops.Dataset.from_tensor_slices(x)
                  .flat_map(lambda y: dataset_ops.Dataset.from_tensors(y)
                            .repeat(y))).make_initializable_iterator(
                                shared_name="shared_flat_map_iterator"))
    init_op = iterator.initializer
    get_next = iterator.get_next()

    # Create two concurrent sessions that share the same iterator
    # resource on the same server, and verify that a random
    # interleaving of `Session.run(get_next)` calls on the two
    # sessions yields the expected result.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess1:
      with session.Session(server.target) as sess2:
        for _ in range(3):
          sess = random.choice([sess1, sess2])
          sess.run(init_op)
          for row in repeats:
            for i in row:
              for _ in range(i):
                sess = random.choice([sess1, sess2])
                self.assertEqual(i, sess.run(get_next))

        with self.assertRaises(errors.OutOfRangeError):
          sess = random.choice([sess1, sess2])
          sess.run(get_next)

  def testMapDict(self):
    iterator = (dataset_ops.Dataset.range(10)
                .map(lambda x: {"foo": x * 2, "bar": x ** 2})
                .flat_map(lambda d: dataset_ops.Dataset.from_tensors(d["foo"])
                          .repeat(d["bar"]))
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(10):
        for _ in range(i ** 2):
          self.assertEqual(i * 2, sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)
  # pylint: enable=g-long-lambda


if __name__ == "__main__":
  test.main()
