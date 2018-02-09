# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for KafkaDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import kafka_dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class KafkaDatasetTest(test.TestCase):

  def setUp(self):
    # The Kafka server has to be setup before the test
    # and tear down after the test manually.
    # The docker engine has to be installed.
    #
    # To setup the Kafka server:
    # $ bash kafka_test.sh start kafka
    #
    # To team down the Kafka server:
    # $ bash kafka_test.sh stop kafka
    pass

  def testKafkaDataset(self):
    topics = array_ops.placeholder(dtypes.string, shape=[None])
    num_epochs = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kafka_dataset_ops.KafkaDataset(
        topics, group="test", eof=True).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    init_batch_op = iterator.make_initializer(batch_dataset)
    get_next = iterator.get_next()

    with self.test_session() as sess:
      # Basic test: read from topic 0.
      sess.run(init_op, feed_dict={topics: ["test:0:0:4"], num_epochs: 1})
      for i in range(5):
        self.assertEqual("D" + str(i), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from topic 1.
      sess.run(init_op, feed_dict={topics: ["test:0:5:-1"], num_epochs: 1})
      for i in range(5):
        self.assertEqual("D" + str(i + 5), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Basic test: read from both topics.
      sess.run(
          init_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 1
          })
      for j in range(2):
        for i in range(5):
          self.assertEqual("D" + str(i + j * 5), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test repeated iteration through both files.
      sess.run(
          init_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 10
          })
      for _ in range(10):
        for j in range(2):
          for i in range(5):
            self.assertEqual("D" + str(i + j * 5), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test batched and repeated iteration through both files.
      sess.run(
          init_batch_op,
          feed_dict={
              topics: ["test:0:0:4", "test:0:5:-1"],
              num_epochs: 10,
              batch_size: 5
          })
      for _ in range(10):
        self.assertAllEqual(["D" + str(i) for i in range(5)],
                            sess.run(get_next))
        self.assertAllEqual(["D" + str(i + 5) for i in range(5)],
                            sess.run(get_next))


if __name__ == "__main__":
  test.main()
