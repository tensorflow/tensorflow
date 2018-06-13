# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for KinesisDataset.
NOTE: boto3 is needed and the test has to be invoked manually:
```
$ bazel test -s --verbose_failures --config=opt \
    --action_env=AWS_ACCESS_KEY_ID=XXXXXX       \
    --action_env=AWS_SECRET_ACCESS_KEY=XXXXXX   \
    //tensorflow/contrib/kinesis:kinesis_test
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import boto3

from tensorflow.contrib.kinesis.python.ops import kinesis_dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class KinesisDatasetTest(test.TestCase):

  def testKinesisDatasetOneShard(self):
    client = boto3.client('kinesis', region_name='us-east-1')

    # Setup the Kinesis with 1 shard.
    stream_name = "tf_kinesis_test_1"
    client.create_stream(StreamName=stream_name, ShardCount=1)
    # Wait until stream exists, default is 10 * 18 seconds.
    client.get_waiter('stream_exists').wait(StreamName=stream_name)
    for i in range(10):
      data = "D" + str(i)
      client.put_record(
          StreamName=stream_name, Data=data, PartitionKey="TensorFlow" + str(i))

    stream = array_ops.placeholder(dtypes.string, shape=[])
    num_epochs = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kinesis_dataset_ops.KinesisDataset(
        stream, read_indefinitely=False).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    init_batch_op = iterator.make_initializer(batch_dataset)
    get_next = iterator.get_next()

    with self.test_session() as sess:
      # Basic test: read from shard 0 of stream 1.
      sess.run(init_op, feed_dict={stream: stream_name, num_epochs: 1})
      for i in range(10):
        self.assertEqual("D" + str(i), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

    client.delete_stream(StreamName=stream_name)
    # Wait until stream deleted, default is 10 * 18 seconds.
    client.get_waiter('stream_not_exists').wait(StreamName=stream_name)

  def testKinesisDatasetTwoShards(self):
    client = boto3.client('kinesis', region_name='us-east-1')

    # Setup the Kinesis with 2 shards.
    stream_name = "tf_kinesis_test_2"
    client.create_stream(StreamName=stream_name, ShardCount=2)
    # Wait until stream exists, default is 10 * 18 seconds.
    client.get_waiter('stream_exists').wait(StreamName=stream_name)

    for i in range(10):
      data = "D" + str(i)
      client.put_record(
          StreamName=stream_name, Data=data, PartitionKey="TensorFlow" + str(i))
    response = client.describe_stream(StreamName=stream_name)
    shard_id_0 = response["StreamDescription"]["Shards"][0]["ShardId"]
    shard_id_1 = response["StreamDescription"]["Shards"][1]["ShardId"]

    stream = array_ops.placeholder(dtypes.string, shape=[])
    shard = array_ops.placeholder(dtypes.string, shape=[])
    num_epochs = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    repeat_dataset = kinesis_dataset_ops.KinesisDataset(
        stream, shard, read_indefinitely=False).repeat(num_epochs)
    batch_dataset = repeat_dataset.batch(batch_size)

    iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
    init_op = iterator.make_initializer(repeat_dataset)
    init_batch_op = iterator.make_initializer(batch_dataset)
    get_next = iterator.get_next()

    data = list()
    with self.test_session() as sess:
      # Basic test: read from shard 0 of stream 2.
      sess.run(
          init_op, feed_dict={
              stream: stream_name, shard: shard_id_0, num_epochs: 1})
      with self.assertRaises(errors.OutOfRangeError):
        # Use range(11) to guarantee the OutOfRangeError.
        for i in range(11):
          data.append(sess.run(get_next))

      # Basic test: read from shard 1 of stream 2.
      sess.run(
          init_op, feed_dict={
              stream: stream_name, shard: shard_id_1, num_epochs: 1})
      with self.assertRaises(errors.OutOfRangeError):
        # Use range(11) to guarantee the OutOfRangeError.
        for i in range(11):
          data.append(sess.run(get_next))

    data.sort()
    self.assertEqual(data, ["D" + str(i) for i in range(10)])

    client.delete_stream(StreamName=stream_name)
    # Wait until stream deleted, default is 10 * 18 seconds.
    client.get_waiter('stream_not_exists').wait(StreamName=stream_name)


if __name__ == "__main__":
  test.main()
