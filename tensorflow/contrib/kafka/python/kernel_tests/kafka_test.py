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
"""Tests for KafkaReader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.kafka.python.ops import kafka_reader_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import queue_runner_impl

class KafkaReaderTest(test.TestCase):

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

  def testBasic(self):
    filename_queue = input_lib.string_input_producer(["test:0:0:10"])
    reader = kafka_reader_ops.KafkaReader(group="test", eof=True)
    key, value = reader.read(filename_queue)
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)
      for i in range(10):
        v = sess.run([key, value])
        self.assertAllEqual(v, [str(i), 'D'+str(i)])
      coord.request_stop()
      coord.join(threads)

if __name__ == "__main__":
  test.main()
