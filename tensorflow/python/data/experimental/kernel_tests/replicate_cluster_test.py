# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Graph mode cluster tests for the experimental `replicate` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class ReplicateClusterTest(test_base.DatasetTestBase, parameterized.TestCase):

  def setUp(self):
    super(ReplicateClusterTest, self).setUp()
    # Start the local server.
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        3, 0, worker_config=worker_config)
    self._device0 = "/job:worker/replica:0/task:0/device:CPU:0"
    self._device1 = "/job:worker/replica:0/task:1/device:CPU:0"
    self._device2 = "/job:worker/replica:0/task:2/device:CPU:0"
    self._target = worker[0].target

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testBasic(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      get_next = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    with session.Session(self._target) as sess:
      for i in range(100):
        self.assertEqual(i, sess.run(get_next()))
        self.assertEqual(i, sess.run(get_next1()))
        self.assertEqual(i, sess.run(get_next2()))

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testMap(self):
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      get_next = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    with session.Session(self._target) as sess:
      for i in range(100):
        self.assertEqual(i * 2, sess.run(get_next()))
        self.assertEqual(i * 2, sess.run(get_next1()))
        self.assertEqual(i * 2, sess.run(get_next2()))

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testVariableInput(self):
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    with ops.device(self._device1):
      it1 = dataset_ops.make_initializable_iterator(dataset1)
    # We don't support stateful ops across processes in functions as of now.
    with session.Session(self._target) as sess:
      with self.assertRaises(errors.OpError):
        sess.run(it1.initializer)


if __name__ == "__main__":
  test.main()
