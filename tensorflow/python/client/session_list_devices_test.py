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

"""Tests for tensorflow.python.client.session.Session's list_devices API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib

ops._USE_C_API = True


class SessionListDevicesTest(test_util.TensorFlowTestCase):

  @test_util.disable_c_api  # list_devices doesn't work with C API
  def testListDevices(self):
    with session.Session() as sess:
      devices = sess.list_devices()
      self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in set(
          [d.name for d in devices]), devices)
      self.assertGreaterEqual(1, len(devices), devices)

  @test_util.disable_c_api  # list_devices doesn't work with C API
  def testListDevicesGrpcSession(self):
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess:
      devices = sess.list_devices()
      self.assertTrue('/job:local/replica:0/task:0/device:CPU:0' in set(
          [d.name for d in devices]), devices)
      self.assertGreaterEqual(1, len(devices), devices)

  @test_util.disable_c_api  # list_devices doesn't work with C API
  def testListDevicesClusterSpecPropagation(self):
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)
    with session.Session(server1.target, config=config) as sess:
      devices = sess.list_devices()
      device_names = set([d.name for d in devices])
      self.assertTrue(
          '/job:worker/replica:0/task:0/device:CPU:0' in device_names)
      self.assertTrue(
          '/job:worker/replica:0/task:1/device:CPU:0' in device_names)
      self.assertGreaterEqual(2, len(devices), devices)


if __name__ == '__main__':
  googletest.main()
