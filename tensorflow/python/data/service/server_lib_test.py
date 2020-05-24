# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.data service server lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.service import server_lib

from tensorflow.python.platform import test

PROTOCOL = "grpc"


class ServerLibTest(test.TestCase):

  def testStartMaster(self):
    master = server_lib.MasterServer(PROTOCOL)
    self.assertRegex(master.target, PROTOCOL + "://.*:.*")

  def testStartWorker(self):
    master = server_lib.MasterServer(PROTOCOL)
    worker = server_lib.WorkerServer(PROTOCOL,
                                     master.target[len(PROTOCOL + "://"):])
    self.assertRegex(worker.target, PROTOCOL + "://.*:.*")


if __name__ == "__main__":
  test.main()
