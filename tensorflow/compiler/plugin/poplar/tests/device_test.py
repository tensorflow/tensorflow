# Copyright 2017 Graphcore Ltd
#

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import device_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class IpuXlaDeviceLibTest(test_util.TensorFlowTestCase):

    def testLoadDevice(self):
        devices = device_lib.list_local_devices()

        self.assertGreater(len(devices), 0)
        self.assertEqual(devices[0].device_type, "CPU")

        found_ipu = False
        for d in devices:
            if d.device_type == "IPU":
                found_ipu = True

        self.assertTrue(found_ipu)

if __name__ == "__main__":
    googletest.main()
