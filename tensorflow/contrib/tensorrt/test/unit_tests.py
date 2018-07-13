# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Script to execute and log all integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensorrt.test.batch_matmul_test import BatchMatMulTest
from tensorflow.contrib.tensorrt.test.biasadd_matmul_test import BiasaddMatMulTest
from tensorflow.contrib.tensorrt.test.binary_tensor_weight_broadcast_test import BinaryTensorWeightBroadcastTest
from tensorflow.contrib.tensorrt.test.concatenation_test import ConcatenationTest
from tensorflow.contrib.tensorrt.test.multi_connection_neighbor_engine_test import MultiConnectionNeighborEngineTest
from tensorflow.contrib.tensorrt.test.neighboring_engine_test import NeighboringEngineTest
from tensorflow.contrib.tensorrt.test.unary_test import UnaryTest
from tensorflow.contrib.tensorrt.test.vgg_block_nchw_test import VGGBlockNCHWTest
from tensorflow.contrib.tensorrt.test.vgg_block_test import VGGBlockTest
from tensorflow.contrib.tensorrt.test.const_broadcast_test import ConstBroadcastTest

from tensorflow.contrib.tensorrt.test.run_test import RunTest

tests = 0
passed_test = 0

failed_list = []
test_list = []

test_list.append(BatchMatMulTest())
test_list.append(BiasaddMatMulTest())
test_list.append(BinaryTensorWeightBroadcastTest())
test_list.append(ConcatenationTest())
test_list.append(NeighboringEngineTest())
test_list.append(UnaryTest())
test_list.append(VGGBlockNCHWTest())
test_list.append(VGGBlockTest())
test_list.append(MultiConnectionNeighborEngineTest())
test_list.append(ConstBroadcastTest())

for test in test_list:
  test.debug = True
  test.check_node_count = False
  with RunTest() as context:
    tests += 1
    if test.run(context):
      passed_test += 1
    else:
      failed_list.append(test.test_name)
      print("Failed test: %s\n", test.test_name)

if passed_test == tests:
  print("Passed\n")
else:
  print(("%d out of %d passed\n  -- failed list:") % (passed_test, tests))
  for test in failed_list:
    print("      - " + test)
