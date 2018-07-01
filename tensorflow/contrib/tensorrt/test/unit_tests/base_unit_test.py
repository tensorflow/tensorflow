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
"""Base class to facilitate development of integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwpb2
from tensorflow.python.client import session as csess
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes as dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.ops import array_ops as aops
from tensorflow.python.ops import nn as nn
from tensorflow.python.ops import nn_ops as nn_ops

class BaseUnitTest(object):
  """Base class for unit tests in TF-TRT"""

  def __init__(self, log_file='log.txt'):
    self.static_mode_list = {}
    self.dynamic_mode_list = {}
    self.dummy_input = None
    self.get_network = None
    self.expect_nb_nodes = None
    self.test_name = None
    self.log_file = log_file
    self.ckpt = None
    self.allclose_rtol = 0.01
    self.allclose_atol = 0.01
    self.allclose_equal_nan = True
    # saves out graphdef
    self.debug = False
    # require node count check fail leads to test failure
    self.check_node_count = False

  def run(self, run_test_context):
    run_test_context.run_test(self.get_network, self.static_mode_list, self.dynamic_mode_list, self.dummy_input, self.ckpt)
    return self.log_result(run_test_context)

  def log_result(self, run_test_result):
    log = open(self.log_file, 'a')
    log.write(("================= model: %s\n")%(self.test_name))

    if self.debug:
      open(self.test_name+"_native.pb", 'wb').write(run_test_result.native_network.SerializeToString())
    all_success = True
    if len(run_test_result.tftrt_conversion_flag) != 0:
      log.write("  -- static_mode\n")
    for static_mode in run_test_result.tftrt_conversion_flag:
      if self.debug:
        open(self.test_name+"_"+static_mode+".pb", 'wb').write(run_test_result.tftrt[static_mode].SerializeToString())
      log.write("     ----\n")
      log.write(("     mode: [%s]\n")%(static_mode))
      if run_test_result.tftrt_conversion_flag[static_mode]:
        if run_test_result.tftrt_nb_nodes[static_mode] != self.expect_nb_nodes:
          log.write(("[WARNING]: converted node number does not match (%d,%d,%d)!!!\n")%(run_test_result.tftrt_nb_nodes[static_mode], self.expect_nb_nodes, run_test_result.native_nb_nodes))
          if self.check_node_count:
            all_success = False

        if np.array_equal(run_test_result.tftrt_result[static_mode], run_test_result.native_result):
          log.write("     output: equal\n")
        elif np.allclose(run_test_result.tftrt_result[static_mode], run_test_result.native_result, atol=self.allclose_atol, rtol=self.allclose_rtol, equal_nan=self.allclose_equal_nan):
          log.write("     output: allclose\n")
        else:
          diff = run_test_result.tftrt_result[static_mode]-run_test_result.native_result
          log.write("[ERROR]: output does not match!!!\n")
          log.write( "max diff: " +str(np.max(diff)))
          log.write( "\ntftrt:\n")
          log.write(str(run_test_result.tftrt_result[static_mode]))
          log.write( "\nnative:\n")
          log.write(str(run_test_result.native_result))
          log.write( "\ndiff:\n")
          log.write(str(diff))
          all_success = False
      else:
        log.write("[ERROR]: conversion failed!!!\n")
        all_success = False

    if len(run_test_result.tftrt_dynamic_conversion_flag) != 0:
      log.write("  -- dynamic_mode\n")
    for dynamic_mode in run_test_result.tftrt_dynamic_conversion_flag:
      log.write("\n     ----\n")
      log.write(("     mode: [%s]\n")%(dynamic_mode))
      if run_test_result.tftrt_dynamic_conversion_flag[dynamic_mode]:
        if np.array_equal(run_test_result.tftrt_dynamic_result[dynamic_mode], run_test_result.native_result):
          log.write("     output: equal\n")
        elif np.allclose(run_test_result.tftrt_dynamic_result[dynamic_mode], run_test_result.native_result):
          log.write("     output: allclose\n")
        else:
          log.write("[ERROR]: output does not match!!!\n")
          all_success = False
      else:
        log.write("[ERROR]: conversion failed!!!\n")
        all_success = False
    return all_success
