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

# RUN: %p/control_flow_upgrade_legacy_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.python.ops import control_flow_ops

# Tests V1 control flow is functionalized.

# CHECK-NOT: tf_executor.Switch
# CHECK-NOT: tf_executor.Merge
# CHECK: "tf.If"
# CHECK-SAME: else_branch = @"key/[[else:[a-zA-Z_0-9]+]]"
# CHECK-SAME: then_branch = @"key/[[then:[a-zA-Z_0-9]+]]"

# CHECK: func private @"key/[[else]]"(
# CHECK: func private @"key/[[then]]"(


def Test():
  data = tf.constant([1, 2, 3, 4, 5, 6])
  # Create placeholders to prevent constant folding.
  x_op = tf.placeholder(dtype=tf.int32)
  y_op = tf.placeholder(dtype=tf.int32)
  less_op = tf.less(x_op, y_op)
  switch_op = control_flow_ops.switch(data, less_op)
  merge_op = control_flow_ops.merge(switch_op)[0]
  result = tf.transpose(merge_op)

  tensor_info_result = tf.compat.v1.saved_model.utils.build_tensor_info(result)

  signature_def = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=None,
      outputs={'result': tensor_info_result},
      method_name='some_function')

  return {'key': signature_def}, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
