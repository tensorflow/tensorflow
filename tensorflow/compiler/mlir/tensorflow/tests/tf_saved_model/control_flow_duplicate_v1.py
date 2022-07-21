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

# RUN: %p/control_flow_duplicate_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# Tests handling dupliate functions after V1 control flow is functionalized.

# CHECK:  func {{.*}} tf_saved_model.exported_names = ["key_1"]
# CHECK: "tf.If"
# CHECK-SAME: else_branch = @[[else:[a-zA-Z_0-9]+]]
# CHECK-SAME: then_branch = @[[then:[a-zA-Z_0-9]+]]

# CHECK:  func {{.*}} tf_saved_model.exported_names = ["key_2"]
# CHECK: "tf.If"
# CHECK-SAME: else_branch = @[[else]]
# CHECK-SAME: then_branch = @[[then]]

# CHECK: func private @[[else]](
# CHECK: func private @[[then]](


def Test():

  zero = tf.constant(0)
  one = tf.constant(1)
  x = tf.placeholder(tf.int32, shape=(), name='input')
  result = tf.cond(x > zero, lambda: tf.square(x), lambda: tf.add(x, one))

  tensor_info_result = tf.compat.v1.saved_model.utils.build_tensor_info(result)

  signature_def = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=None,
      outputs={'result': tensor_info_result},
      method_name='some_function')

  return {'key_1': signature_def, 'key_2': signature_def}, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
