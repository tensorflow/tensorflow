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

# RUN: %p/multi_arguments_results_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.python.ops import array_ops

# Tests multiple inputs and outputs with index paths.

# CHECK-LABEL:      func @key(
# CHECK-SAME:   %[[ARG0:.*]]: tensor<3x5xf32> {tf_saved_model.index_path = ["y"]}
# CHECK-SAME:   %[[ARG1:.*]]: tensor<5x3xf32> {tf_saved_model.index_path = ["x"]}
# CHECK-SAME:                  tensor<3x3xf32> {tf_saved_model.index_path = ["t"]}
# CHECK-SAME:                  tensor<5x5xf32> {tf_saved_model.index_path = ["s"]}
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key"]
# CHECK-DAG: %[[MUL0:.*]] = "tf.MatMul"(%[[ARG1]], %[[ARG0]])
# CHECK-DAG: %[[MUL1:.*]] = "tf.MatMul"(%[[ARG0]], %[[ARG1]])
# CHECK:  %[[IDENTITY:.*]]:2 = "tf.IdentityN"(%[[MUL1]], %[[MUL0]])
# CHECK: return %[[IDENTITY]]#0, %[[IDENTITY]]#1

# CHECK-LABEL:      func @key2(
# CHECK-SAME:   %[[ARG1:.*]]: tensor<5x3xf32> {tf_saved_model.index_path = ["b"]}
# CHECK-SAME:   %[[ARG0:.*]]: tensor<3x5xf32> {tf_saved_model.index_path = ["a"]}
# CHECK-SAME:                  tensor<5x5xf32> {tf_saved_model.index_path = ["d"]}
# CHECK-SAME:                  tensor<3x3xf32> {tf_saved_model.index_path = ["c"]}
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key2"]
# CHECK-DAG: %[[MUL1:.*]] = "tf.MatMul"(%[[ARG0]], %[[ARG1]])
# CHECK-DAG: %[[MUL2:.*]] = "tf.MatMul"(%[[ARG1]], %[[ARG0]])
# CHECK:  %[[IDENTITY:.*]]:2 = "tf.IdentityN"(%[[MUL1]], %[[MUL2]])
# CHECK: return %[[IDENTITY]]#1, %[[IDENTITY]]#0


def Test():

  x = tf.constant(1.0, shape=(5, 3))
  y = tf.constant(1.0, shape=(3, 5))

  s = tf.matmul(x, y)
  t = tf.matmul(y, x)
  [t, s] = array_ops.identity_n([t, s])

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
  tensor_info_s = tf.compat.v1.saved_model.utils.build_tensor_info(s)
  tensor_info_t = tf.compat.v1.saved_model.utils.build_tensor_info(t)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={
              'x': tensor_info_x,
              'y': tensor_info_y
          },
          outputs={
              's': tensor_info_s,
              't': tensor_info_t
          },
          method_name='some_function')),
      'key2': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={
              'a': tensor_info_y,
              'b': tensor_info_x,
          },
          outputs={
              'c': tensor_info_t,
              'd': tensor_info_s,
          },
          method_name='reverse_arguments'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
