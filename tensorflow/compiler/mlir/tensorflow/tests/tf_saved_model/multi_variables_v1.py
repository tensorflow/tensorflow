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

# RUN: %p/multi_variables_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# CHECK: "tf_saved_model.global_tensor"() {is_mutable, sym_name = "[[VAR0:[a-zA-Z_0-9]+]]", type = tensor<5x3xf32>, value = {{.*}} : tensor<5x3xf32>} : () -> ()
# CHECK: "tf_saved_model.global_tensor"() {is_mutable, sym_name = "[[VAR1:[a-zA-Z_0-9]+]]", type = tensor<3x5xf32>, value = {{.*}} : tensor<3x5xf32>} : () -> ()
# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME:   [[ARG0:%.*]]: tensor<!tf.resource<tensor<5x3xf32>>> {tf_saved_model.bound_input = @[[VAR0]]},
# CHECK-SAME:   [[ARG1:%.*]]: tensor<!tf.resource<tensor<3x5xf32>>> {tf_saved_model.bound_input = @[[VAR1]]})
# CHECK-SAME:             -> (tensor<5x5xf32> {{{.*}}})
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key"]

# CHECK-NEXT: [[R0:%.*]] = "tf.ReadVariableOp"([[ARG0]]) {{{.*}}} : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
# CHECK-NEXT: [[R1:%.*]] = "tf.ReadVariableOp"([[ARG1]]) {{{.*}}} : (tensor<!tf.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
# CHECK-NEXT: [[R2:%.*]] = "tf.MatMul"([[R0]], [[R1]]) {{{.*}}} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>


def Test():

  x = tf.compat.v1.get_variable(
      name='x',
      shape=(5, 3),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  y = tf.compat.v1.get_variable(
      name='y',
      shape=(3, 5),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  z = tf.matmul(x, y)
  tensor_info_z = tf.compat.v1.saved_model.utils.build_tensor_info(z)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs=None,
          outputs={'z': tensor_info_z},
          method_name='some_function'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
