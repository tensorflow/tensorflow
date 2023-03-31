# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# RUN: %p/include_variables_in_init_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-SAME: tf.versions
# CHECK-SAME: bad_consumers
# CHECK-SAME: min_consumer
# CHECK-SAME: producer

# CHECK: "tf_saved_model.global_tensor"()
# CHECK: "tf_saved_model.session_initializer"() {initializers = [@[[INIT_FUNC:[a-zA-Z_0-9]+]]]} : () -> ()

# Initializer function. This should contain the initialization sequence for the
# variable.
# CHECK: func @[[INIT_FUNC]](%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y}) attributes {
# CHECK-SAME: tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init"]
# CHECK-SAME: tf_saved_model.initializer_type = "init_op"
# CHECK-SAME: }
# CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() {{{.*dense<.*> : tensor<2xi32>.*}}} : () -> tensor<2xi32>
# CHECK: %[[RAND_STD_NORMAL:.*]] = "tf.RandomStandardNormal"(%[[CST_0]])
# CHECK: "tf.AssignVariableOp"(%[[ARG_0]], %[[RAND_STD_NORMAL]]){{.*}}: (tensor<!tf_type.resource<tensor<1x3xf32>>>, tensor<1x3xf32>) -> ()
# CHECK: return

# The function for the signature "key".
# CHECK: func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME: %[[ARG_1:.*]]: tensor<3x1xf32> {tf_saved_model.index_path = ["x"]}
# CHECK-SAME: %[[ARG_2:.*]]: tensor<!tf_type.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y}
# CHECK-SAME: -> (tensor<3x3xf32> {tf_saved_model.index_path = ["r"]})
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key"]
# CHECK-NEXT: %[[READ_VAR_0:.*]] = "tf.ReadVariableOp"(%[[ARG_2]]) {{{.*}}} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
# CHECK-NEXT: %[[MATMUL_0:.*]] = "tf.MatMul"(%[[ARG_1]], %[[READ_VAR_0]]) {{{.*}}} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
# CHECK-NEXT: return %[[MATMUL_0]] : tensor<3x3xf32>


def Test():
  x = tf.constant([[1.0], [1.0], [1.0]])
  y = tf.compat.v1.get_variable(
      name='y',
      shape=(1, 3),
      initializer=tf.random_normal_initializer(),
      trainable=True,
  )
  r = tf.matmul(x, y)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return (
      {
          'key': (
              tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                  inputs={'x': tensor_info_x},
                  outputs={'r': tensor_info_r},
                  method_name='some_function',
              )
          )
      },
      tf.initializers.global_variables(),
      None,
  )


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test, include_variables_in_initializers=True)
