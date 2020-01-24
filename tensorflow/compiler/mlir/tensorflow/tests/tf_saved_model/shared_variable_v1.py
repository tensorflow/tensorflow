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

# RUN: %p/shared_variable_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# CHECK: "tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<1x3xf32>, value = {{.*}} : tensor<1x3xf32>} : () -> ()
# CHECK: func {{@.*}}([[ARG0:%.*]]: tensor<3x1xf32>,
# CHECK-SAME: [[ARG1:%.*]]: tensor<!tf.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y}) -> tensor<3x3xf32>

# CHECK: func {{@.*}}([[ARG2:%.*]]: tensor<3x1xf32>,
# CHECK-SAME: [[ARG3:%.*]]: tensor<!tf.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y}) -> tensor<3x3xf32>


def Test():

  # Default TF1.x uses reference variables that are not supported by SavedModel
  # v1 Importer. To use SavedModel V1 Importer, resource variables should be
  # enabled.
  tf.enable_resource_variables()

  tf.compat.v1.disable_eager_execution()

  x = tf.constant([[1.0], [1.0], [1.0]])
  y = tf.get_variable(
      name='y',
      shape=(1, 3),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  r = tf.matmul(x, y)

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.saved_model.utils.build_tensor_info(r)

  signature_def = tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'x': tensor_info_x},
      outputs={'r': tensor_info_r},
      method_name=tf.saved_model.PREDICT_METHOD_NAME)

  # Create two signatures that share the same variable.
  return {'basic': signature_def, 'basic_2': signature_def}


if __name__ == '__main__':
  common_v1.do_test(Test())
