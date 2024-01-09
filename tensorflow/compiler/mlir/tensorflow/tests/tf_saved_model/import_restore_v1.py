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

# RUN: %p/import_restore_v1 | FileCheck %s

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

# CHECK: tf_saved_model.session_initializer
# CHECK-SAME: initializers = [@[[restore:.*]]]

# CHECK: "tf_saved_model.asset"()
# CHECK-SAME: <{filename = [[filename:.*]], sym_name = "[[sym_name:.*]]"}> : () -> ()

# CHECK:      func @[[restore]](
# CHECK-SAME:   [[variable_path:%.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @[[sym_name]]}
# CHECK-SAME: tf_saved_model.exported_names = ["{{__tf_saved_model_session_initializer.*}}"]
# CHECK-SAME: tf_saved_model.initializer_type = "restore_op"
# CHECK: [[v0:%.*]] = "tf.RestoreV2"([[variable_path]]
# CHECK: [[v1:%.*]] = "tf.Identity"([[v0]])
# CHECK: [[handle:%.*]] = "tf.VarHandleOp"
# CHECK-SAME: shared_name = [[shared_name:".*"]]
# CHECK: "tf.AssignVariableOp"([[handle]], [[v1]])

# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME: tf_saved_model.exported_names = ["key"]
# CHECK: tf.VarHandleOp
# CHECK-SAME: shared_name = [[shared_name]]


def Test():

  x = tf.constant([[1.0], [1.0], [1.0]])
  y = tf.compat.v1.get_variable(
      name='y',
      shape=(1, 3),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  r = tf.matmul(x, y)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test, use_lite=True)
