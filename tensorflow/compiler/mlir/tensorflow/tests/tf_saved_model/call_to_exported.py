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

# RUN: %p/call_to_exported | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    self.v = tf.Variable(42.0)

  # We guarantee that there are no calls to exported functions from inside the
  # module.
  #
  # If there is a call to an exported function, we create a wrapper function
  # that forwards to the other function and put the tf_saved_model attributes on
  # the wrapper.
  #
  # The reason for doing this is so that later interprocedural passes don't have
  # to worry about what to do with these attributes.
  # An example of where this would happen is when converting to XLA, which
  # requires eliminating mutable variables (and is thus sort of like an
  # interprocedural SSA formation, which in particular will
  # modify signatures interprocedurally).
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<!tf_type.resource<{{.*}}>>
  # CHECK-SAME: ) -> (
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes{{.*}}tf_saved_model.exported_names = ["callee"]
  # CHECK:        "tf.StatefulPartitionedCall"{{.*}}f = @[[CALLEE_INTERNAL:[a-zA-Z_0-9]+]]
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<!tf_type.resource<{{.*}}>>
  # CHECK-SAME: ) -> (
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<*xf32> {tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes{{.*}}tf_saved_model.exported_names = ["caller"]
  # CHECK:        "tf.StatefulPartitionedCall"{{.*}}f = @[[CALLEE_INTERNAL]]
  #
  # CHECK:      func private @[[CALLEE_INTERNAL]]
  # CHECK-NOT:    tf_saved_model.exported_names

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def callee(self, x):
    return x, self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def caller(self, x):
    return self.callee(x)


if __name__ == '__main__':
  common.do_test(TestModule)
