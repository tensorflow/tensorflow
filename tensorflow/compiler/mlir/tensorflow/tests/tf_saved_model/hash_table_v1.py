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

# RUN: %p/hash_table_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# CHECK: "tf_saved_model.session_initializer"() {initializers = [@[[init:.*]]]} : () -> ()

# CHECK:      func @[[init]]
# CHECK-NEXT: [[R5:%.*]] = "tf.Const"()
# CHECK-NEXT: [[R6:%.*]] = "tf.Const"()
# CHECK-NEXT: [[R7:%.*]] = "tf.HashTableV2"()
# CHECK-SAME: shared_name = "[[hash_table:.*]]"
# CHECK-NEXT: "tf.LookupTableImportV2"([[R7]], [[R5]], [[R6]])

# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME: [[ARG0:%.*]]: tensor<i32>
# CHECK-SAME: [[ARG1:%.*]]: tensor<!tf.resource
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key"]

# CHECK-NEXT: [[R0:%.*]] = "tf.Const"()
# CHECK-NEXT: [[R1:%.*]] = "tf.HashTableV2"()
# CHECK-SAME: shared_name = "[[hash_table]]"
# CHECK-NEXT: [[R2:%.*]] = "tf.LookupTableFindV2"([[R1]], [[ARG0]], [[R0]])
# CHECK-NEXT: [[R3:%.*]] = "tf.ReadVariableOp"([[ARG1]])
# CHECK-NEXT: [[R4:%.*]] = "tf.AddV2"([[R2]], [[R3]])
# CHECK-NEXT: return [[R4]]


def Test():

  z = tf.compat.v1.get_variable(
      name='y',
      shape=(),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  table_initializer = tf.lookup.KeyValueTensorInitializer(
      keys=[1, 2, 3, 4],
      values=[5, 6, 7, 8],
      key_dtype=tf.int32,
      value_dtype=tf.float32)
  table = tf.lookup.StaticHashTable(
      table_initializer, default_value=tf.constant(0.0))

  x = tf.placeholder(tf.int32, shape=(), name='input')
  y = table.lookup(x)
  r = tf.add(y, z)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, tf.tables_initializer(), None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test, canonicalize=True)
