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

# RUN: %p/structured_input | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # The fNNNN name prefixes in this file are such that the sorted order of the
  # functions in the resulting MLIR output match the order in the source file,
  # allowing us to conveniently co-locate the CHECK's with the code they are
  # checking.
  #
  # Note: CHECK-DAG doesn't work with CHECK-SAME/CHECK-NEXT.

  # Check index paths for arguments.
  # The outer layer of the index path indexes into the arguments.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "y", tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0000_function_arity"]
  @tf.function(input_signature=[
      tf.TensorSpec([1], tf.float32),
      tf.TensorSpec([2], tf.float32)
  ])
  def f0000_function_arity(self, x, y):
    return

  # Check index paths for lists.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "l", tf_saved_model.index_path = [0, 0]},
  # CHECK-SAME:   %arg1: tensor<f32> {tf._user_specified_name = "l", tf_saved_model.index_path = [0, 1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0001_list_2_elements"]
  @tf.function(input_signature=[[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
  ]])
  def f0001_list_2_elements(self, l):
    return

  # Check index paths for dicts.
  # Keys are linearized in sorted order, matching `tf.nest.flatten`.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0002_dict_2_keys"]
  @tf.function(input_signature=[{
      'x': tf.TensorSpec([1], tf.float32),
      'y': tf.TensorSpec([2], tf.float32),
  }])
  def f0002_dict_2_keys(self, d):
    return

  # Check index paths for dicts, where the keys are not in sorted order.
  # The index path should be insensitive to the key order.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0003_dict_2_keys_out_of_order"]
  @tf.function(input_signature=[{
      'y': tf.TensorSpec([2], tf.float32),
      'x': tf.TensorSpec([1], tf.float32),
  }])
  def f0003_dict_2_keys_out_of_order(self, d):
    return

  # Slightly stronger stress test of multiple dict keys.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "a"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "b"]},
  # CHECK-SAME:   %arg2: tensor<3xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "c"]},
  # CHECK-SAME:   %arg3: tensor<4xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg4: tensor<5xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]},
  # CHECK-SAME:   %arg5: tensor<6xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "z"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0004_dict_many_keys"]
  @tf.function(input_signature=[{
      'x': tf.TensorSpec([4], tf.float32),
      'y': tf.TensorSpec([5], tf.float32),
      'z': tf.TensorSpec([6], tf.float32),
      'a': tf.TensorSpec([1], tf.float32),
      'b': tf.TensorSpec([2], tf.float32),
      'c': tf.TensorSpec([3], tf.float32),
  }])
  def f0004_dict_many_keys(self, d):
    return

  # Check a slightly more complex recursive structure.
  # Note that list elements can have heterogenous types.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x", 0]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x", 1]},
  # CHECK-SAME:   %arg2: tensor<3xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0005_more_complex_recursive_structure"]
  @tf.function(input_signature=[{
      'x': [tf.TensorSpec([1], tf.float32),
            tf.TensorSpec([2], tf.float32)],
      'y': tf.TensorSpec([3], tf.float32),
  }])
  def f0005_more_complex_recursive_structure(self, d):
    return


if __name__ == '__main__':
  common.do_test(TestModule)
