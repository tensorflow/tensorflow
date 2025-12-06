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

# RUN: %p/basic | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-SAME: tf.versions
# CHECK-SAME: bad_consumers
# CHECK-SAME: min_consumer
# CHECK-SAME: producer


class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    self.v42 = tf.Variable(42.0)
    # Use convert_to_tensor to avoid forcing eager `.numpy()` in graph/XLA mode.
    self.c43 = tf.convert_to_tensor(43.0, dtype=tf.float32)

  # During serialization, the constants are given internal (non-user-accessible, non-semantically-load-bearing) exported names.
  # CHECK: "tf_saved_model.global_tensor"() <{sym_name = "[[CONST:[a-zA-Z_0-9.]+]]", type = tensor<f32>, value = dense<4.300000e+01> : tensor<f32>}> {tf_saved_model.exported_names = [{{.*}}]} : () -> ()

  # CHECK: "tf_saved_model.global_tensor"() <{is_mutable, sym_name = "[[VAR:[a-zA-Z_0-9]+]]", type = tensor<f32>, value = dense<4.200000e+01> : tensor<f32>}> {tf_saved_model.exported_names = ["v42"]} : () -> ()
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   %arg2: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = []})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["some_function"]
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return x + self.v42 + self.c43

  # Test for robust graph/XLA-friendly tensor operations avoiding Python built-ins
  # CHECK: func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<3x?xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<?xf32> {tf_saved_model.index_path = []})
  @tf.function(input_signature=[tf.TensorSpec([3, None, 16], tf.float32)])
  def robust_concat_function(self, x):
    # Stack candidate feature tensors
    candidates = tf.stack([x, x * 2, x * 3], axis=0)  # shape (3, batch, 16)
    # Compute scalar sums per candidate
    sums = tf.reduce_sum(candidates, axis=[1, 2])  # shape (3,)
    # Create mask
    mask = sums > 0.5  # shape (3,)
    # Select filtered candidates
    filtered = tf.boolean_mask(candidates, mask, axis=0)  # shape (k, batch, 16)
    # Apply mapping
    mapped = tf.nn.sigmoid(filtered)  # shape (k, batch, 16)
    # Prepare ones_like part
    ones = tf.ones_like(x, dtype=x.dtype)  # shape (batch, 16)
    ones_tiled = tf.tile(tf.expand_dims(ones, 0), [tf.shape(mapped)[0], 1, 1])  # shape (k, batch, 16)
    # Concatenate per candidate along last axis
    per_candidate = tf.concat([mapped, ones_tiled], axis=-1)  # shape (k, batch, 32)
    # Unstack and concatenate across candidate axis
    per_candidate_list = tf.unstack(per_candidate, axis=0)  # list of (batch, 32)
    combined = tf.concat(per_candidate_list, axis=-1)  # shape (batch, 32 * k)
    return combined
