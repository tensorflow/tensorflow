// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-tfrt-opt -pass-pipeline='builtin.module(tf-executor-to-tfrt-pipeline{target-tpurt=true})' %s | FileCheck %s

module attributes {tf_saved_model.semantics}  {
  // CHECK-LABEL: func @main
  func.func @main_func() -> (tensor<*xf32> {tf_saved_model.index_path = ["a"]}) attributes {tf_saved_model.exported_names = ["main_func"]} {
    %0 = tf_executor.graph {
      %outputs_0, %control_0 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = ""} : () -> tensor<!tf_type.resource<tensor<501000x128xf32>>>
      %outputs_1, %control_1 = tf_executor.island wraps "tf.Cast"(%outputs_0) {Truncate = false} : (tensor<!tf_type.resource<tensor<501000x128xf32>>>) -> tensor<*x!tf_type.resource>

      // CHECK: tfrt_fallback_async.batch_function device([[DEVICE:.*]]) @batched_func ([[BATCHED_FUNC_ARG:%.*]])
      // CHECK-SAME: Tcaptured = [!corert.resource]
      // CHECK-SAME: Tin = []
      // CHECK-SAME: Tout = [f32]
      %outputs_2, %control_2 = tf_executor.island wraps "tf.BatchFunction"(%outputs_1) {batch_timeout_micros = 5000 : i64, batching_queue = "", container = "", f = @batched_func, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 0, 1>, shared_name = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
      tf_executor.fetch %outputs_2 : tensor<*xf32>
    }
    func.return %0 : tensor<*xf32>
  }
  func.func private @batched_func(%arg0: tensor<*x!tf_type.resource>) -> tensor<?xf32> {
    %0 = tf_executor.graph {
      %outputs_0, %control_0 = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource>) -> tensor<?xf32>
      tf_executor.fetch %outputs_0 : tensor<?xf32>
    }
    func.return %0 : tensor<?xf32>
  }
}
