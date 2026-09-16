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
// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @assign_variable
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain) -> !tfrt.chain
func.func @assign_variable() {
  // CHECK: [[ch1:%.*]], %results = tfrt_fallback_async.executeop.seq([[in_chain]]) key(0) cost({{.*}}) device("/device:CPU:0") "tf.VarHandleOp"
  // CHECK-NEXT: [[ch2:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(1) cost({{.*}}) device("/device:CPU:0") "tf.AssignVariableOp"
  // CHECK-NEXT: [[out_ch:%.*]] = tfrt.merge.chains [[ch1]], [[ch2]]
  // CHECK-NEXT: tfrt.return [[out_ch]]

  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.AssignVariableOp"(%1, %0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  func.return
}
