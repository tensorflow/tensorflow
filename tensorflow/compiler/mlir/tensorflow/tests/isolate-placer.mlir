// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s --run-tf-graph-optimization="graph-passes=IsolatePlacerInspectionRequiredOpsPass" | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {
func.func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.VarHandleOp"() {container = "c", shared_name = "n"} : () -> tensor<!tf_type.resource<tensor<8xf32>>>
    %1:2 = tf_executor.island wraps "tf.StatefulPartitionedCall"(%0#0) {Tin = ["tfdtype$DT_RESOURCE"], Tout = ["tfdtype$DT_RESOURCE"], config = "", config_proto = "", executor_type = "", f = @foo} : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<!tf_type.resource<tensor<8xf32>>> loc("call_foo")
    tf_executor.fetch
  }
  func.return
}

func.func @foo(%arg0: tensor<!tf_type.resource>) -> tensor<!tf_type.resource> {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<!tf_type.resource>
  }
  func.return %graph : tensor<!tf_type.resource>
}
}

// The IsolatePlacerInspectionRequiredOpsPass adds Identities for each input/output of function-calling ops.

// Capture the result of input to function call.
// CHECK: [[VARIABLE_REG:%.*]], [[VARIABLE_REG_control:%.*]] = tf_executor.island wraps "tf.VarHandleOp"()

// Test for the presence of Identity op between input and function call.
// CHECK: [[IDENTITY_REG:%.*]], [[IDENTITY_REG_control:%.*]] = tf_executor.island wraps "tf.Identity"([[VARIABLE_REG]])

// CHECK: [[CALL_RESULT_REG:%.*]], [[CALL_RESULT_REG_control:%.*]] = tf_executor.island wraps "tf.StatefulPartitionedCall"([[IDENTITY_REG]])
// CHECK-SAME: f = @[[FUNCTION:[a-zA-Z0-9_]*]]

// Match the inserted Identity op for call output.
// CHECK: "tf.Identity"([[CALL_RESULT_REG]])

// Match the function name
// CHECK: func private @[[FUNCTION]]
