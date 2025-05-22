// RUN: litert-opt -tfl-raise-custom-ops -canonicalize %s --split-input-file | FileCheck %s
// RUN: litert-opt -tfl-raise-custom-ops="test-raise-tf-targets=tf.FakeQuantWithMinMaxVarsPerChannel" -canonicalize %s --split-input-file | FileCheck --check-prefix=WRAPPED %s

// CHECK-LABEL: custom_op
func.func @custom_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "arith.constant" () {value = dense<2.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // will be preserved since it has uses.
  %2 = "tf.MyCustomOp"(%1, %0) {fused_activation_function = "RELU", int_attr = 2 : i32}  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // will be preserved since it has side-effect.
  "tf.MyCustomOp"(%1, %0) {fused_activation_function = "RELU", int_attr = 2 : i32}  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %2 : tensor<4xf32>

// CHECK-NEXT: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
// CHECK-NEXT: %[[MUL:.*]] = tfl.mul %arg0, %[[CST]] {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK-NEXT: %[[CUSTOM_1:.*]] = "tfl.custom_tf"(%[[MUL]], %[[CST]]) ({
// CHECK-NEXT: ^bb0(%arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
// CHECK-NEXT:   %[[MY_CUSTOM:.*]] = "tf.MyCustomOp"(%arg1, %arg2) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:   "tfl.yield"(%[[MY_CUSTOM]]) : (tensor<4xf32>) -> ()
// CHECK-NEXT: }) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[CUSTOM_2:.*]] = "tfl.custom_tf"(%[[MUL]], %[[CST]]) ({
// CHECK-NEXT: ^bb0(%arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
// CHECK-NEXT:   %[[MY_CUSTOM:.*]] = "tf.MyCustomOp"(%arg1, %arg2) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:   "tfl.yield"(%[[MY_CUSTOM]]) : (tensor<4xf32>) -> ()
// CHECK-NEXT: }) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[CUSTOM_1]] : tensor<4xf32>
}

// -----

// CHECK-LABEL: tf_executor_wrapper
// WRAPPED-LABEL: tf_executor_wrapper
func.func @tf_executor_wrapper(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input", outputs = "output"}} {
  %0 = tf_executor.graph {
    %outputs_14, %control_15 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1.0> : tensor<186xf32>} : () -> tensor<186xf32>
    %outputs_16, %control_17 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<2.0> : tensor<186xf32>} : () -> tensor<186xf32>
    %outputs_18, %control_19 = tf_executor.island wraps "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %outputs_16, %outputs_14) {device = "", narrow_range = true, num_bits = 8 : i64} : (tensor<*xf32>, tensor<186xf32>, tensor<186xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs_18 : tensor<*xf32>
  }
  func.return %0 : tensor<*xf32>

// CHECK: tf_executor.island wraps "tf.FakeQuantWithMinMaxVarsPerChannel"

// WRAPPED-NEXT: tf_executor.graph {
// WRAPPED-NEXT:   tf_executor.island wraps "tf.Const"() <{value = dense<1.000000e+00> : tensor<186xf32>}> {device = ""} : () -> tensor<186xf32>
// WRAPPED-NEXT:   tf_executor.island wraps "tf.Const"() <{value = dense<2.000000e+00> : tensor<186xf32>}> {device = ""} : () -> tensor<186xf32>
// WRAPPED-NEXT:   tf_executor.island wraps "tfl.custom_tf"
// WRAPPED-NEXT:     ^bb0(%arg1: tensor<*xf32>, %arg2: tensor<186xf32>, %arg3: tensor<186xf32>):
// WRAPPED-NEXT:   %[[fq:.*]] = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg1, %arg2, %arg3) <{narrow_range = true, num_bits = 8 : i64}> {device = ""} : (tensor<*xf32>, tensor<186xf32>, tensor<186xf32>) -> tensor<*xf32>
// WRAPPED-NEXT:   "tfl.yield"(%[[fq]]) : (tensor<*xf32>) -> ()
// WRAPPED-NEXT:   }) {device = "", narrow_range = true, num_bits = 8 : i64} : (tensor<*xf32>, tensor<186xf32>, tensor<186xf32>) -> tensor<*xf32>
}
