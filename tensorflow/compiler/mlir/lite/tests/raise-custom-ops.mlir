// RUN: tf-opt -tfl-raise-custom-ops -canonicalize %s -o - | FileCheck %s

// CHECK-LABEL: custom_op
func @custom_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tfl.pseudo_const" () {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // will be preserved since it has uses.
  %2 = "tf.MyCustomOp"(%1, %0) {fused_activation_function = "RELU", int_attr = 2 : i32}  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // will be removed since it doesn't have uses and doesn't have side effect.
  "tf.MyCustomOp"(%1, %0) {fused_activation_function = "RELU", int_attr = 2 : i32}  : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>

// CHECK-NEXT: %[[CST:.*]] = constant dense<1.000000e+00>
// CHECK-NEXT: %[[MUL:.*]] = tfl.mul %arg0, %[[CST]] {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK-NEXT: %[[CUSTOM:.*]] = "tfl.custom_tf"(%[[MUL]], %[[CST]]) ( {
// CHECK-NEXT: ^bb0(%arg1: tensor<4xf32>, %arg2: tensor<4xf32>): // no predecessors
// CHECK-NEXT:   %[[MY_CUSTOM:.*]] = "tf.MyCustomOp"(%arg1, %arg2) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:   "tfl.yield"(%[[MY_CUSTOM]]) : (tensor<4xf32>) -> ()
// CHECK-NEXT: }) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[CUSTOM]] : tensor<4xf32>
}
