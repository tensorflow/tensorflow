// RUN: tf-opt -tfl-lower-custom-ops -canonicalize %s -o - | FileCheck %s

// CHECK-LABEL: custom_op
func @custom_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = constant dense<1.000000e+00> : tensor<4xf32>
  %0 = tfl.mul %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  %1 = "tfl.custom_tf"(%0, %cst) ( {
  ^bb0(%arg1: tensor<4xf32>, %arg2: tensor<4xf32>):  // no predecessors
    %2 = "tf.MyCustomOp"(%arg1, %arg2) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "tfl.yield"(%2) : (tensor<4xf32>) -> ()
  }) {fused_activation_function = "RELU", int_attr = 2 : i32} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>

// CHECK-NEXT: %[[CST:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT: %[[MUL:.*]] = tfl.mul %arg0, %[[CST]] {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK-NEXT: %[[MY_CUSTOM:.*]] = "tf.MyCustomOp"(%[[MUL]], %[[CST]])
// CHECK-NEXT: return %[[MY_CUSTOM]] : tensor<4xf32>
}
