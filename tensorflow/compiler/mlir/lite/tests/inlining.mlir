// RUN: tf-opt %s -inline="disable-simplify" | FileCheck %s

// Inline a function that contains only tfl ops.
func @func_with_tfl_ops(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
  %0 = "tfl.sub"(%arg0, %arg0) {fused_activation_function = "RELU6"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tfl.add"(%0, %arg0) {fused_activation_function = "RELU6"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %1: tensor<2xi32>
}

// CHECK-LABEL: func @inline_with_arg
// CHECK-SAME:    [[VAL_0:%.*]]: tensor<2xi32>
func @inline_with_arg(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
// CHECK-NEXT:  [[VAL_1:%.*]] = tfl.sub [[VAL_0]], [[VAL_0]] {fused_activation_function = "RELU6"} : tensor<2xi32>
// CHECK-NEXT:  [[VAL_2:%.*]] = tfl.add [[VAL_1]], [[VAL_0]] {fused_activation_function = "RELU6"} : tensor<2xi32>
// CHECK-NEXT:  return [[VAL_2]] : tensor<2xi32>
  %0 = call @func_with_tfl_ops(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
