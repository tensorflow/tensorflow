// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Confirm function references in if ops are preserved
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK:   %{{.*}} = "tf.If"(%{{.*}}, %{{.*}}, %{{.*}}) <{else_branch = @cond_false, is_stateless = false, then_branch = @cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %1 = "tf.If"(%0, %arg0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @tfl_if(%arg0: tensor<i1>) -> tensor<i32> {
// CHECK:   %{{.*}} = "tf.If"(%{{.*}}, %{{.*}}) <{else_branch = @tfl.if_else, is_stateless = false, then_branch = @tfl.if_then}> : (tensor<i1>, tensor<i32>) -> tensor<i32>
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = tfl.add %cst, %cst {fused_activation_function = "NONE"} : tensor<i32>
  %1 = "tfl.if"(%arg0) ({
    %2 = func.call @tfl.if_then(%0) : (tensor<i32>) -> tensor<i32>
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  }, {
    %2 = func.call @tfl.if_else(%0) : (tensor<i32>) -> tensor<i32>
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i1>) -> tensor<i32>
  return %1 : tensor<i32>
}
func.func private @tfl.if_then(%arg0: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}
func.func private @tfl.if_else(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @tfl_if_multi_args(%arg0: tensor<i1>) -> tensor<i32> {
// CHECK:   %{{.*}} = "tf.If"(%{{.*}}, %{{.*}}, %{{.*}}) <{else_branch = @tfl.if_else_1, is_stateless = false, then_branch = @tfl.if_then_1}> : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = tfl.add %cst, %cst {fused_activation_function = "NONE"} : tensor<i32>
  %1 = tfl.mul %cst, %cst {fused_activation_function = "NONE"} : tensor<i32>
  %2 = "tfl.if"(%arg0) ({
    %2 = func.call @tfl.if_then_1(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  }, {
    %2 = func.call @tfl.if_else_1(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  }) : (tensor<i1>) -> tensor<i32>
  return %1 : tensor<i32>
}
func.func private @tfl.if_then_1(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  return %arg0 : tensor<i32>
}
func.func private @tfl.if_else_1(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<i32>
  return %0 : tensor<i32>
}