// RUN: tf-opt %s -tfl-optimize-functional-ops -split-input-file | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: main
func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
  // CHECK: %[[INPUT0:.*]] = "tf.Placeholder.input"
  %0 = "tf.Placeholder.input"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[INPUT1:.*]] = "tf.Placeholder.input"
  %1 = "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32>
  %2 = constant dense<true> : tensor<i1>

  // CHECK: "tf.Add"(%[[INPUT0]], %[[INPUT1]])
  %3 = "tf.If"(%2, %0, %1) {else_branch = @sub, then_branch = @add, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// CHECK-NOT: add
// CHECK-NOT: sub
func @add(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sub(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Verify handling of nested If ops to inline.

// CHECK-LABEL: main
func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
  // CHECK: %[[INPUT0:.*]] = "tf.Placeholder.input"
  %0 = "tf.Placeholder.input"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[INPUT1:.*]] = "tf.Placeholder.input"
  %1 = "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32>
  %2 = constant dense<true> : tensor<i1>

  // CHECK: "tf.Multiply"(%[[INPUT1]], %[[INPUT0]])
  %3 = "tf.If"(%2, %0, %1) {else_branch = @sub, then_branch = @addormul, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// CHECK-NOT: addormul
// CHECK-NOT: sub
// CHECK-NOT: mul
// CHECK-NOT: add

func @addormul(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = constant dense<false> : tensor<i1>
  %1 = "tf.If"(%0, %arg1, %arg0) {else_branch = @mul, then_branch = @add, is_stateless = true} : (tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

func @sub(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @add(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @mul(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Multiply"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Verify that branch functions with multiple references are not erased.

func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> (tensor<f32>, tensor<f32>) {
  %0 = "tf.Placeholder.input"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32>
  %2 = constant dense<true> : tensor<i1>

  // CHECK: tf.Add
  %3 = "tf.If"(%2, %0, %1) {else_branch = @sub, then_branch = @add, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: tf.If
  %4 = "tf.If"(%arg2, %0, %1) {else_branch = @sub, then_branch = @add, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %3, %4 : tensor<f32>, tensor<f32>
}

// CHECK: add
// CHECK: sub
func @add(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sub(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Verify unused if with functions without side-effects are removed.

func @main(%arg0: tensor<3x15x14x3xf32>) -> tensor<3x15x14x8xf32>
    attributes {tf.entry_function = {inputs = "input", outputs = "Conv2D"}} {
  %cst = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %cst_0 = constant dense<1.000000e+00> : tensor<f32>
  %cst_1 = constant dense<0.000000e+00> : tensor<8xf32>
  %cst_2 = constant dense<0.000000e+00> : tensor<8x3x3x3xf32>
  %0 = "tfl.sub"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3x15x14x3xf32>, tensor<f32>) -> tensor<3x15x14x3xf32>
  %1 = "tfl.greater_equal"(%arg0, %0) : (tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<3x15x14x3xi1>
  %2 = "tf.All"(%1, %cst) {Tidx = i32, device = "/device:CPU:0", keep_dims = false} : (tensor<3x15x14x3xi1>, tensor<4xi32>) -> tensor<i1>
  %3 = "tf.If"(%2, %2, %arg0, %0) {Tcond = i1,
    else_branch = @_functionalize_if_else_branch_00, is_stateless = false,
    then_branch = @_functionalize_if_then_branch_00} :
      (tensor<i1>, tensor<i1>, tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<i1>
  %4 = "tfl.conv_2d"(%arg0, %cst_2, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<3x15x14x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<3x15x14x8xf32>
  return %4 : tensor<3x15x14x8xf32>
}

func @_functionalize_if_else_branch_00(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %cst = constant dense<false> : tensor<i1>
  return %cst : tensor<i1>
}

func @_functionalize_if_then_branch_00(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %cst = constant dense<true> : tensor<i1>
  return %cst : tensor<i1>
}

// CHECK: func @main
// CHECK-NOT: tf.If
// CHECK: return
// CHECK-NOT: func @_functionalize_if_else_branch_00
// CHECK-NOT: func @_functionalize_if_then_branch_00

// -----

// Verify unused if with function with side-effects is not removed.

func @main(%arg0: tensor<3x15x14x3xf32>) -> tensor<3x15x14x8xf32>
    attributes {tf.entry_function = {inputs = "input", outputs = "Conv2D"}} {
  %cst = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %cst_0 = constant dense<1.000000e+00> : tensor<f32>
  %cst_1 = constant dense<0.000000e+00> : tensor<8xf32>
  %cst_2 = constant dense<0.000000e+00> : tensor<8x3x3x3xf32>
  %0 = "tfl.sub"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3x15x14x3xf32>, tensor<f32>) -> tensor<3x15x14x3xf32>
  %1 = "tfl.greater_equal"(%arg0, %0) : (tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<3x15x14x3xi1>
  %2 = "tf.All"(%1, %cst) {Tidx = i32, device = "/device:CPU:0", keep_dims = false} : (tensor<3x15x14x3xi1>, tensor<4xi32>) -> tensor<i1>
  %3 = "tf.If"(%2, %2, %arg0, %0) {Tcond = i1,
    else_branch = @_functionalize_if_else_branch_01, is_stateless = false,
    then_branch = @_functionalize_if_then_branch_01} :
      (tensor<i1>, tensor<i1>, tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<i1>
  %4 = "tfl.conv_2d"(%arg0, %cst_2, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<3x15x14x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<3x15x14x8xf32>
  return %4 : tensor<3x15x14x8xf32>
}

func @_functionalize_if_else_branch_01(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %cst = constant dense<false> : tensor<i1>
  return %cst : tensor<i1>
}

func @_functionalize_if_then_branch_01(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %0 = "tf.blah"() : () -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK: func @main
// CHECK: tf.If
// CHECK: return
// CHECK: func @_functionalize_if_else_branch_01
// CHECK: func @_functionalize_if_then_branch_01

// -----

// Verify unused if with function with side-effects is removed if op says
// stateless.

func @main(%arg0: tensor<3x15x14x3xf32>) -> tensor<3x15x14x8xf32>
    attributes {tf.entry_function = {inputs = "input", outputs = "Conv2D"}} {
  %cst = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %cst_0 = constant dense<1.000000e+00> : tensor<f32>
  %cst_1 = constant dense<0.000000e+00> : tensor<8xf32>
  %cst_2 = constant dense<0.000000e+00> : tensor<8x3x3x3xf32>
  %0 = "tfl.sub"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3x15x14x3xf32>, tensor<f32>) -> tensor<3x15x14x3xf32>
  %1 = "tfl.greater_equal"(%arg0, %0) : (tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<3x15x14x3xi1>
  %2 = "tf.All"(%1, %cst) {Tidx = i32, device = "/device:CPU:0", keep_dims = false} : (tensor<3x15x14x3xi1>, tensor<4xi32>) -> tensor<i1>
  %3 = "tf.If"(%2, %2, %arg0, %0) {Tcond = i1,
    else_branch = @_functionalize_if_else_branch_02, is_stateless = true,
    then_branch = @_functionalize_if_then_branch_02} :
      (tensor<i1>, tensor<i1>, tensor<3x15x14x3xf32>, tensor<3x15x14x3xf32>) -> tensor<i1>
  %4 = "tfl.conv_2d"(%arg0, %cst_2, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<3x15x14x3xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<3x15x14x8xf32>
  return %4 : tensor<3x15x14x8xf32>
}

func @_functionalize_if_else_branch_02(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %cst = constant dense<false> : tensor<i1>
  return %cst : tensor<i1>
}

func @_functionalize_if_then_branch_02(%arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
  %0 = "tf.blah"() : () -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK: func @main
// CHECK-NOT: tf.If
// CHECK: return
// CHECK-NOT: func @_functionalize_if_else_branch_02
// CHECK-NOT: func @_functionalize_if_then_branch_02
