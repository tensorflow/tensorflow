// RUN: tf-opt %s -tfl-optimize-functional-ops -split-input-file | FileCheck %s

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
