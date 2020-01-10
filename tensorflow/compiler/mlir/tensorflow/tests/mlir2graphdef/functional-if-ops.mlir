// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0 = "tf.Placeholder.input"(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32>
  %2 = "tf.Less"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %3 = "tf.If"(%2, %0, %1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32> loc("StatefulIf")
  %4 = "tf.If"(%2, %0, %1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32> loc("StatelessIf")
  return %3, %4 : tensor<f32>, tensor<f32>
}

func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Mul"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// Verify that If op is mapped to TensorFlow StatelessIf op if the is_stateless
// attribute is present and otherwise it is mapped to TensorFlow If op. In both
// cases, the additional attribute should be dropped.

// CHECK: name: "StatefulIf"
// CHECK-NOT: name:
// CHECK: op: "If"
// CHECK-NOT: is_stateless

// CHECK: name: "StatelessIf"
// CHECK-NOT: name:
// CHECK: op: "StatelessIf"
// CHECK-NOT: is_stateless
