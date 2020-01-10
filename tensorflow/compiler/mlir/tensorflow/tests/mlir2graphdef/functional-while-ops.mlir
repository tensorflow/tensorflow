// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %iter = "tf.Placeholder.input"(%arg0) : (tensor<i32>) -> tensor<i32> loc("iter")
  %val = "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32> loc("val")

  // Element wise add `val` with itself for `iter` number of times.
  %2:2 = "tf.While"(%iter, %val) {
    cond = @cond, body = @body, is_stateless = false
  } : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("StatefulWhile")
  %3:2 = "tf.While"(%iter, %val) {
    cond = @cond, body = @body, is_stateless = true
  } : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("StatelessWhile")

  return %2#1, %3#1 : tensor<f32>, tensor<f32>
}

func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %0 = "tf.Const" () {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
  %1 = "tf.Greater"(%arg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %0 = "tf.Const" () {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
  %1 = "tf.Sub"(%arg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %2 = "tf.Add"(%arg1, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %1, %2 : tensor<*xi32>, tensor<*xf32>
}

// Verify that While op is mapped to TensorFlow StatelessWhile op if the
// is_stateless attribute is present and otherwise it is mapped to TensorFlow
// While op. In both cases, the additional attribute should be dropped.

// CHECK: name: "StatefulWhile"
// CHECK-NOT: name:
// CHECK: op: "While"
// CHECK-NOT: is_stateless

// CHECK: name: "StatelessWhile"
// CHECK-NOT: name:
// CHECK: op: "StatelessWhile"
// CHECK-NOT: is_stateless
