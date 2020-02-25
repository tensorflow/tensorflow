// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %graph:2 = tf_executor.graph {
    %iter:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg0) : (tensor<i32>) -> tensor<i32> loc("iter")
    %val:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32> loc("val")

    // Element wise add `val` with itself for `iter` number of times.
    %2:3 = tf_executor.island wraps "tf.While"(%iter#0, %val#0) {cond = @cond, body = @body, is_stateless = false} : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("StatefulWhile")
    %3:3 = tf_executor.island wraps "tf.While"(%iter#0, %val#0) {cond = @cond, body = @body, is_stateless = true} : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("StatelessWhile")
    tf_executor.fetch %2#1, %3#1 : tensor<f32>, tensor<f32>
  }
  return %graph#0, %graph#1 : tensor<f32>, tensor<f32>
}

func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
    %1:2 = tf_executor.island wraps "tf.Greater"(%arg0, %0#0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %graph : tensor<i1>
}

func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %graph:2 = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
    %1:2 = tf_executor.island wraps "tf.Sub"(%arg0, %0#0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %2:2 = tf_executor.island wraps "tf.Add"(%arg1, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %1#0, %2#0 : tensor<*xi32>, tensor<*xf32>
  }
  return %graph#0, %graph#1 : tensor<*xi32>, tensor<*xf32>
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
