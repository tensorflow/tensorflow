// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %graph:2 = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg0) : (tensor<f32>) -> tensor<f32>
    %1:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg1) : (tensor<f32>) -> tensor<f32>
    %2:2 = tf_executor.island wraps "tf.Less"(%0#0, %1#0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3:2 = tf_executor.island wraps "tf.If"(%2#0, %0#0, %1#0) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32> loc("StatefulIf")
    %4:2 = tf_executor.island wraps "tf.If"(%2#0, %0#0, %1#0) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32> loc("StatelessIf")
    tf_executor.fetch %3#0, %4#0 : tensor<f32>, tensor<f32>
  }
  return %graph#0, %graph#1 : tensor<f32>, tensor<f32>
}

func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %0#0 : tensor<*xf32>
  }
  return %graph : tensor<*xf32>
}

func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Mul"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %0#0 : tensor<*xf32>
  }
  return %graph : tensor<*xf32>
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
