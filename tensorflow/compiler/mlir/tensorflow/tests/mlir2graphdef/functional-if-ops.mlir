// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = tf_executor.graph {
    %outputs_2, %control_3 = tf_executor.island wraps "tf.Less"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %outputs_4, %control_5 = tf_executor.island wraps "tf.If"(%outputs_2, %arg2, %arg3) {else_branch = @cond_false, is_stateless = false, then_branch = @cond_true} : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("StatefulIf")
    %outputs_6, %control_7 = tf_executor.island wraps "tf.If"(%outputs_2, %arg2, %arg3) {else_branch = @cond_false, is_stateless = true, then_branch = @cond_true} : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32> loc("StatelessIf")
    tf_executor.fetch %outputs_4, %outputs_6 : tensor<4xf32>, tensor<4xf32>
  }
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Add"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Mul"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}

// Verify that If op is mapped to TensorFlow StatelessIf op if the is_stateless
// attribute is present and otherwise it is mapped to TensorFlow If op. In both
// cases, the additional attribute should be dropped.

// CHECK: name: "StatefulIf"
// CHECK-NOT: name:
// CHECK: op: "If"
// CHECK-NOT: is_stateless
// CHECK:  attr {
// CHECK:    key: "output_shapes"
// CHECK:    value {
// CHECK:      list {
// CHECK:        shape {
// CHECK:          dim {
// CHECK:            size: 4
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:  }

// CHECK: name: "StatelessIf"
// CHECK-NOT: name:
// CHECK: op: "StatelessIf"
// CHECK-NOT: is_stateless
// CHECK:  attr {
// CHECK:    key: "output_shapes"
// CHECK:    value {
// CHECK:      list {
// CHECK:        shape {
// CHECK:          dim {
// CHECK:            size: 4
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:  }
