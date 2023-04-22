// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<i32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>, tensor<5xf32>) {
  %0:3 = tf_executor.graph {
    %outputs_2:2, %control_3 = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @body, cond = @cond, is_stateless = false} : (tensor<i32>, tensor<5xf32>) -> (tensor<i32>, tensor<5xf32>) loc("StatefulWhile")
    %outputs_4:2, %control_5 = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @body, cond = @cond, is_stateless = true} : (tensor<i32>, tensor<5xf32>) -> (tensor<i32>, tensor<5xf32>) loc("StatelessWhile")
    %outputs_6:2, %control_7 = tf_executor.island wraps "tf.While"(%arg0, %arg1) {body = @body, cond = @cond, is_stateless = false, shape_invariant} : (tensor<i32>, tensor<5xf32>) -> (tensor<i32>, tensor<5xf32>) loc("WhileWithOutputShapes")
    tf_executor.fetch %outputs_2#1, %outputs_4#1, %outputs_6#1 : tensor<5xf32>, tensor<5xf32>, tensor<5xf32>
  }
  return %0#0, %0#1, %0#2 : tensor<5xf32>, tensor<5xf32>, tensor<5xf32>
}

func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Greater"(%arg0, %outputs) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
    tf_executor.fetch %outputs_0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %0:2 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Sub"(%arg0, %outputs) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
    %outputs_2, %control_3 = tf_executor.island wraps "tf.Add"(%arg1, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    tf_executor.fetch %outputs_0, %outputs_2 : tensor<*xi32>, tensor<*xf32>
  }
  return %0#0, %0#1 : tensor<*xi32>, tensor<*xf32>
}

// Verify that While op is mapped to TensorFlow StatelessWhile op if the
// is_stateless attribute is present and otherwise it is mapped to TensorFlow
// While op. In both cases, the additional attribute should be dropped.

// CHECK: name: "StatefulWhile"
// CHECK-NOT: name:
// CHECK: op: "While"
// CHECK-NOT: is_stateless
// CHECK-NOT: shape_invariant
// CHECK:  attr {
// CHECK:    key: "output_shapes"
// CHECK:    value {
// CHECK:      list {
// CHECK:        shape {
// CHECK:          dim {
// CHECK:            size: 5
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:  }


// CHECK: name: "StatelessWhile"
// CHECK-NOT: name:
// CHECK: op: "StatelessWhile"
// CHECK-NOT: is_stateless
// CHECK-NOT: shape_invariant
// CHECK:  attr {
// CHECK:    key: "output_shapes"
// CHECK:    value {
// CHECK:      list {
// CHECK:        shape {
// CHECK:          dim {
// CHECK:            size: 5
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:  }

// CHECK: name: "WhileWithOutputShapes"
// CHECK-NOT: name:
// CHECK: op: "While"
// CHECK-NOT: is_stateless
// CHECK-NOT: shape_invariant
// CHECK:  attr {
// CHECK:    key: "output_shapes"
// CHECK:    value {
// CHECK:      list {
// CHECK:        shape {
// CHECK:          dim {
// CHECK:            size: 5
// CHECK:          }
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:  }
