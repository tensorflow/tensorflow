// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s


func.func @main() {
  tf_executor.graph {
    // CHECK: node {
    // CHECK-NEXT: name: "tf.foo"
    // CHECK-NEXT: op: "foo"
    // CHECK: }
    %0:2 = tf_executor.island wraps "tf.foo"() {name = "tf.foo"} : () -> tensor<*xf32>
    tf_executor.fetch
  }
  func.return
}

// CHECK:      library {
// CHECK:        function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "bar"
// CHECK-NEXT:     }
// CHECK:          node_def {
// CHECK-NEXT:       name: "tf.Const"
// CHECK-NEXT:       op: "Const"
// CHECK-NEXT:       attr {
// CHECK:              key: "dtype"
// CHECK-NEXT:         value {
// CHECK-NEXT:           type: DT_INT32
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       attr {
// CHECK-NEXT:         key: "value"
// CHECK-NEXT:         value {
// CHECK-NEXT:           tensor {
// CHECK-NEXT:             dtype: DT_INT32
// CHECK-NEXT:             tensor_shape {
// CHECK-NEXT:             }
// CHECK-NEXT:             int_val: 1
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK:          }
// CHECK:          node_def {
// CHECK-NEXT:       name: "tf.Empty"
// CHECK-NEXT:       op: "Empty"
// CHECK-NEXT:       input: "tf.Const:output:0"
// CHECK-NEXT:         attr {
// CHECK:              key: "dtype"
// CHECK-NEXT:         value {
// CHECK-NEXT:           type: DT_FLOAT
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK:          }
// CHECK-NEXT:   }
func.func @bar() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", name = "tf.Const", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1:2 = tf_executor.island wraps "tf.Empty"(%0#0) {dtype = "tfdtype$DT_FLOAT", name = "tf.Empty"} : (tensor<i32>) -> tensor<*xf32>
    tf_executor.fetch
  }
  func.return
}

// CHECK:        function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo"
// CHECK-NEXT:     }
// CHECK-NEXT:     node_def {
// CHECK-NEXT:       name: "tf.bar"
// CHECK-NEXT:       op: "bar"
// CHECK:          }
// CHECK-NEXT:   }
// CHECK:      }
func.func @foo() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.bar"() {name = "tf.bar"} : () -> tensor<*xf32>
    tf_executor.fetch
  }
  func.return
}
