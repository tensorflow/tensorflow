// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s


func @main() {
^bb0:
  // CHECK: node {
  // CHECK-NEXT: name: "_tf.foo"
  // CHECK-NEXT: op: "foo"
  // CHECK: }
  %0 = "_tf.foo"() {name = "_tf.foo"} : () -> (tensor<*xf32>)
  return
}

// CHECK:      library {
// CHECK:        function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "bar"
// CHECK-NEXT:     }
// CHECK:          node_def {
// CHECK-NEXT:       name: "_tf.Const"
// CHECK-NEXT:       op: "Const"
// CHECK-NEXT:       attr {
// CHECK-NEXT:         key: "dtype"
// CHECK-NEXT:         value {
// CHECK-NEXT:           type: DT_INT32
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       attr {
// CHECK-NEXT:         key: "value"
// CHECK-NEXT:         value {
// CHECK-NEXT:           i: 1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK:          }
// CHECK:          node_def {
// CHECK-NEXT:       name: "_tf.Empty"
// CHECK-NEXT:       op: "Empty"
// CHECK-NEXT:       input: "_tf.Const:output:0"
// CHECK-NEXT:       attr {
// CHECK-NEXT:         key: "dtype"
// CHECK-NEXT:         value {
// CHECK-NEXT:           type: DT_FLOAT
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK:          }
// CHECK-NEXT:   }
func @bar() {
^bb0:
  %0 = "_tf.Const"() {dtype = "tfdtype$DT_INT32", name = "_tf.Const", value = 1 : i32} : () -> tensor<i32>
  %1 = "_tf.Empty"(%0) {dtype = "tfdtype$DT_FLOAT", name = "_tf.Empty"} : (tensor<i32>) -> (tensor<*xf32>)
  return
}

// CHECK:        function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo"
// CHECK-NEXT:     }
// CHECK-NEXT:     node_def {
// CHECK-NEXT:       name: "_tf.bar"
// CHECK-NEXT:       op: "bar"
// CHECK:          }
// CHECK-NEXT:   }
// CHECK:      }
func @foo() {
^bb0:
  %0 = "_tf.bar"() {name = "_tf.bar"} : () -> (tensor<*xf32>)
  return
}
