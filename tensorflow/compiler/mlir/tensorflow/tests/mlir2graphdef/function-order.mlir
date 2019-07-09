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
// CHECK-NEXT:     node_def {
// CHECK-NEXT:       name: "_tf.Empty"
// CHECK-NEXT:       op: "Empty"
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
  %0 = "_tf.Empty"() {dtype = "tfdtype$DT_FLOAT", name = "_tf.Empty"} : () -> (tensor<*xf32>)
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
