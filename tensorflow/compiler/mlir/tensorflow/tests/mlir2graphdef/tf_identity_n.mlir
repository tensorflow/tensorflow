// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() -> tensor<2x3xi32> {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {value = dense<5> : tensor<2x3xi32>} : () -> tensor<2x3xi32> loc("Const0")
    %1:2 = tf_executor.island wraps "tf.Const"() {value = dense<4.2> : tensor<4x5xf32>} : () -> tensor<4x5xf32> loc("Const1")
    %2:3 = tf_executor.island wraps "tf.IdentityN"(%0, %1) : (tensor<2x3xi32>, tensor<4x5xf32>) -> (tensor<2x3xi32>, tensor<4x5xf32>) loc("MyIdentityN")
    tf_executor.fetch %2#0 : tensor<2x3xi32>
  }
  return %graph : tensor<2x3xi32>
}

// CHECK:        name: "MyIdentityN"
// CHECK-NEXT:   op: "IdentityN"
// CHECK-NEXT:   input: "Const0"
// CHECK-NEXT:   input: "Const1"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         type: DT_INT32
// CHECK-NEXT:         type: DT_FLOAT
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
