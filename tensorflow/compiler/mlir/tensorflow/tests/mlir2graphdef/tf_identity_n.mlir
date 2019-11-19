// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() -> tensor<2x3xi32> {
  %0 = "tf.Const"() {value = dense<5> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>) loc("Const0")
  %1 = "tf.Const"() {value = dense<4.2> : tensor<4x5xf32>} : () -> (tensor<4x5xf32>) loc("Const1")
  %2:2 = "tf.IdentityN"(%0, %1) : (tensor<2x3xi32>, tensor<4x5xf32>) -> (tensor<2x3xi32>, tensor<4x5xf32>) loc("MyIdentityN")
  return %2#0 : tensor<2x3xi32>
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
