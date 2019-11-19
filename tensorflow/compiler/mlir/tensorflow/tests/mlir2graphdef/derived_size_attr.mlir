// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s --dump-input-on-failure

// CHECK: op: "Split"
// CHECK: attr {
// CHECK:   key: "num_split"
// CHECK:   value {
// CHECK:     i: 2
// CHECK:   }
// CHECK: }

func @main() {
  %dim = "tf.Const"() {dtype = "tftype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  %input = "tf.Const"() {dtype = "tftype$DT_INT32", value = dense<1.0> : tensor<4x6xf32>} : () -> (tensor<4x6xf32>)
  %0:2 = "tf.Split"(%dim, %input) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
  return
}
