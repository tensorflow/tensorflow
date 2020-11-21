// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// CHECK: op: "Split"
// CHECK: attr {
// CHECK:   key: "num_split"
// CHECK:   value {
// CHECK:     i: 2
// CHECK:   }
// CHECK: }

func @main() {
  tf_executor.graph {
    %dim:2 = tf_executor.island wraps "tf.Const"() {dtype = "tftype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %input:2 = tf_executor.island wraps "tf.Const"() {dtype = "tftype$DT_INT32", value = dense<1.0> : tensor<4x6xf32>} : () -> tensor<4x6xf32>
    %split:3 = tf_executor.island wraps "tf.Split"(%dim#0, %input#0) : (tensor<i32>, tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
    tf_executor.fetch
  }
  return
}
