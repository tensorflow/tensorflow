// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "Constant", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.LegacyCall"(%outputs) {f = @foo0} : (tensor<i32>) -> tensor<i32>
    tf_executor.fetch
  }
  return
}
func @foo0(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xi32>
  }
  return %0 : tensor<*xi32>
}

// CHECK: node {
// CHECK:  name: "_tf.LegacyCall"
// CHECK-NEXT:  op: "foo0"

// CHECK: library {
// CHECK-NEXT:  function {
// CHECK-NEXT:    signature {
// CHECK-NEXT:      name: "foo0"

