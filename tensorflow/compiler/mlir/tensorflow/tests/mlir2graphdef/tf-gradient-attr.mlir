// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
  // CHECK:      node {
  // CHECK-NEXT:   name: "Const"
  // CHECK-NEXT:   op: "Const"
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32> loc("Const")

  // CHECK:      node {
  // CHECK-NEXT:   name: "foo"
  // CHECK-NEXT:   op: "foo"
  // CHECK-NEXT:   input: "Const"
    %1:2 = tf_executor.island wraps "tf.foo"(%0#0) {device = ""} : (tensor<f32>) -> tensor<*xf32> loc("foo")
    tf_executor.fetch
  }
  return
}

// CHECK:      library {
// CHECK-NEXT:   function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo"
// CHECK:      function {
// CHECK-NEXT:     signature {
// CHECK-NEXT:       name: "foo_grad"
// CHECK:      gradient {
// CHECK-NEXT:     function_name: "foo"
// CHECK-NEXT:     gradient_func: "foo_grad"
// CHECK-NEXT:   }
// CHECK-NEXT: }
func @foo_grad(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %graph : tensor<*xf32>
}

func @foo(%arg0: tensor<*xf32>) -> tensor<*xf32>
  attributes  {tf.gradient = @foo_grad} {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<*xf32>
  }
  return %graph : tensor<*xf32>
}
