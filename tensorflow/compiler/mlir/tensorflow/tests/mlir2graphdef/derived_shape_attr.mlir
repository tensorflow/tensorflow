// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Check that attributes that define derived shapes are exported.

// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape
// CHECK: unknown_rank: true
// CHECK: name: "static"
// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape {
// CHECK-NEXT: }
// CHECK: name: "static_10"
// CHECK: op: "PlaceholderWithDefault"
// CHECK: shape
// CHECK: dim
// CHECK: size: 10

func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%1#0) {type = i32} : (tensor<i32>) -> tensor<*xi32> loc("unranked")
    %3:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%1#0) {type = i32} : (tensor<i32>) -> tensor<i32> loc("static")
    %4:2 = tf_executor.island wraps "tf.PlaceholderWithDefault"(%0#0) {type = i32} : (tensor<10xi32>) -> tensor<10xi32> loc("static_10")
    tf_executor.fetch
  }
  return
}
