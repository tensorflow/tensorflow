// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  tf_executor.graph {
    // CHECK: name: "foo"
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>) loc("foo")
    // CHECK: name: "foo{{_.*_1}}"
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> (tensor<i32>) loc("foo")
    // CHECK: name: "foo1"
    %2:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("foo1")
    // CHECK: name: "foo{{_.*_2}}"
    %3:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("foo")
    // CHECK: name: "2"
    %4:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("2")
    // CHECK: name: "3"
    %5:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("3")
    tf_executor.fetch
  }
  func.return
}
