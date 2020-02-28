// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s --dump-input-on-failure

func @main() {
  tf_executor.graph {
    // CHECK: name: ".foo"
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>) loc("^foo")
    // CHECK: name: "fo.o"
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> (tensor<i32>) loc("fo{o")
    // CHECK: name: "foo"
    %2:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("foo@1")
    // CHECK: name: "ba.r"
    %3:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("ba r")
    // CHECK: name: "2"
    %4:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("2")
    // CHECK: name: "_3"
    %5:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("_3")
    // CHECK: name: "foo_"
    %6:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("foo_")
    tf_executor.fetch
  }
  return
}
