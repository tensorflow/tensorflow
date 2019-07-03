// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
^bb0:
  // CHECK: name: "foo"
  %0 = "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>) loc("foo")
  // CHECK: name: "foo1"
  %1 = "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> (tensor<i32>) loc("foo")
  // CHECK: name: "foo2"
  %2 = "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<i32>) loc("foo1")
  // CHECK: name: "Unnamed"
  %3 = "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("2")
  // CHECK: name: "Unnamed1"
  %4 = "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<3> : tensor<i32>} : () -> (tensor<i32>) loc("3")
  return
}
