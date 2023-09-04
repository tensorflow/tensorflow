// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  // CHECK: node {
  // CHECK: "_XlaSharding"
  // CHECK-NEXT: value {
  // CHECK-NEXT: s: "\010\001\032\001\001\"\001\000"
  tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf._Arg"() {T = i32, _XlaSharding = "{maximal device=0}", device = "", index = 0 : i64} : () -> tensor<*xi32>
    // CHECK: node {
    // CHECK: "_XlaSharding"
    // CHECK-NEXT: value {
    // CHECK-NEXT: s: "\010\004"
    %outputs2, %control2 = tf_executor.island wraps "tf._Arg"() {T = i32, _XlaSharding = "{manual}", device = "", index = 0 : i64} : () -> tensor<*xi32>
    // CHECK: node {
    // CHECK: "_XlaSharding"
    // CHECK-NEXT: value {
    // CHECK-NEXT: s: "\010\003\032\002\001\002\"\002\000\001"
    %outputs3, %control3 = tf_executor.island wraps "tf._Arg"() {T = i32, _XlaSharding = "{devices=[1,2]0,1}", device = "", index = 0 : i64} : () -> tensor<*xi32>
    // CHECK: node {
    // CHECK: "_XlaSharding"
    // CHECK-NEXT: value {
    // CHECK-NEXT: s: ""
    %outputs4, %control4 = tf_executor.island wraps "tf._Arg"() {T = i32, _XlaSharding = "{replicated}", device = "", index = 0 : i64} : () -> tensor<*xi32>
    tf_executor.fetch
  }
  return
}