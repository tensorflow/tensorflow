// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
^bb0:
  // CHECK:      name: "while/Merge"
  // CHECK-NEXT: op: "Merge"
  // CHECK-NEXT: input: "while/Enter"
  // CHECK-NEXT: input: "while/NextIteration"
  // CHECK:      name: "while/NextIteration"
  // CHECK-NEXT: op: "NextIteration"
  // CHECK-NEXT: input: "while/Add"
  %0:2 = "_tf.NextIteration.source"() {device = "", T = "tfdtype$DT_INT32"} : () -> (tensor<*xi32>, !_tf.control) loc("while/NextIteration")
  %1:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<i32>, !_tf.control) loc("Const")
  %2:2 = "_tf.Enter"(%1#0) {device = "", T = "tfdtype$DT_INT32", frame_name = "while/while_context", is_constant = false, parallel_iterations = 10} : (tensor<i32>) -> (tensor<*xi32>, !_tf.control) loc("while/Enter")
  %3:3 = "_tf.Merge"(%2#0, %0#0) {device = "", N = 2, T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<i32>, !_tf.control) loc("while/Merge")
  %4:2 = "_tf.Const"(%3#2) {device = "", dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control) loc("while/Less/y")
  %5:2 = "_tf.Less"(%3#0, %4#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>, !_tf.control) loc("while/Less")
  %6:2 = "_tf.LoopCond"(%5#0) {device = ""} : (tensor<*xi1>) -> (tensor<i1>, !_tf.control) loc("while/LoopCond")
  %7:3 = "_tf.Switch"(%3#0, %6#0) {device = "", T = "tfdtype$DT_INT32", _class = ["loc:@while/Merge"]} : (tensor<*xi32>, tensor<i1>) -> (tensor<*xi32>, tensor<*xi32>, !_tf.control) loc("while/Switch")
  %8:2 = "_tf.Exit"(%7#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control) loc("while/Exit")
  %9:2 = "_tf.Identity"(%7#1) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> (tensor<*xi32>, !_tf.control) loc("while/Identity")
  %10:2 = "_tf.Const"(%9#1) {device = "", dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control) loc("while/Add/y")
  %11:2 = "_tf.Add"(%9#0, %10#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, !_tf.control) loc("while/Add")
  %12 = "_tf.NextIteration.sink"(%11#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*xi32>) -> !_tf.control loc("while/NextIteration")
  return
}
