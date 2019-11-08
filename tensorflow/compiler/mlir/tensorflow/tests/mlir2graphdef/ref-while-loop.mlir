// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Verify the ops generated when Ref type is used in a while loop.
func @main() {
  // CHECK:  op: "RefEnter"
  // CHECK:  op: "RefMerge"
  // CHECK:  op: "RefSwitch"
  // CHECK:  op: "RefExit"
  // CHECK:  op: "RefNextIteration"
  %0:2 = "_tf.NextIteration.source"() {device = "", T = "tfdtype$DT_INT32"} : () -> (tensor<*x!tf.int32ref>, !_tf.control) loc("while/NextIteration")
  %1:2 = "_tf.VariableV2"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>} : () -> (tensor<!tf.int32ref>, !_tf.control) loc("Ref_Variable")
  %2:2 = "_tf.Enter"(%1#0) {device = "", T = "tfdtype$DT_INT32", frame_name = "while/while_context", is_constant = false, parallel_iterations = 10} : (tensor<!tf.int32ref>) -> (tensor<*x!tf.int32ref>, !_tf.control) loc("while/Enter")
  %3:3 = "_tf.Merge"(%2#0, %0#0) {device = "", N = 2, T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>, tensor<*x!tf.int32ref>) -> (tensor<*x!tf.int32ref>, tensor<i32>, !_tf.control) loc("while/Merge")
  %4:2 = "_tf.Const"(%3#2) {device = "", dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control) loc("while/Less/y")
  %5:2 = "_tf.Less"(%3#0, %4#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>, tensor<i32>) -> (tensor<*xi1>, !_tf.control) loc("while/Less")
  %6:2 = "_tf.LoopCond"(%5#0) {device = ""} : (tensor<*xi1>) -> (tensor<i1>, !_tf.control) loc("while/LoopCond")
  %7:3 = "_tf.Switch"(%3#0, %6#0) {device = "", T = "tfdtype$DT_INT32", _class = ["loc:@while/Merge"]} : (tensor<*x!tf.int32ref>, tensor<i1>) -> (tensor<*x!tf.int32ref>, tensor<*x!tf.int32ref>, !_tf.control) loc("while/Switch")
  %8:2 = "_tf.Exit"(%7#1) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>) -> (tensor<*x!tf.int32ref>, !_tf.control) loc("while/Exit")
  %10:2 = "_tf.Const"(%7#2) {device = "", dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : (!_tf.control) -> (tensor<i32>, !_tf.control) loc("while/Add/y")
  %11:2 = "_tf.AssignAdd"(%7#0, %10#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>, tensor<i32>) -> (tensor<*x!tf.int32ref>, !_tf.control) loc("while/Add")
  %12 = "_tf.NextIteration.sink"(%11#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>) -> !_tf.control loc("while/NextIteration")
  return
}
