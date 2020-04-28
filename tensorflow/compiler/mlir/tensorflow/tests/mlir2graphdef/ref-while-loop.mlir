// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Verify the ops generated when Ref type is used in a while loop.
func @main() {
  // CHECK:  op: "RefEnter"
  // CHECK:  op: "RefMerge"
  // CHECK:  op: "RefSwitch"
  // CHECK:  op: "RefExit"
  // CHECK:  op: "RefNextIteration"
  tf_executor.graph {
    %0:3 = tf_executor.NextIteration.Source : tensor<*x!tf.int32ref> {device = "", T = "tfdtype$DT_INT32"} loc("while/NextIteration")
    %1:2 = tf_executor.island wraps "tf.VariableV2"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<i32>, shape = #tf.shape<0>, container = "", shared_name = ""} : () -> tensor<!tf.int32ref> loc("Ref_Variable")
    %2:2 = tf_executor.Enter %1#0 frame "while/while_context" parallel_iterations 10 : (tensor<!tf.int32ref>) -> (tensor<*x!tf.int32ref>, !tf_executor.control) {device = "", T = "tfdtype$DT_INT32"} loc("while/Enter")
    %3:3 = tf_executor.Merge %2#0, %0#0 : tensor<*x!tf.int32ref> {device = "", N = 2, T = "tfdtype$DT_INT32"} loc("while/Merge")
    %4:2 = tf_executor.island(%3#2) wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<10> : tensor<i32>} : () -> tensor<i32> loc("while/Less/y")
    %5:2 = tf_executor.island(%3#2) wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_BOOL", value = dense<true> : tensor<i1>} : () -> tensor<i1> loc("while/Less")
    %6:2 = tf_executor.LoopCond %5#0 : (tensor<i1>) -> (tensor<i1>, !tf_executor.control) {device = ""} loc("while/LoopCond")
    %7:3 = tf_executor.Switch %3#0, %6#0 : (tensor<*x!tf.int32ref>, tensor<i1>) -> (tensor<*x!tf.int32ref>, tensor<*x!tf.int32ref>, !tf_executor.control) {device = "", T = "tfdtype$DT_INT32", _class = ["loc:@while/Merge"]} loc("while/Switch")
    %8:2 = tf_executor.Exit %7#1 : tensor<*x!tf.int32ref> {device = "", T = "tfdtype$DT_INT32"} loc("while/Exit")
    %10:2 = tf_executor.island(%7#2) wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("while/Add/y")
    %11:2 = tf_executor.island wraps "tf.AssignAdd"(%7#0, %10#0) {device = "", T = "tfdtype$DT_INT32"} : (tensor<*x!tf.int32ref>, tensor<i32>) -> tensor<*x!tf.int32ref> loc("while/Add")
    tf_executor.NextIteration.Sink [%0#1] %11#0 : tensor<*x!tf.int32ref> {device = "", T = "tfdtype$DT_INT32"} loc("while/NextIteration")
    tf_executor.fetch
  }
  return
}
