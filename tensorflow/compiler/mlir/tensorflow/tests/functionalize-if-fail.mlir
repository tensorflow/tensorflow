// RUN: not tf-opt %s --run-tf-graph-optimization --graph-passes=FunctionalizeControlFlowForXlaPass 2>&1 | FileCheck %s

// CHECK:       error: FunctionalizeControlFlowForXlaPass: Graph contains node with inputs predicated on incompatible predicates: {s(Cond:0,then)} and {s(Cond:0,else)}
// CHECK-NEXT:  for node {{[{][{]node Add[}][}]}}

func @main() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf._TPUReplicate"() {computation = @foo, Tinputs = [], Tbroadcast_inputs = [], NumVariables = 0, Tguaranteed_constants = [], output_types = []} : () -> () loc("_TPUReplicate")
    tf_executor.fetch
  }
  return
}

func @foo() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<17> : tensor<i32>} : () -> tensor<i32> loc("x")
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_BOOL", value = dense<true> : tensor<i1>} : () -> tensor<i1> loc("Cond")
    %2:3 = tf_executor.Switch %0#0, %1#0 : (tensor<i32>, tensor<i1>) -> (tensor<i32>, tensor<i32>, !tf_executor.control) {device = "", T = "tfdtype$DT_INT32"} loc("switch")
    %3:2 = tf_executor.island wraps "tf.Add"(%2#0, %2#1) {T = "tfdtype$DT_INT32", device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("Add")
    %4:2 = tf_executor.island wraps "tf.Mul"(%2#1, %2#0) {T = "tfdtype$DT_INT32", device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("Square")
    %5:3 = tf_executor.Merge %3#0, %4#0 : tensor<i32> {device = "", N = 2, T = "tfdtype$DT_INT32"} loc("Merge")
    tf_executor.fetch
  }
  return
}
