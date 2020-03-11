// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=FunctionalizeControlFlowForXlaPass | FileCheck %s --dump-input-on-failure

func @main() {
  tf_executor.graph {
    %0 = tf_executor.island wraps "tf._TPUReplicate"() {computation = @foo, Tinputs = [], Tbroadcast_inputs = [], NumVariables = 0, Tguaranteed_constants = [], output_types = []} : () -> () loc("_TPUReplicate")
    tf_executor.fetch
  }
  return
}

func @foo() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<17> : tensor<i32>} : () -> tensor<i32> loc("x")
    %1:2 = tf_executor.island wraps "tf.Const"() {dtype = "tfdtype$DT_BOOL", value = dense<true> : tensor<i1>} : () -> tensor<i1> loc("predicate")
    %2:3 = tf_executor.Switch %0#0, %1#0 : (tensor<i32>, tensor<i1>) -> (tensor<i32>, tensor<i32>, !tf_executor.control) {device = "", T = "tfdtype$DT_INT32"} loc("switch")
    %3:2 = tf_executor.island wraps "tf.Add"(%2#0, %2#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("Addition")
    %4:2 = tf_executor.island wraps "tf.Mul"(%2#1, %2#1) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("Multiplication")
    %5:3 = tf_executor.Merge %3#0, %4#0 : tensor<i32> {device = "", N = 2, T = "tfdtype$DT_INT32"} loc("Merge")
    tf_executor.fetch
  }
  return
}

// Match the name of the cloned function with functionalized control-flow at call site
// CHECK: func @main()
// CHECK: computation = @[[FUNCTIONALIZE_FUNC:[A-Za-z0-9_]*]]


// In the newly cloned function, check that we have a _tf.If operation and capture the then and else branch.
// CHECK: func @[[FUNCTIONALIZE_FUNC]]
// CHECK: "tf.If"
// CHECK-SAME:  else_branch = @[[ELSE_FUNC:[A-Za-z0-9_]*]]
// CHECK-SAME:  then_branch = @[[THEN_FUNC:[A-Za-z0-9_]*]]

// We expect the _tf.Add in the else func and the _tf.Mul in the then func

// CHECK: func @[[ELSE_FUNC]]
// CHECK: "tf.Add"
// CHECK: func @[[THEN_FUNC]]
// CHECK: "tf.Mul"
