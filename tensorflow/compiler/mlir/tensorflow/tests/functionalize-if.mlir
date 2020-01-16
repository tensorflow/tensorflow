// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=FunctionalizeControlFlowForXlaPass | FileCheck %s --dump-input-on-failure

func @main() {
  %0 = "_tf._TPUReplicate"() {computation = @foo, Tinputs = [], Tbroadcast_inputs = [], NumVariables = 0, Tguaranteed_constants = [], output_types = []} : () -> !_tf.control loc("_TPUReplicate")
  return
}

func @foo() {
  %0:2 = "_tf.Const"() {dtype = "tfdtype$DT_INT32", value = dense<17> : tensor<i32>} : () -> (tensor<i32>, !_tf.control) loc("x")
  %1:2 = "_tf.Const"() {dtype = "tfdtype$DT_BOOL", value = dense<true> : tensor<i1>} : () -> (tensor<i1>, !_tf.control) loc("predicate")
  %2:3 = "_tf.Switch"(%0#0, %1#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i1>) -> (tensor<i32>, tensor<i32>, !_tf.control) loc("switch")
  %3:2 = "_tf.Add"(%2#0, %2#0) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, !_tf.control) loc("Addition")
  %4:2 = "_tf.Mul"(%2#1, %2#1) {T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, !_tf.control) loc("Multiplication")
  %5:3 = "_tf.Merge"(%3#0, %4#0) {N = 2 : i64, T = "tfdtype$DT_INT32"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, !_tf.control) loc("Merge")
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
