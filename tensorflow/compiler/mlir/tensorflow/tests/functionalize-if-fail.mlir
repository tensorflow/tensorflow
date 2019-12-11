// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=FunctionalizeControlFlowForXlaPass 2>&1 | FileCheck %s; if [[ ${PIPESTATUS[0]} != 0  &&  ${PIPESTATUS[1]} == 0 ]]; then exit 0; else exit 1; fi

// CHECK:       error: FunctionalizeControlFlowForXlaPass: Graph contains node with inputs predicated on incompatible predicates: {s(Cond:0,then)} and {s(Cond:0,else)}
// CHECK-NEXT:  for node {{[{][{]node Add[}][}]}}

func @main() {
  %0 = "_tf._TPUReplicate"() {computation = @foo, Tinputs = [], Tbroadcast_inputs = [], NumVariables = 0, Tguaranteed_constants = [], output_types = []} : () -> !_tf.control loc("_TPUReplicate")
  return
}

func @foo() {
  %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<17> : tensor<i32>} : () -> (tensor<i32>, !_tf.control) loc("x")
  %1:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_BOOL", value = dense<true> : tensor<i1>} : () -> (tensor<i1>, !_tf.control) loc("Cond")
  %2:3 = "_tf.Switch"(%0#0, %1#0) {T = "tfdtype$DT_INT32", device = ""} : (tensor<i32>, tensor<i1>) -> (tensor<i32>, tensor<i32>, !_tf.control) loc("switch")
  %3:2 = "_tf.Add"(%2#0, %2#1) {T = "tfdtype$DT_INT32", device = ""} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, !_tf.control) loc("Add")
  %4:2 = "_tf.Mul"(%2#1, %2#0) {T = "tfdtype$DT_INT32", device = ""} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, !_tf.control) loc("Square")
  %5:3 = "_tf.Merge"(%3#0, %4#0) {N = 2 : i64, T = "tfdtype$DT_INT32", device = "", name = "_tf.Merge"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, !_tf.control) loc("Merge")
  return
}
