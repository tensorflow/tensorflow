// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline="enable-optimizer=true tfrt-cost-threshold=1024" %s | FileCheck %s --dump-input=fail

// Check that unused While op results and the associated ops are removed.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 462 : i32}} {
  func.func @while_cond_lt9(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    func.return %1 : tensor<i1>
  }

  func.func @while_body_add2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.AddV2"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Div"(%arg1, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    func.return %1, %2 : tensor<i32>, tensor<i32>
  }

  // CHECK-LABEL: func @while_test_remove_unused_results
  // CHECK:       [[pred:%.*]] = tfrt_fallback_async.predicate
  // CHECK-NEXT:  tfrt.while [[pred]] @"[[while_func_prefix:.*]]/tfrt_body_1"
  // CHECK-SAME:  (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
  // CHECK-NOT:   func.call
  func.func @while_test_remove_unused_results(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0:2 = "tf.While"(%arg0, %arg1) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %1:2 = func.call @while_body_add2(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    func.return %0#0, %1#0 : tensor<i32>, tensor<i32>
  }

  // CHECK:     func @"[[while_func_prefix]]/tfrt_body_1"
  // CHECK:     "tf.AddV2"
  // CHECK-NOT: "tf.Div"
}
