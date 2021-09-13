// RUN: tf-tfrt-opt -pass-pipeline='tf-to-tfrt{func-use-fallback-tensor=true target-tpurt=true}'  %s | FileCheck %s --dump-input=fail

func @callee(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  return %arg0: tensor<i32>
}
// CHECK-LABEL: func @serving_default
// CHECK-SAME: ([[chain:%.*]]: !tfrt.chain,
func @serving_default(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[chain:%.*]], [[results:%.*]] = tfrt_fallback_async.batch_function
  // CHECK-NEXT: [[fb_tensor:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[results]]
  // CHECK-SAME: (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  // CHECK-NEXT: tpurt.compile_and_execute([[fb_tensor]])
  %0 = "tf.BatchFunction"(%arg0, %arg0) {allowed_batch_sizes = [64], batch_timeout_micros = 1 : i64, batching_queue = "", container = "", f = @callee, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operand_segment_sizes = dense<1> : vector<2xi32>, shared_name = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module"} : (tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}
