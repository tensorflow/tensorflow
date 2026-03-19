// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @__inference_pruned_131
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
// CHECK-NEXT: [[o_chain:%.*]], [[o:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(0) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.VarHandleOp"()
// CHECK-NEXT: [[o_chain_0:%.*]], [[o1:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(1) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"([[o]]) {dtype = f32} : 1
// CHECK-NEXT: [[out_ch:%.*]] = tfrt.merge.chains [[o_chain]], [[o_chain_0]]
// CHECK-NEXT: tfrt.return [[out_ch]], [[o1]] : !tfrt.chain, !tfrt_fallback.tf_tensor
module attributes {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0"], tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 679 : i32}}  {
  func.func @__inference_pruned_131() -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "variable", outputs = "identity_retval_RetVal"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.VariableV2"() {container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", shape = #tf_type.shape<>, shared_name = "v_load_44"} : () -> tensor<!tf_type.f32ref>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Identity"(%outputs) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.f32ref>) -> tensor<*xf32>
      tf_executor.fetch %outputs_0 : tensor<*xf32>
    }
    func.return %0 : tensor<*xf32>
  }
}
