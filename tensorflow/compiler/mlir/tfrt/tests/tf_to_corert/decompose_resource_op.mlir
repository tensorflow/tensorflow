// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline=decompose-resource-ops=true %s | FileCheck %s --dump-input=fail

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 293 : i32}} {

// CHECK-LABEL: func @gather
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain
// CHECK-SAME: [[arg0:%.*]]: !tfrt_fallback.tf_tensor, [[arg1:%.*]]: !tfrt_fallback.tf_tensor)
// CHECK: [[const:%.*]] = tfrt_fallback_async.const_dense_tensor
// CHECK-NEXT: [[out_chain:%.*]], [[value:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(0) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"({{.*}})
// CHECK-NEXT: [[res:%.*]] = tfrt_fallback_async.executeop key(1) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.GatherV2"([[value]], {{.*}}, [[const]])
// CHECK-NEXT: tfrt.return [[out_chain]], [[res]] : !tfrt.chain, !tfrt_fallback.tf_tensor
func.func @gather(%indices: tensor<?xi32>,
             %resource: tensor<*x!tf_type.resource>) -> tensor<*xi32> {
  %0 = "tf.ResourceGather"(%resource, %indices) {batch_dims = 0 : i64, device = "/device:CPU:0", validate_indices = true}: (tensor<*x!tf_type.resource>, tensor<?xi32>) -> (tensor<*xi32>)
  func.return %0 : tensor<*xi32>
}

}
