// RUN: tf-opt -split-input-file -verify-diagnostics -tf-tpu-annotate-dynamic-shape-inputs %s | FileCheck %s

// Test that annotate the inputs of the cluster func to be dynamic shaped.

module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
   func.func @main(
      %arg0: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"},
      %arg1: tensor<2048xi64> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> tensor<2048xi32> {
      %cst = "tf.Const"() {value = dense<1024> : tensor<i32>} : () -> tensor<i32>
      %0:2 = "tf_device.launch"() ({
            %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
            %2 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2048xi64>) -> tensor<2048xi32>
            %3:2 = "tf.TPUCopyWithDynamicShape"(%1, %2, %cst, %cst) {operand_segment_sizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
            // CHECK-NOT: tf.TPUAnnotateTensorsWithDynamicShape
            %4:2 = "tf.TPUAnnotateTensorsWithDynamicShape"(%3#0, %3#1) : (tensor<2048xi32>, tensor<2048xi32>) -> (tensor<2048xi32>, tensor<2048xi32>)
            tf_device.return %4#0, %4#1 : tensor<2048xi32>, tensor<2048xi32>
            }) {device = "TPU_REPLICATED_HOST_0"} : () -> (tensor<2048xi32>, tensor<2048xi32>)
      %1 = "tf_device.cluster_func"(%0#0, %0#1) {_replication_info = "cluster_test_fn", func = @tpu_func} : (tensor<2048xi32>, tensor<2048xi32>) -> tensor<2048xi32>
      return %1: tensor<2048xi32>
   }
   // CHECK-LABEL: func @tpu_func
   // CHECK: mhlo.type_extensions
   func.func @tpu_func (
      %arg0: tensor<2048xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<2048xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<2048xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
    %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<2048xi32>, tensor<2048xi32>) -> tensor<2048xi32>
    return %0 : tensor<2048xi32>
   }
}