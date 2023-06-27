// RUN: LOWER_DTENSOR_GATHER_TO_COLLECTIVE_GATHER_V2=1 dtensor-opt %s -split-input-file -dtensor-all-gather-lowering -verify-diagnostics | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @lower_allgather_tpu_mesh
func.func @lower_allgather_tpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<2x2xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<2x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[UPDATED:.*]] = "tf.XlaDynamicUpdateSlice"(%{{[0-9]*}}, %arg1, %{{[0-9]*}})
  // CHECK:      %[[REDUCED:.*]] = "tf.DTensorAllReduce"(%[[UPDATED]]
  // CHECK:      tf_device.return %[[REDUCED]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorAllGather"(%arg1) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y
=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xf32>) -> tensor<2x4xf32>
    tf_device.return %1 : tensor<2x4xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @lower_allgather_gpu_mesh
func.func @lower_allgather_gpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<2x2xf32> {tf._layout = "sharding_specs:x,y, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"}) -> tensor<2x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:   "tf.Transpose"(%arg1
  // CHECK:  %[[ALLGATHER_OUT:.*]] = "tf.CollectiveGatherV2"
  // CHECK:  "tf.Transpose"(%[[ALLGATHER_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorAllGather"(%arg1) {input_layout = #dtensor.layout<sharding_specs:x,y, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:GPU|x=2,y
=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>} : (tensor<2x2xf32>) -> tensor<2x4xf32>
    tf_device.return %1 : tensor<2x4xf32>
  }) {_mesh = "GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}
