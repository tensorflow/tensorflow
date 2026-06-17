
// RUN: dtensor-opt %s -split-input-file -dtensor-all-to-all-lowering -verify-diagnostics | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @lower_alltoall_tpu_mesh
func.func @lower_alltoall_tpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<4x2xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<2x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[ALLTOALL_OUT:.*]] = "tf.AllToAll"(%arg1
  // CHECK:      tf_device.return %[[ALLTOALL_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorAllToAll"(%arg1) {_layout = ["sharding_specs:x,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"] , 
    input_layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, 
    output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y
=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
    tf_device.return %1 : tensor<2x4xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @lower_alltoall_gpu_mesh
func.func @lower_alltoall_gpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<4x2xf32> {tf._layout = "sharding_specs:x,y, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"}) -> tensor<2x4xf32> {
  // CHECK:  "tf_device.cluster"
  // CHECK:  %[[FLATTEN_OUT:.*]] = "tf.Reshape"(%arg1
  // CHECK:  %[[ALLTOALL_OUT:.*]] = "tf.CollectiveAllToAllV2"(%[[FLATTEN_OUT]]
  // CHECK:  %[[RESHAPE_1_OUT:.*]] = "tf.Reshape"(%[[ALLTOALL_OUT]]
  // CHECK:  %[[TRANSPOSE_OUT:.*]] = "tf.Transpose"(%[[RESHAPE_1_OUT]]
  // CHECK:  %[[RESHAPE_2_OUT:.*]] = "tf.Reshape"(%[[TRANSPOSE_OUT]]
  // CHECK:  tf_device.return %[[RESHAPE_2_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorAllToAll"(%arg1) {_layout = ["sharding_specs:x,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"], 
    input_layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>, 
    output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
    tf_device.return %1 : tensor<2x4xf32>
  }) {_mesh = "GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}
