// RUN: dtensor-opt %s -split-input-file -dtensor-all-scatter-lowering -verify-diagnostics | FileCheck %s --dump-input=fail

func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<4x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: %[[MOD_CONST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DIV_CONST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[PRE_MESH_COORDS:[0-9]*]] = "tf.Div"(%arg0, %[[DIV_CONST]])
  // CHECK-NEXT: %[[MESH_COORDS:.*]] = "tf.FloorMod"(%[[PRE_MESH_COORDS]], %[[MOD_CONST]])
  // CHECK-NEXT: %[[SLICE_SHAPE:.*]] = "tf.Const"
  // CHECK-NEXT: %[[PRE_SLICE_OFFSET:.*]] = "tf.Const"
  // CHECK-NEXT: %[[SLICE_OFFSET:[0-9]*]] = "tf.MatMul"(%[[MESH_COORDS]], %[[PRE_SLICE_OFFSET]])
  // CHECK-NEXT: %[[SQUEEZED_OFFSET:[0-9]*]] = "tf.Squeeze"(%[[SLICE_OFFSET]])
  // CHECK-NEXT: %[[SLICE:[0-9]*]] = "tf.Slice"(%arg1, %[[SQUEEZED_OFFSET]], %[[SLICE_SHAPE]])
  // CHECK:      tf_device.return %[[SLICE]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorAllScatter"(%arg1) {input_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x2xf32>
    tf_device.return %1 : tensor<4x2xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
}
