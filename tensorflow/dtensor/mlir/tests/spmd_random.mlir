// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Random with no sharding
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: %[[SHAPE:.*]] = "tf.Const"()
  // CHECK-SAME: value = dense<[32, 32, 64]>
  // CHECK-NEXT: %[[RANDOM:.*]] = "tf.StatelessRandomUniform"(%[[SHAPE]], %arg1)
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[RANDOM]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2xi32>) -> tensor<2xi32>
    %2 = "tf.Const"() {value = dense<[32, 32, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
    %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<3xi64>) -> tensor<3xi64>
    %4 = "tf.StatelessRandomUniform"(%3, %1) : (tensor<3xi64>, tensor<2xi32>) -> tensor<32x32x64xf32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<32x32x64>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<32x32x64xf32>) -> tensor<32x32x64xf32>
    tf_device.return %5 : tensor<32x32x64xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Random with x,z sharding
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:TPU|x=4,y=2,z=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:10,/job:localhost/task:0/device:TPU:11,/job:localhost/task:0/device:TPU:12,/job:localhost/task:0/device:TPU:13,/job:localhost/task:0/device:TPU:14,/job:localhost/task:0/device:TPU:15"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: %[[MESH_SIZES:.*]] = "tf.Const"()
  // CHECK-SAME: 4, 2, 2
  // CHECK-NEXT: %[[MESH_SIZES_RUNNING_PRODUCT:.*]] = "tf.Const"() {value =
  // CHECK-SAME: 4, 2, 1
  // CHECK-NEXT: %[[MESH_COORDS_PRE_MOD:.*]] = "tf.Div"(%arg0, %[[MESH_SIZES_RUNNING_PRODUCT]])
  // CHECK-NEXT: %[[MESH_COORDS:.*]] = "tf.FloorMod"(%[[MESH_COORDS_PRE_MOD]], %[[MESH_SIZES]])
  // CHECK-SAME: _mesh_coordinates =
  // CHECK-NEXT: %[[MESH_MULTIPLER:.*]] = "tf.Const"()
  // CHECK-SAME [65536], [0], [262144]
  // CHECK-NEXT: %[[DEVICE_SEED:.*]] = "tf.MatMul"(%[[MESH_COORDS]], %[[MESH_MULTIPLER]])
  // CHECK-NEXT: %[[PRIME:.*]] = "tf.Const"() {value = dense<65521>
  // CHECK-NEXT: %[[DEVICE_SEED_PRIME:.*]] = "tf.AddV2"(%[[DEVICE_SEED]], %[[PRIME]])
  // CHECK-NEXT: %[[DEVICE_SEED_SQUEEZE:.*]] = "tf.Squeeze"(%[[DEVICE_SEED_PRIME]]) {
  // CHECK-NOT: dtensor.device_seed_for_mesh_dims
  // CHECK-SAME: }
  // CHECK-NEXT: %[[OLD_SHAPE:.*]] = "tf.Const"(
  // CHECK-NEXT: %[[DEVICE_SEED_CAST:.*]] = "tf.Cast"(%[[DEVICE_SEED_SQUEEZE]])
  // CHECK-NEXT: %[[NEW_SEED:.*]] = "tf.BitwiseXor"(%arg1, %[[DEVICE_SEED_CAST]])
  // CHECK-NEXT: %[[NEW_SHAPE:.*]] = "tf.Const"() {value = dense<[8, 32, 32]>
  // CHECK-NEXT: %[[RANDOM:.*]] = "tf.StatelessRandomUniform"(%[[NEW_SHAPE]], %[[NEW_SEED]])
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[RANDOM]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=4,y=2,z=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:10,/job:localhost/task:0/device:TPU:11,/job:localhost/task:0/device:TPU:12,/job:localhost/task:0/device:TPU:13,/job:localhost/task:0/device:TPU:14,/job:localhost/task:0/device:TPU:15>} : (tensor<2xi32>) -> tensor<2xi32>
    %2 = "tf.Const"() {value = dense<[32, 32, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
    %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=4,y=2,z=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:10,/job:localhost/task:0/device:TPU:11,/job:localhost/task:0/device:TPU:12,/job:localhost/task:0/device:TPU:13,/job:localhost/task:0/device:TPU:14,/job:localhost/task:0/device:TPU:15>} : (tensor<3xi64>) -> tensor<3xi64>
    %4 = "tf.StatelessRandomUniform"(%3, %1) : (tensor<3xi64>, tensor<2xi32>) -> tensor<32x32x64xf32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<32x32x64>, layout = #dtensor.layout<sharding_specs:x,unsharded,z, mesh:TPU|x=4,y=2,z=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:10,/job:localhost/task:0/device:TPU:11,/job:localhost/task:0/device:TPU:12,/job:localhost/task:0/device:TPU:13,/job:localhost/task:0/device:TPU:14,/job:localhost/task:0/device:TPU:15>} : (tensor<32x32x64xf32>) -> tensor<32x32x64xf32>
    tf_device.return %5 : tensor<32x32x64xf32>
  }) {_mesh = "TPU|x=4,y=2,z=2|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:8,/job:localhost/task:0/device:TPU:10,/job:localhost/task:0/device:TPU:11,/job:localhost/task:0/device:TPU:12,/job:localhost/task:0/device:TPU:13,/job:localhost/task:0/device:TPU:14,/job:localhost/task:0/device:TPU:15"} : () -> (tensor<i32>)
  func.return
}
