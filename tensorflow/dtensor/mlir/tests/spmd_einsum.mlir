// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Einsum (normal matrix multiplication)
// No AllToAll on input or output, only AllReduce on output.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:unsharded,y, mesh:|x=2,y=2|*GPU"},
           %arg2: tensor<4x2xi32> {tf._layout = "sharding_specs:y,unsharded, mesh:|x=2,y=2|*GPU"}) -> tensor<2x2xi32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[EINSUM_RESULT:.*]] = "tf.Einsum"(%arg1, %arg2)
  // CHECK:        %[[RETURN:.*]] = "tf.DTensorAllReduce"(%[[EINSUM_RESULT]]
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:   %[[RETURN]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:|x=2,y=2|*GPU>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x2>, layout = #dtensor.layout<sharding_specs:y,unsharded, mesh:|x=2,y=2|*GPU>} : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*GPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %4 : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2|*GPU", _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*GPU"]} : () -> (tensor<2x2xi32>)
  func.return %0 : tensor<2x2xi32>
}

// -----

// Replicated Einsum (normal matrix multiplication, no sharded dimensions reduced)
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2: tensor<4x2xi32> {tf._layout = "sharding_specs:unsharded,y, mesh:|x=2,y=2|*TPU"}) -> tensor<2x2xi32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[EINSUM_RESULT:.*]] = "tf.Einsum"(%arg1, %arg2)
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:   %[[EINSUM_RESULT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x2>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:|x=2,y=2|*TPU>} : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|*TPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %4 : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2|*TPU", _layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*TPU"]} : () -> (tensor<2x2xi32>)
  func.return %0 : tensor<2x2xi32>
}

// -----

// Einsum for transformer, no CRS
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<16x25x8xi32>{ tf._layout="sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2: tensor<25x8x8x50xi32>{ tf._layout="sharding_specs:unsharded,unsharded,y,unsharded, mesh:|x=2,y=2|*TPU"}) -> tensor<16x25x8x50xi32> {
  // CHECK:     "tf_device.cluster"
  // CHECK:       %[[EINSUM_RESULT:.*]] = "tf.Einsum"(%arg1, %arg2)
  // CHECK-NEXT:  tf_device.return
  // CHECK-SAME:  %[[EINSUM_RESULT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<16x25x8>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<16x25x8xi32>) -> tensor<16x25x8xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<25x8x8x50>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<25x8x8x50xi32>) -> tensor<25x8x8x50xi32>
    %3 = "tf.Einsum"(%1, %2) {equation="bse,sehq->bshq"} : (tensor<16x25x8xi32>, tensor<25x8x8x50xi32>) -> tensor<16x25x8x50xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<16x25x8x50>, layout = #dtensor.layout<sharding_specs:x,unsharded,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<16x25x8x50xi32>) -> tensor<16x25x8x50xi32>
    tf_device.return %4 : tensor<16x25x8x50xi32>
  }) {_mesh = "|x=2,y=2|*TPU"} : () -> (tensor<16x25x8x50xi32>)
  func.return %0 : tensor<16x25x8x50xi32>
}

// -----

// Invalid equation
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<16x24x8xi32>{ tf._layout="sharding_specs:x,z,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"},
           %arg2: tensor<24x8x8x50xi32>{ tf._layout="sharding_specs:y,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"}) -> tensor<16x24x8x50xi32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<16x24x8>, layout = #dtensor.layout<sharding_specs:x,z,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<16x24x8xi32>) -> tensor<16x24x8xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<24x8x8x50>, layout = #dtensor.layout<sharding_specs:y,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<24x8x8x50xi32>) -> tensor<24x8x8x50xi32>
    // expected-error @+1 {{incompatible mesh dimensions in equation, label 's' is mapped to mesh dimension 'y' and 'z'}}
    %3 = "tf.Einsum"(%1, %2) {equation="bse,sehq->bshq"} : (tensor<16x24x8xi32>, tensor<24x8x8x50xi32>) -> tensor<16x24x8x50xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<16x24x8x50>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<16x24x8x50xi32>) -> tensor<16x24x8x50xi32>
    tf_device.return %4 : tensor<16x24x8x50xi32>
  }) {_mesh = "TPU|x=2,y=2,z=2|*TPU", _layout = ["sharding_specs:x,unsharded,unsharded,unsharded,  mesh:TPU|x=2,y=2,z=2|*TPU"]} : () -> (tensor<16x24x8x50xi32>)
  func.return %0 : tensor<16x24x8x50xi32>
}

// -----

// y,x . x,y -> *,y
// We unshard %arg1
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:y,x, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%[[GATHERED]], %arg2)
  // CHECK:      %[[FINAL_REDUCE:[0-9]*]] = "tf.DTensorAllReduce"(%[[EINSUM_RESULT]], %cst)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:y,x, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// *,x . x,* -> *,y
// We should slice arg2 before matmul rather than slicing the result.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[SLICE:[0-9]*]] = "tf.DTensorAllScatter"(%arg2)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%arg1, %[[SLICE]])
  // CHECK:      %[[FINAL_REDUCE:[0-9]*]] = "tf.DTensorAllReduce"(%[[EINSUM_RESULT]], %cst)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// x,y . *,y -> x,y
// We unshard %arg1 on y.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%[[GATHERED]], %arg2)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// Example from BERT 64 way sharding.
// bsd,dnh->bsnh  x,*,y . *,y,* -> x,*,y,*
// Unshard arg1 along y.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<8x128x128xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<128x16x64xf32> {tf._layout = "sharding_specs:unsharded,y,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<8x128x16x64xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%[[GATHERED]], %arg2)
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[EINSUM_RESULT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128xf32>) -> tensor<8x128x128xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<128x16x64>, layout = #dtensor.layout<sharding_specs:unsharded,y,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<128x16x64xf32>) -> tensor<128x16x64xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="bsd,dnh->bsnh"} : (tensor<8x128x128xf32>, tensor<128x16x64xf32>) -> tensor<8x128x16x64xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x128x16x64>, layout = #dtensor.layout<sharding_specs:x,unsharded,y,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x16x64xf32>) -> tensor<8x128x16x64xf32>
    tf_device.return %4 : tensor<8x128x16x64xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<8x128x16x64xf32>
  func.return %0 : tensor<8x128x16x64xf32>
}

// -----

// Example from BERT 64 way sharding.
// bfd,bfi->id  x,*,y . x,*,y -> y,*
// Unshard arg1 along y, reduce on output.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<8x128x128xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<8x128x256xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<256x128xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%[[GATHERED]], %arg2)
  // CHECK:      %[[RETURN:.*]] = "tf.DTensorAllReduce"(%[[EINSUM_RESULT]]
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[RETURN]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128xf32>) -> tensor<8x128x128xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x128x256>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x256xf32>) -> tensor<8x128x256xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="bfd,bfi->id"} : (tensor<8x128x128xf32>, tensor<8x128x256xf32>) -> tensor<256x128xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<256x128>, layout = #dtensor.layout<sharding_specs:y,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    tf_device.return %4 : tensor<256x128xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<256x128xf32>
  func.return %0 : tensor<256x128xf32>
}

// -----

// Example from BERT 64 way sharding.
// bfi,bfd->di  x,*,y . x,*,y -> *,y
// Unshard arg2 along y, reduce on output.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<8x128x256xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU"},
           %arg2: tensor<8x128x128xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<128x256xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg2)
  // CHECK-NEXT: %[[EINSUM_RESULT:[0-9]*]] = "tf.Einsum"(%arg1, %[[GATHERED]])
  // CHECK:      %[[RETURN:.*]] = "tf.DTensorAllReduce"(%[[EINSUM_RESULT]]
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[RETURN]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x256>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x256xf32>) -> tensor<8x128x256xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x128x128>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128xf32>) -> tensor<8x128x128xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="bfi,bfd->di"} : (tensor<8x128x256xf32>, tensor<8x128x128xf32>) -> tensor<128x256xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<128x256>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|*TPU>} : (tensor<128x256xf32>) -> tensor<128x256xf32>
    tf_device.return %4 : tensor<128x256xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<128x256xf32>
  func.return %0 : tensor<128x256xf32>
}
