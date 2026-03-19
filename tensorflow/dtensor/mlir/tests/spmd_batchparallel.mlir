// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Test replicated layout.

func.func @main(%arg0: tensor<1xf32>,
           %arg1: tensor<8x128x128x3xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<8x128x128x3xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.AdjustSaturation"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    %3 = "tf.AdjustSaturation"(%2, %1) {} : (tensor<8x128x128x3xf32>, tensor<1xf32>) -> tensor<8x128x128x3xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    tf_device.return %4 : tensor<8x128x128x3xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<8x128x128x3xf32>
  func.return %0 : tensor<8x128x128x3xf32>
}

// -----

// Test batch sharded layout. Should emit Identity op.

func.func @main(%arg0: tensor<1xf32>,
           %arg1: tensor<8x128x128x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<8x128x128x3xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.AdjustSaturation"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    %3 = "tf.AdjustSaturation"(%2, %1) {} : (tensor<8x128x128x3xf32>, tensor<1xf32>) -> tensor<8x128x128x3xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    tf_device.return %4 : tensor<8x128x128x3xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<8x128x128x3xf32>
  func.return %0 : tensor<8x128x128x3xf32>
}
