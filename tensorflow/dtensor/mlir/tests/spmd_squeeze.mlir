// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check Squeeze with postive index.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> , %arg1: tensor<2x1xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> tensor<2xf32> {
  // CHECK:      "tf.Squeeze"(%arg1)
  // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=1|0,1|0,1|
  // CHECK-SAME: (tensor<1x1xf32>) -> tensor<1xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<2x1>], global_shape = #tf_type.shape<2x1>,
      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %2 = "tf.Squeeze"(%1) {_global_shape = [#tf_type.shape<2>], device = "", squeeze_dims = [1]} : (tensor<2x1xf32>) -> tensor<2xf32>
    tf_device.return %2 : tensor<2xf32>
  }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Check Squeeze with negative index.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> , %arg1: tensor<2x1xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> tensor<2xf32> {
  // CHECK:      "tf.Squeeze"(%arg1)
  // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=1|0,1|0,1|
  // CHECK-SAME: (tensor<1x1xf32>) -> tensor<1xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<2x1>], global_shape = #tf_type.shape<2x1>,
      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %2 = "tf.Squeeze"(%1) {_global_shape = [#tf_type.shape<2>], device = "", squeeze_dims = [-1]} : (tensor<2x1xf32>) -> tensor<2xf32>
    tf_device.return %2 : tensor<2xf32>
  }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Check Squeeze that does not locally squeeze the dim with local shape 1.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> , %arg1: tensor<2x1xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> tensor<2xf32> {
  // CHECK:      "tf.Squeeze"(%arg1)
  // CHECK-SAME: squeeze_dims = [1]
  // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=1|0,1|0,1|
  // CHECK-SAME: (tensor<1x1xf32>) -> tensor<1xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<2x1>], global_shape = #tf_type.shape<2x1>,
      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %2 = "tf.Squeeze"(%1) {_global_shape = [#tf_type.shape<2>], device = "", squeeze_dims = []} : (tensor<2x1xf32>) -> tensor<2xf32>
    tf_device.return %2 : tensor<2xf32>
  }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

