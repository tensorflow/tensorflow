// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check that IteratorGetNextOp layout is set correctly based on iterator
// resource attribute `tf._element_layouts`.
// CHECK-LABEL: func @main
func.func @main(
    %arg0: tensor<1xi32>,
    %arg1: tensor<*x!tf_type.resource> {
        tf._element_layouts = ["sharding_specs:x,unsharded, mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"],
        tf._layout = "sharding_specs: mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/task:0/device:CPU:7"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[ITER_OUT:.*]] = "tf.IteratorGetNext"(%arg1)
  // CHECK-SAME:     _layout = ["sharding_specs:x,unsharded, mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:     (tensor<*x!tf_type.resource>) -> tensor<8x16xf32>
  // CHECK:        tf_device.return
  // CHECK-SAME:     tensor<8x16xf32>
  %0 = "tf_device.cluster"() ({
    %elem = "tf.IteratorGetNext"(%arg1) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<*x!tf_type.resource>) -> tensor<32x16xf32>
    %identity = "tf.Identity"(%elem) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    tf_device.return %identity : tensor<32x16xf32>
  }) {_mesh="|x=4,y=2|*TPU"} : () -> (tensor<32x16xf32>)
  func.return
}

// -----

// Check that IteratorGetNextOp layout is set correctly based on iterator
// resource attribute `tf._element_layouts`, where a DTensorLayout op has been
// applied to the iterator resource tensor.
// CHECK-LABEL: func @main
func.func @main(
    %arg0: tensor<1xi32>,
    %arg1: tensor<*x!tf_type.resource> {
        tf._element_layouts = ["sharding_specs:x,unsharded, mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"],
        tf._layout = "sharding_specs: mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/task:0/device:CPU:7"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[ITER_OUT:.*]] = "tf.IteratorGetNext"(%arg1)
  // CHECK-SAME:     _layout = ["sharding_specs:x,unsharded, mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:     (tensor<*x!tf_type.resource>) -> tensor<8x16xf32>
  // CHECK:        tf_device.return
  // CHECK-SAME:     tensor<8x16xf32>
  %0 = "tf_device.cluster"() ({
    %elem_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<*>, layout = #dtensor.layout<sharding_specs: mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/task:0/device:CPU:7>} : (tensor<*x!tf_type.resource>) -> tensor<*x!tf_type.resource>
    %elem = "tf.IteratorGetNext"(%elem_layout) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<*x!tf_type.resource>) -> tensor<32x16xf32>
    %identity = "tf.Identity"(%elem) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    tf_device.return %identity : tensor<32x16xf32>
  }) {_mesh="|x=4,y=2|*TPU"} : () -> (tensor<32x16xf32>)
  func.return
}
