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

// -----

// Check that element layouts from iterator with optional output is set
// correctly based on iterator resource attribute `tf._element_layouts`.
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
  // CHECK:        "tf.WhileRegion"
  // CHECK:        %[[ITER_OPTIONAL_OUT:.*]] = "tf.IteratorGetNextAsOptional"(%arg1)
  // CHECK-SAME:     _layout = ["sharding_specs: mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:     output_shapes = [#tf_type.shape<8x16>]
  // CHECK-SAME:     (tensor<*x!tf_type.resource>) -> tensor<!tf_type.variant>
  // CHECK-NEXT:   %[[HAS_VALUE:.*]] = "tf.OptionalHasValue"(%[[ITER_OPTIONAL_OUT]])
  // CHECK-SAME:     _layout = ["sharding_specs: mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:     (tensor<!tf_type.variant>) -> tensor<i1>
  // CHECK:        %[[GET_VALUE:.*]] = "tf.OptionalGetValue"(%[[ITER_OPTIONAL_OUT]])
  // CHECK-SAME:     _layout = ["sharding_specs:x,unsharded, mesh:|x=4,y=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:     (tensor<!tf_type.variant>) -> tensor<8x16xf32>
  // CHECK:        tf_device.return
  // CHECK-SAME:     tensor<8x16xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %elem = "tf.IteratorGetNext"(%arg1) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<*x!tf_type.resource>) -> tensor<32x16xf32>
    %while_region:2 = "tf.WhileRegion"(%cst, %elem) ({
      ^bb0(%arg2: tensor<i1>, %arg3: tensor<32x16xf32>):
        %identity = "tf.Identity"(%arg2) {_global_shape = [#tf_type.shape<>]} : (tensor<i1>) -> tensor<i1>
        "tf.Yield"(%identity) {_global_shape = []} : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg2: tensor<i1>, %arg3: tensor<32x16xf32>):
        %iter_optional_out = "tf.IteratorGetNextAsOptional"(%arg1) {_global_shape = [#tf_type.shape<>], output_shapes = [#tf_type.shape<32x16>], output_types = [f32]} : (tensor<*x!tf_type.resource>) -> tensor<!tf_type.variant>
        %has_value = "tf.OptionalHasValue"(%iter_optional_out) {_global_shape = [#tf_type.shape<>]} : (tensor<!tf_type.variant>) -> tensor<i1>
        %if_region:2 = "tf.IfRegion"(%has_value) ({
          %has_value_identity = "tf.Identity"(%has_value) {_global_shape = [#tf_type.shape<>]} : (tensor<i1>) -> tensor<i1>
          %20 = "tf.OptionalGetValue"(%iter_optional_out) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<!tf_type.variant>) -> tensor<32x16xf32>
          %22 = "tf.Identity"(%20) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<32x16xf32>) -> tensor<32x16xf32>
          "tf.Yield"(%has_value_identity, %22) {_global_shape = []} : (tensor<i1>, tensor<32x16xf32>) -> ()
        }, {
          %arg_identity = "tf.Identity"(%arg3) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<32x16xf32>) -> tensor<32x16xf32>
          %has_value_identity = "tf.Identity"(%has_value) {_global_shape = [#tf_type.shape<>]} : (tensor<i1>) -> tensor<i1>
          "tf.Yield"(%has_value_identity, %arg_identity) {_global_shape = []} : (tensor<i1>, tensor<32x16xf32>) -> ()
        }) {_global_shape = [#tf_type.shape<>, #tf_type.shape<32x16>], _lower_using_switch_merge = true, is_stateless = true} : (tensor<i1>) -> (tensor<i1>, tensor<32x16xf32>)
        %1 = "tf.Identity"(%if_region#0) {_global_shape = [#tf_type.shape<>]} : (tensor<i1>) -> tensor<i1>
        %2 = "tf.Identity"(%if_region#1) {_global_shape = [#tf_type.shape<32x16>]} : (tensor<32x16xf32>) -> tensor<32x16xf32>
        "tf.Yield"(%1, %2) {_global_shape = []} : (tensor<i1>, tensor<32x16xf32>) -> ()
      }
    ) {_global_shape = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<32x16>], _lower_using_switch_merge = true, is_stateless = false, shape_invariant} : (tensor<i1>, tensor<32x16xf32>) -> (tensor<i1>, tensor<32x16xf32>)
    tf_device.return %while_region#1 : tensor<32x16xf32>
  }) {_mesh="|x=4,y=2|*TPU"} : () -> (tensor<32x16xf32>)
  func.return
}
