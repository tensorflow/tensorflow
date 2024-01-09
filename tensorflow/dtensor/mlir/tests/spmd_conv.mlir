// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check that Conv2D uses input image layout as output layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
           %arg2:tensor<8x3x3x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2D"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK-SAME: (tensor<4x32x32x3xf32>, tensor<8x3x3x3xf32>) -> tensor<4x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU>} : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU>} : (tensor<8x3x3x3xf32>) -> tensor<8x3x3x3xf32>
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<8x3x3x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="|x=2,y=1|*TPU"} : () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv3D uses input image layout as output layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
           %arg2:tensor<8x3x3x3x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3D"
  // CHECK-SAME: data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: (tensor<4x32x32x32x3xf32>, tensor<8x3x3x3x3xf32>) -> tensor<4x32x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU>} : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU>} : (tensor<8x3x3x3x3xf32>) -> tensor<8x3x3x3x3xf32>
    %conv = "tf.Conv3D"(%img_layout, %filter_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<8x3x3x3x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x32x3xf32>
  }) {_mesh="|x=2,y=1|*TPU"} : () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv2D backprop uses grads as output layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<1x3x3x3xf32>,
           %arg2: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropInput"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK-SAME: (tensor<4xi32>, tensor<1x3x3x3xf32>, tensor<4x32x32x3xf32>) -> tensor<4x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_shape = "tf.Const"() { value=dense<[8,32,32,3]> : tensor<4xi32>} : () -> tensor<4xi32>
    %filter_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropInput"(%img_shape, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<4xi32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv3DBackPropInputV2 uses grads as output layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<1x3x3x3x3xf32>,
           %arg2: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3DBackpropInputV2"
  // CHECK-SAME: data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: (tensor<5xi32>, tensor<1x3x3x3x3xf32>, tensor<4x32x32x32x3xf32>) -> tensor<4x32x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_shape = "tf.Const"() { value=dense<[8,32,32,32,3]> : tensor<5xi32>} : () -> tensor<5xi32>
    %filter_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<1x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropInputV2"(%img_shape, %filter_layout, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<5xi32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check all reduce emitted in Conv2DBackpropFilter when image is batch sharded
// but input_shape is replicated const.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %input_img: tensor<2x9x9x1xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
  %grad: tensor<2x9x9x2xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU"}
  ) -> tensor<2x2x1x2xf32> {
  // CHECK: "tf_device.cluster"
  // CHECK: %[[FILTER_SHAPE:.*]] = "tf.Const"()
  // CHECK: %[[SLICE_OUT:.*]] = "tf.DTensorAllScatter"(%arg2)
  // CHECK: %[[BACKPROP_OUT:.*]] = "tf.Conv2DBackpropFilter"(%arg1, %[[FILTER_SHAPE]], %[[SLICE_OUT]])
  // CHECK: %[[TPU_GROUP:.*]] = "tf.Const"()
  // CHECK: %[[XLA_ALL_REDUCE:.*]] = "tf.DTensorAllReduce"(%[[BACKPROP_OUT]], %[[TPU_GROUP]])
  %0 = "tf_device.cluster"() ({
    %filter_shape = "tf.Const"() { value = dense<[2, 2, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %input_layout = "tf.DTensorLayout"(%input_img) {global_shape = #tf_type.shape<2x9x9x1>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<2x9x9x1xf32>) -> tensor<2x9x9x1xf32>
    %grad_layout = "tf.DTensorLayout"(%grad) {global_shape = #tf_type.shape<2x9x9x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32>
    %2 = "tf.Conv2DBackpropFilter"(%input_layout, %filter_shape, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<2x9x9x1xf32>, tensor<4xi32>, tensor<2x9x9x2xf32>) -> tensor<2x2x1x2xf32>
    tf_device.return %2 : tensor<2x2x1x2xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"} : () -> tensor<2x2x1x2xf32>
  func.return %0 : tensor<2x2x1x2xf32>
}

// -----

// Check all reduce emitted in Conv3DBackpropFilterV2 when image is batch
// sharded but input_shape is replicated const.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %input_img: tensor<2x9x9x9x1xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
  %grad: tensor<2x9x9x9x2xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU"}
  ) -> tensor<2x2x2x1x2xf32> {
  // CHECK: "tf_device.cluster"
  // CHECK: %[[FILTER_SHAPE:.*]] = "tf.Const"()
  // CHECK: %[[SLICE_OUT:.*]] = "tf.DTensorAllScatter"(%arg2)
  // CHECK: %[[BACKPROP_OUT:.*]] = "tf.Conv3DBackpropFilterV2"(%arg1, %[[FILTER_SHAPE]], %[[SLICE_OUT]])
  // CHECK: %[[TPU_GROUP:.*]] = "tf.Const"()
  // CHECK: %[[XLA_ALL_REDUCE:.*]] = "tf.DTensorAllReduce"(%[[BACKPROP_OUT]], %[[TPU_GROUP]])
  %0 = "tf_device.cluster"() ({
    %filter_shape = "tf.Const"() { value = dense<[2, 2, 2, 1, 2]> : tensor<5xi32>} : () -> tensor<5xi32>
    %input_layout = "tf.DTensorLayout"(%input_img) {global_shape = #tf_type.shape<2x9x9x9x1>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<2x9x9x9x1xf32>) -> tensor<2x9x9x9x1xf32>
    %grad_layout = "tf.DTensorLayout"(%grad) {global_shape = #tf_type.shape<2x9x9x9x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<2x9x9x9x2xf32>) -> tensor<2x9x9x9x2xf32>
    %2 = "tf.Conv3DBackpropFilterV2"(%input_layout, %filter_shape, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<2x9x9x9x1xf32>, tensor<5xi32>, tensor<2x9x9x9x2xf32>) -> tensor<2x2x2x1x2xf32>
    tf_device.return %2 : tensor<2x2x2x1x2xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"} : () -> tensor<2x2x2x1x2xf32>
  func.return %0 : tensor<2x2x2x1x2xf32>
}

// -----

// Check that Conv2D with spatial partitioning requires kernel to be fully
// replicated.
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<8x3x3x3xf32>) {
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x3x3x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x3x3x3xf32>) -> tensor<8x3x3x3xf32>
    // expected-error @+1 {{Filter for convolution must have fully replicated layout.}}
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<8x32x32x3xf32>, tensor<8x3x3x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv3D with spatial partitioning requires kernel to be fully
// replicated.
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<8x3x3x3x3xf32>) {
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x3x3x3x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x3x3x3x3xf32>) -> tensor<8x3x3x3x3xf32>
    // expected-error @+1 {{Filter for convolution must have fully replicated layout.}}
    %conv = "tf.Conv3D"(%img_layout, %filter_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<8x32x32x32x3xf32>, tensor<8x3x3x3x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x32x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv2D with spatial partitioning requires input dimension size to
// be greater than halo size.
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<15x15x3x3xf32>) {
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x8x3xf32>) -> tensor<8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<15x15x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<15x15x3x3xf32>) -> tensor<15x15x3x3xf32>
    // expected-error @+1 {{input shard tensor size of each processor must be greater than halo size}}
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<8x8x8x3xf32>, tensor<15x15x3x3xf32>) -> tensor<8x8x8x3xf32>
    tf_device.return %conv : tensor<8x8x8x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x8x8x3xf32>)
  func.return
}

// -----

// Check that Conv3D with spatial partitioning requires input dimension size to
// be greater than halo size.
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<15x3x15x3x3xf32>) {
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x8x8x3xf32>) -> tensor<8x8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<15x3x15x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<15x3x15x3x3xf32>) -> tensor<15x3x15x3x3xf32>
    // expected-error @+1 {{input shard tensor size of each processor must be greater than halo size}}
    %conv = "tf.Conv3D"(%img_layout, %filter_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<8x8x8x8x3xf32>, tensor<15x3x15x3x3xf32>) -> tensor<8x8x8x8x3xf32>
    tf_device.return %conv : tensor<8x8x8x8x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x8x8x8x3xf32>)
  func.return
}

// -----

// Check that Conv2D with spatial partitioning using "SAME" padding produces
// begin and end halos on both spatial dimensions.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<3x3x3x3xf32>) {
  // CHECK:         "tf_device.cluster"

  // Build left halo on height dim.
  // CHECK:           %[[SLICE_H_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 3, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT:.*]] = "tf.Slice"(%arg1, %[[SLICE_H_LEFT_BEGIN]], %[[SLICE_H_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_H_LEFT]], %[[PAIRS_H_LEFT]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<4x2xi32>) -> tensor<8x1x4x3xf32>
  // Build right halo on height dim.
  // CHECK:           %[[SLICE_H_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT:.*]] = "tf.Slice"(%arg1, %[[SLICE_H_RIGHT_BEGIN]], %[[SLICE_H_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_H_RIGHT]], %[[PAIRS_H_RIGHT]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<4x2xi32>) -> tensor<8x1x4x3xf32>
  // Concat the halos with the shard on the height dim.
  // CHECK-NEXT:      %[[CONCAT_H_AXIS:.*]] = "tf.Const"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_H_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_H_LEFT]], %arg1, %[[EXCHANGED_HALO_H_RIGHT]], %[[CONCAT_H_AXIS]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<8x4x4x3xf32>, tensor<8x1x4x3xf32>, tensor<i64>) -> tensor<8x6x4x3xf32>

  // Build left halo on width dim.
  // CHECK:           %[[SLICE_W_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 3, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT:.*]] = "tf.Slice"(%[[CONCAT_H_TENSOR]], %[[SLICE_W_LEFT_BEGIN]], %[[SLICE_W_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x6x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_W_LEFT]], %[[PAIRS_W_LEFT]])
  // CHECK-SAME:          (tensor<8x6x1x3xf32>, tensor<4x2xi32>) -> tensor<8x6x1x3xf32>
  // Build right halo on width dim.
  // CHECK:           %[[SLICE_W_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT:.*]] = "tf.Slice"(%[[CONCAT_H_TENSOR]], %[[SLICE_W_RIGHT_BEGIN]], %[[SLICE_W_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x6x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_W_RIGHT]], %[[PAIRS_W_RIGHT]])
  // CHECK-SAME:          (tensor<8x6x1x3xf32>, tensor<4x2xi32>) -> tensor<8x6x1x3xf32>
  // Concat the halos with the shard on the width dim.
  // CHECK-NEXT:      %[[CONCAT_W_AXIS:.*]] = "tf.Const"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_HW_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_W_LEFT]], %[[CONCAT_H_TENSOR]], %[[EXCHANGED_HALO_W_RIGHT]], %[[CONCAT_W_AXIS]])
  // CHECK-SAME:          (tensor<8x6x1x3xf32>, tensor<8x6x4x3xf32>, tensor<8x6x1x3xf32>, tensor<i64>) -> tensor<8x6x6x3xf32>

  // CHECK-NEXT:      "tf.Conv2D"(%[[CONCAT_HW_TENSOR]], %arg2)
  // CHECK-SAME:          padding = "VALID"
  // CHECK-SAME:          (tensor<8x6x6x3xf32>, tensor<3x3x3x3xf32>) -> tensor<8x4x4x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x8x3xf32>) -> tensor<8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<8x8x8x3xf32>, tensor<3x3x3x3xf32>) -> tensor<8x8x8x3xf32>
    tf_device.return %conv : tensor<8x8x8x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x8x8x3xf32>)
  func.return
}

// -----

// Check that Conv3D with spatial partitioning using "SAME" padding produces
// begin and end halos on all spatial dimensions.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,z,unsharded, mesh:|x=2,y=2,z=2|*TPU"},
           %arg2:tensor<3x3x3x3x3xf32>) {
  // CHECK:         "tf_device.cluster"

  // Build left halo on depth dim.
  // CHECK:           %[[SLICE_D_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 3, 0, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_LEFT:.*]] = "tf.Slice"(%arg1, %[[SLICE_D_LEFT_BEGIN]], %[[SLICE_D_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x1x4x4x3xf32>
  // CHECK-NEXT:      %[[HALO_D_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_D_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 0], [5, 1], [6, 2], [7, 3]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_D_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_D_LEFT]], %[[PAIRS_D_LEFT]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x2xi32>) -> tensor<8x1x4x4x3xf32>
  // Build right halo on depth dim.
  // CHECK:           %[[SLICE_D_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_RIGHT:.*]] = "tf.Slice"(%arg1, %[[SLICE_D_RIGHT_BEGIN]], %[[SLICE_D_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x1x4x4x3xf32>
  // CHECK-NEXT:      %[[HALO_D_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_D_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 0], [5, 1], [6, 2], [7, 3]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_D_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_D_RIGHT]], %[[PAIRS_D_RIGHT]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x2xi32>) -> tensor<8x1x4x4x3xf32>
  // Concat the halos with the shard on the depth dim.
  // CHECK-NEXT:      %[[CONCAT_D_AXIS:.*]] = "tf.Const"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_D_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_D_LEFT]], %arg1, %[[EXCHANGED_HALO_D_RIGHT]], %[[CONCAT_D_AXIS]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x4x4x4x3xf32>, tensor<8x1x4x4x3xf32>, tensor<i64>) -> tensor<8x6x4x4x3xf32>

  // Build left halo on height dim.
  // CHECK:           %[[SLICE_H_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 3, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 1, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT:.*]] = "tf.Slice"(%[[CONCAT_D_TENSOR]], %[[SLICE_H_LEFT_BEGIN]], %[[SLICE_H_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x6x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1], [4, 6], [5, 7], [6, 4], [7, 5]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_H_LEFT]], %[[PAIRS_H_LEFT]])
  // CHECK-SAME:          (tensor<8x6x1x4x3xf32>, tensor<8x2xi32>) -> tensor<8x6x1x4x3xf32>
  // Build right halo on height dim.
  // CHECK:           %[[SLICE_H_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 1, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT:.*]] = "tf.Slice"(%[[CONCAT_D_TENSOR]], %[[SLICE_H_RIGHT_BEGIN]], %[[SLICE_H_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x6x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1], [4, 6], [5, 7], [6, 4], [7, 5]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_H_RIGHT]], %[[PAIRS_H_RIGHT]])
  // CHECK-SAME:          (tensor<8x6x1x4x3xf32>, tensor<8x2xi32>) -> tensor<8x6x1x4x3xf32>
  // Concat the halos with the shard on the height dim.
  // CHECK-NEXT:      %[[CONCAT_H_AXIS:.*]] = "tf.Const"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_DH_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_H_LEFT]], %[[CONCAT_D_TENSOR]], %[[EXCHANGED_HALO_H_RIGHT]], %[[CONCAT_H_AXIS]])
  // CHECK-SAME:          (tensor<8x6x1x4x3xf32>, tensor<8x6x4x4x3xf32>, tensor<8x6x1x4x3xf32>, tensor<i64>) -> tensor<8x6x6x4x3xf32>

  // Build left halo on width dim.
  // CHECK:           %[[SLICE_W_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 0, 3, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 6, 1, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT:.*]] = "tf.Slice"(%[[CONCAT_DH_TENSOR]], %[[SLICE_W_LEFT_BEGIN]], %[[SLICE_W_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x6x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x6x6x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_W_LEFT]], %[[PAIRS_W_LEFT]])
  // CHECK-SAME:          (tensor<8x6x6x1x3xf32>, tensor<8x2xi32>) -> tensor<8x6x6x1x3xf32>
  // Build right halo on width dim.
  // CHECK:           %[[SLICE_W_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 6, 6, 1, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT:.*]] = "tf.Slice"(%[[CONCAT_DH_TENSOR]], %[[SLICE_W_RIGHT_BEGIN]], %[[SLICE_W_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x6x6x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x6x6x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_W_RIGHT]], %[[PAIRS_W_RIGHT]])
  // CHECK-SAME:          (tensor<8x6x6x1x3xf32>, tensor<8x2xi32>) -> tensor<8x6x6x1x3xf32>
  // Concat the halos with the shard on the width dim.
  // CHECK-NEXT:      %[[CONCAT_W_AXIS:.*]] = "tf.Const"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_DHW_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_W_LEFT]], %[[CONCAT_DH_TENSOR]], %[[EXCHANGED_HALO_W_RIGHT]], %[[CONCAT_W_AXIS]])
  // CHECK-SAME:          (tensor<8x6x6x1x3xf32>, tensor<8x6x6x4x3xf32>, tensor<8x6x6x1x3xf32>, tensor<i64>) -> tensor<8x6x6x6x3xf32>

  // CHECK-NEXT:      "tf.Conv3D"(%[[CONCAT_DHW_TENSOR]], %arg2)
  // CHECK-SAME:          padding = "VALID"
  // CHECK-SAME:          (tensor<8x6x6x6x3xf32>, tensor<3x3x3x3x3xf32>) -> tensor<8x4x4x4x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,z,unsharded, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<8x8x8x8x3xf32>) -> tensor<8x8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2,z=8|*TPU>} : (tensor<3x3x3x3x3xf32>) -> tensor<3x3x3x3x3xf32>
    %conv = "tf.Conv3D"(%img_layout, %filter_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<8x8x8x8x3xf32>, tensor<3x3x3x3x3xf32>) -> tensor<8x8x8x8x3xf32>
    tf_device.return %conv : tensor<8x8x8x8x3xf32>
  }) {_mesh="|x=2,y=2,z=2|*TPU"} : () -> (tensor<8x8x8x8x3xf32>)
  func.return
}

// -----

// Check that Conv2D with spatial partitioning using "VALID" padding produces
// begin and end halos on both spatial dimensions and all necessary slice ops.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2:tensor<3x3x3x3xf32>) {
  // CHECK:         "tf_device.cluster"

  // Build left halo on height dim.
  // CHECK:           %[[SLICE_H_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 3, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT:.*]] = "tf.Slice"(%arg1, %[[SLICE_H_LEFT_BEGIN]], %[[SLICE_H_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_H_LEFT]], %[[PAIRS_H_LEFT]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<4x2xi32>) -> tensor<8x1x4x3xf32>
  // Build right halo on height dim.
  // CHECK:           %[[SLICE_H_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT:.*]] = "tf.Slice"(%arg1, %[[SLICE_H_RIGHT_BEGIN]], %[[SLICE_H_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_H_RIGHT]], %[[PAIRS_H_RIGHT]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<4x2xi32>) -> tensor<8x1x4x3xf32>
  // Concat the halos with the shard on the height dim.
  // CHECK-NEXT:      %[[CONCAT_H_AXIS:.*]] = "tf.Const"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_H_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_H_LEFT]], %arg1, %[[EXCHANGED_HALO_H_RIGHT]], %[[CONCAT_H_AXIS]])
  // CHECK-SAME:          (tensor<8x1x4x3xf32>, tensor<8x4x4x3xf32>, tensor<8x1x4x3xf32>, tensor<i64>) -> tensor<8x6x4x3xf32>
  // Dynamically slice the concatenated tensor to get correct size for VALID padding.
  // CHECK-NEXT:      %[[HALO_SIZES_H:.*]] = "tf.Const"() <{value = dense<[0, 1, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[HALO_INCREMENTS_H:.*]] = "tf.Const"() <{value = dense<[0, 1, 0, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[VALID_OFFSET_H:.*]] = "tf.Mul"
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_H:.*]] = "tf.Sub"(%[[HALO_SIZES_H]], %[[VALID_OFFSET_H]])
  // CHECK-NEXT:      %[[VALID_SLICE_SIZE_H:.*]] = "tf.Const"() <{value = dense<[8, 5, 4, 3]> : tensor<4xi64>}> : () -> tensor<4xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_CAST_I64_H:.*]] = "tf.Cast"(%[[VALID_SLICE_BEGIN_H]]) <{Truncate = false}> : (tensor<4xi32>) -> tensor<4xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_H_TENSOR:.*]] = "tf.Slice"(%[[CONCAT_H_TENSOR]], %[[VALID_SLICE_BEGIN_CAST_I64_H]], %[[VALID_SLICE_SIZE_H]])

  // Build left halo on width dim.
  // CHECK:           %[[SLICE_W_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 3, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT:.*]] = "tf.Slice"(%[[VALID_SLICE_H_TENSOR]], %[[SLICE_W_LEFT_BEGIN]], %[[SLICE_W_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x5x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_W_LEFT]], %[[PAIRS_W_LEFT]])
  // CHECK-SAME:          (tensor<8x5x1x3xf32>, tensor<4x2xi32>) -> tensor<8x5x1x3xf32>
  // Build right halo on width dim.
  // CHECK:           %[[SLICE_W_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT:.*]] = "tf.Slice"(%[[VALID_SLICE_H_TENSOR]], %[[SLICE_W_RIGHT_BEGIN]], %[[SLICE_W_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x4x3xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<8x5x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_W_RIGHT]], %[[PAIRS_W_RIGHT]])
  // CHECK-SAME:          (tensor<8x5x1x3xf32>, tensor<4x2xi32>) -> tensor<8x5x1x3xf32>
  // Concat the halos with the shard on the width dim.
  // CHECK-NEXT:      %[[CONCAT_W_AXIS:.*]] = "tf.Const"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_HW_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_W_LEFT]], %[[VALID_SLICE_H_TENSOR]], %[[EXCHANGED_HALO_W_RIGHT]], %[[CONCAT_W_AXIS]])
  // CHECK-SAME:          (tensor<8x5x1x3xf32>, tensor<8x5x4x3xf32>, tensor<8x5x1x3xf32>, tensor<i64>) -> tensor<8x5x6x3xf32>
  // Dynamically slice the concatenated tensor to get correct size for VALID padding.
  // CHECK-NEXT:      %[[HALO_SIZES_W:.*]] = "tf.Const"() <{value = dense<[0, 0, 1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[HALO_INCREMENTS_W:.*]] = "tf.Const"() <{value = dense<[0, 0, 1, 0]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NEXT:      %[[VALID_OFFSET_W:.*]] = "tf.Mul"
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_W:.*]] = "tf.Sub"(%[[HALO_SIZES_W]], %[[VALID_OFFSET_W]])
  // CHECK-NEXT:      %[[VALID_SLICE_SIZE_W:.*]] = "tf.Const"() <{value = dense<[8, 5, 5, 3]> : tensor<4xi64>}> : () -> tensor<4xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_CAST_I64_W:.*]] = "tf.Cast"(%[[VALID_SLICE_BEGIN_W]]) <{Truncate = false}> : (tensor<4xi32>) -> tensor<4xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_HW_TENSOR:.*]] = "tf.Slice"(%[[CONCAT_HW_TENSOR]], %[[VALID_SLICE_BEGIN_CAST_I64_W]], %[[VALID_SLICE_SIZE_W]])

  // CHECK-NEXT:      "tf.Conv2D"(%[[VALID_SLICE_HW_TENSOR]], %arg2)
  // CHECK-SAME:          padding = "VALID"
  // CHECK-SAME:          (tensor<8x5x5x3xf32>, tensor<3x3x3x3xf32>) -> tensor<8x3x3x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x8x3xf32>) -> tensor<8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3xf32>
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<8x8x8x3xf32>, tensor<3x3x3x3xf32>) -> tensor<8x6x6x3xf32>
    tf_device.return %conv : tensor<8x6x6x3xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x6x6x3xf32>)
  func.return
}

// -----

// Check that Conv3D with spatial partitioning using "VALID" padding produces
// begin and end halos on all spatial dimensions and all necessary slice ops.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x8x8x8x3xf32> {tf._layout = "sharding_specs:unsharded,x,y,z,unsharded, mesh:|x=2,y=2,z=2|*TPU"},
           %arg2:tensor<3x3x3x3x3xf32>) {
  // CHECK:         "tf_device.cluster"

  // Build left halo on depth dim.
  // CHECK:           %[[SLICE_D_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 3, 0, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_LEFT:.*]] = "tf.Slice"(%arg1, %[[SLICE_D_LEFT_BEGIN]], %[[SLICE_D_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x1x4x4x3xf32>
  // CHECK-NEXT:      %[[HALO_D_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_D_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 0], [5, 1], [6, 2], [7, 3]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_D_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_D_LEFT]], %[[PAIRS_D_LEFT]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x2xi32>) -> tensor<8x1x4x4x3xf32>
  // Build right halo on depth dim.
  // CHECK:           %[[SLICE_D_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 1, 4, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_D_RIGHT:.*]] = "tf.Slice"(%arg1, %[[SLICE_D_RIGHT_BEGIN]], %[[SLICE_D_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x4x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x1x4x4x3xf32>
  // CHECK-NEXT:      %[[HALO_D_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_D_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [4, 0], [5, 1], [6, 2], [7, 3]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_D_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_D_RIGHT]], %[[PAIRS_D_RIGHT]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x2xi32>) -> tensor<8x1x4x4x3xf32>
  // Concat the halos with the shard on the depth dim.
  // CHECK-NEXT:      %[[CONCAT_D_AXIS:.*]] = "tf.Const"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_D_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_D_LEFT]], %arg1, %[[EXCHANGED_HALO_D_RIGHT]], %[[CONCAT_D_AXIS]])
  // CHECK-SAME:          (tensor<8x1x4x4x3xf32>, tensor<8x4x4x4x3xf32>, tensor<8x1x4x4x3xf32>, tensor<i64>) -> tensor<8x6x4x4x3xf32>
  // Dynamically slice the concatenated tensor to get correct size for VALID padding.
  // CHECK-NEXT:      %[[HALO_SIZES_D:.*]] = "tf.Const"() <{value = dense<[0, 1, 0, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[HALO_INCREMENTS_D:.*]] = "tf.Const"() <{value = dense<[0, 1, 0, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[VALID_OFFSET_D:.*]] = "tf.Mul"
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_D:.*]] = "tf.Sub"(%[[HALO_SIZES_D]], %[[VALID_OFFSET_D]])
  // CHECK-NEXT:      %[[VALID_SLICE_SIZE_D:.*]] = "tf.Const"() <{value = dense<[8, 5, 4, 4, 3]> : tensor<5xi64>}> : () -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_CAST_I64_D:.*]] = "tf.Cast"(%[[VALID_SLICE_BEGIN_D]]) <{Truncate = false}> : (tensor<5xi32>) -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_D_TENSOR:.*]] = "tf.Slice"(%[[CONCAT_D_TENSOR]], %[[VALID_SLICE_BEGIN_CAST_I64_D]], %[[VALID_SLICE_SIZE_D]])

  // Build left halo on height dim.
  // CHECK:           %[[SLICE_H_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 3, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 1, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_LEFT:.*]] = "tf.Slice"(%[[VALID_SLICE_D_TENSOR]], %[[SLICE_H_LEFT_BEGIN]], %[[SLICE_H_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x5x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1], [4, 6], [5, 7], [6, 4], [7, 5]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_H_LEFT]], %[[PAIRS_H_LEFT]])
  // CHECK-SAME:          (tensor<8x5x1x4x3xf32>, tensor<8x2xi32>) -> tensor<8x5x1x4x3xf32>
  // Build right halo on height dim.
  // CHECK:           %[[SLICE_H_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 1, 4, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_H_RIGHT:.*]] = "tf.Slice"(%[[VALID_SLICE_D_TENSOR]], %[[SLICE_H_RIGHT_BEGIN]], %[[SLICE_H_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x4x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x5x1x4x3xf32>
  // CHECK-NEXT:      %[[HALO_H_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_H_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 2], [1, 3], [2, 0], [3, 1], [4, 6], [5, 7], [6, 4], [7, 5]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_H_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_H_RIGHT]], %[[PAIRS_H_RIGHT]])
  // CHECK-SAME:          (tensor<8x5x1x4x3xf32>, tensor<8x2xi32>) -> tensor<8x5x1x4x3xf32>
  // Concat the halos with the shard on the height dim.
  // CHECK-NEXT:      %[[CONCAT_H_AXIS:.*]] = "tf.Const"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_DH_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_H_LEFT]], %[[VALID_SLICE_D_TENSOR]], %[[EXCHANGED_HALO_H_RIGHT]], %[[CONCAT_H_AXIS]])
  // CHECK-SAME:          (tensor<8x5x1x4x3xf32>, tensor<8x5x4x4x3xf32>, tensor<8x5x1x4x3xf32>, tensor<i64>) -> tensor<8x5x6x4x3xf32>
  // Dynamically slice the concatenated tensor to get correct size for VALID padding.
  // CHECK-NEXT:      %[[HALO_SIZES_H:.*]] = "tf.Const"() <{value = dense<[0, 0, 1, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[HALO_INCREMENTS_H:.*]] = "tf.Const"() <{value = dense<[0, 0, 1, 0, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[VALID_OFFSET_H:.*]] = "tf.Mul"
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_H:.*]] = "tf.Sub"(%[[HALO_SIZES_H]], %[[VALID_OFFSET_H]])
  // CHECK-NEXT:      %[[VALID_SLICE_SIZE_H:.*]] = "tf.Const"() <{value = dense<[8, 5, 5, 4, 3]> : tensor<5xi64>}> : () -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_CAST_I64_H:.*]] = "tf.Cast"(%[[VALID_SLICE_BEGIN_H]]) <{Truncate = false}> : (tensor<5xi32>) -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_DH_TENSOR:.*]] = "tf.Slice"(%[[CONCAT_DH_TENSOR]], %[[VALID_SLICE_BEGIN_CAST_I64_H]], %[[VALID_SLICE_SIZE_H]])

  // Build left halo on width dim.
  // CHECK:           %[[SLICE_W_LEFT_BEGIN:.*]] = "tf.Const"() <{value = dense<[0, 0, 0, 3, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 5, 1, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_LEFT:.*]] = "tf.Slice"(%[[VALID_SLICE_DH_TENSOR]], %[[SLICE_W_LEFT_BEGIN]], %[[SLICE_W_LEFT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x5x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x5x5x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_LEFT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_LEFT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_LEFT:.*]] = "tf.CollectivePermute"(%[[HALO_W_LEFT]], %[[PAIRS_W_LEFT]])
  // CHECK-SAME:          (tensor<8x5x5x1x3xf32>, tensor<8x2xi32>) -> tensor<8x5x5x1x3xf32>
  // Build right halo on width dim.
  // CHECK:           %[[SLICE_W_RIGHT_BEGIN:.*]] = "tf.Const"() <{value = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT_SIZE:.*]] = "tf.Const"() <{value = dense<[8, 5, 5, 1, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[SLICE_W_RIGHT:.*]] = "tf.Slice"(%[[VALID_SLICE_DH_TENSOR]], %[[SLICE_W_RIGHT_BEGIN]], %[[SLICE_W_RIGHT_SIZE]])
  // CHECK-SAME:          (tensor<8x5x5x4x3xf32>, tensor<5xi32>, tensor<5xi32>) -> tensor<8x5x5x1x3xf32>
  // CHECK-NEXT:      %[[HALO_W_RIGHT:.*]] = "tf.SelectV2"
  // CHECK-NEXT:      %[[PAIRS_W_RIGHT:.*]] = "tf.Const"
  // CHECK-SAME{LITERAL}: value = dense<[[0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4], [6, 7], [7, 6]]>
  // CHECK-NEXT:      %[[EXCHANGED_HALO_W_RIGHT:.*]] = "tf.CollectivePermute"(%[[HALO_W_RIGHT]], %[[PAIRS_W_RIGHT]])
  // CHECK-SAME:          (tensor<8x5x5x1x3xf32>, tensor<8x2xi32>) -> tensor<8x5x5x1x3xf32>
  // Concat the halos with the shard on the width dim.
  // CHECK-NEXT:      %[[CONCAT_W_AXIS:.*]] = "tf.Const"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
  // CHECK-NEXT:      %[[CONCAT_DHW_TENSOR:.*]] = "tf.ConcatV2"(%[[EXCHANGED_HALO_W_LEFT]], %[[VALID_SLICE_DH_TENSOR]], %[[EXCHANGED_HALO_W_RIGHT]], %[[CONCAT_W_AXIS]])
  // CHECK-SAME:          (tensor<8x5x5x1x3xf32>, tensor<8x5x5x4x3xf32>, tensor<8x5x5x1x3xf32>, tensor<i64>) -> tensor<8x5x5x6x3xf32>
  // Dynamically slice the concatenated tensor to get correct size for VALID padding.
  // CHECK-NEXT:      %[[HALO_SIZES_W:.*]] = "tf.Const"() <{value = dense<[0, 0, 0, 1, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[HALO_INCREMENTS_W:.*]] = "tf.Const"() <{value = dense<[0, 0, 0, 1, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK-NEXT:      %[[VALID_OFFSET_W:.*]] = "tf.Mul"
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_W:.*]] = "tf.Sub"(%[[HALO_SIZES_W]], %[[VALID_OFFSET_W]])
  // CHECK-NEXT:      %[[VALID_SLICE_SIZE_W:.*]] = "tf.Const"() <{value = dense<[8, 5, 5, 5, 3]> : tensor<5xi64>}> : () -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_BEGIN_CAST_I64_W:.*]] = "tf.Cast"(%[[VALID_SLICE_BEGIN_W]]) <{Truncate = false}> : (tensor<5xi32>) -> tensor<5xi64>
  // CHECK-NEXT:      %[[VALID_SLICE_DHW_TENSOR:.*]] = "tf.Slice"(%[[CONCAT_DHW_TENSOR]], %[[VALID_SLICE_BEGIN_CAST_I64_W]], %[[VALID_SLICE_SIZE_W]])

  // CHECK-NEXT:      "tf.Conv3D"(%[[VALID_SLICE_DHW_TENSOR]], %arg2)
  // CHECK-SAME:          padding = "VALID"
  // CHECK-SAME:          (tensor<8x5x5x5x3xf32>, tensor<3x3x3x3x3xf32>) -> tensor<8x3x3x3x3xf32>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x8x8x3>, layout = #dtensor.layout<sharding_specs:unsharded,x,y,z,unsharded, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<8x8x8x8x3xf32>) -> tensor<8x8x8x8x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2,z=8|*TPU>} : (tensor<3x3x3x3x3xf32>) -> tensor<3x3x3x3x3xf32>
    %conv = "tf.Conv3D"(%img_layout, %filter_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1, 1]} : (tensor<8x8x8x8x3xf32>, tensor<3x3x3x3x3xf32>) -> tensor<8x6x6x6x3xf32>
    tf_device.return %conv : tensor<8x6x6x6x3xf32>
  }) {_mesh="|x=2,y=2,z=2|*TPU"} : () -> (tensor<8x6x6x6x3xf32>)
  func.return
}

// -----

// Check that Conv2DBackpropInputV2 becomes Conv2DBackpropInput in SPMD expansion
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
                %arg2: tensor<1x3x3x3xf32>,
                %arg3: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropInput"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK-SAME: (tensor<4xi32>, tensor<1x3x3x3xf32>, tensor<4x32x32x3xf32>) -> tensor<4x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropInputV2"(%input_layout, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv_layout : tensor<8x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv3DBackpropInput becomes Conv3DBackpropInputV2 in SPMD expansion
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
                %arg2: tensor<1x3x3x3x3xf32>,
                %arg3: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3DBackpropInputV2"
  // CHECK-SAME: data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: (tensor<5xi32>, tensor<1x3x3x3x3xf32>, tensor<4x32x32x32x3xf32>) -> tensor<4x32x32x32x3xf32>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropInput"(%input_layout, %filter_layout, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv_layout : tensor<8x32x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check that Conv2DBackpropFilterV2 becomes Conv2DBackpropFilter in SPMD expansion
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
                %arg2: tensor<1x3x3x3xf32>,
                %arg3: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropFilter"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK-SAME: (tensor<4x32x32x3xf32>, tensor<4xi32>, tensor<4x32x32x3xf32>) -> tensor<1x3x3x3xf32>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropFilterV2"(%input_layout, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<1x3x3x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    tf_device.return %conv_layout : tensor<1x3x3x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<1x3x3x3xf32>)
  func.return
}

// -----

// Check that Conv3DBackpropFilter becomes Conv3DBackpropFilterV2 in SPMD expansion
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"},
                %arg2: tensor<1x3x3x3x3xf32>,
                %arg3: tensor<8x32x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|*TPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3DBackpropFilterV2"
  // CHECK-SAME: data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: (tensor<4x32x32x32x3xf32>, tensor<5xi32>, tensor<4x32x32x32x3xf32>) -> tensor<1x3x3x3x3xf32>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropFilter"(%input_layout, %filter_layout, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<1x3x3x3x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<1x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    tf_device.return %conv_layout : tensor<1x3x3x3x3xf32>
  }) {_mesh="TPU|x=2,y=2|*TPU"}: () -> (tensor<1x3x3x3x3xf32>)
  func.return
}
