// RUN: dtensor-opt %s -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -split-input-file -verify-diagnostics | FileCheck %s

// Check that layouts for constant ops automatically set as replicated on mesh.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:        "tf_device.cluster"()
    // CHECK-NEXT:     %[[CONST_OUT:.*]] = "tf.Const"() <{value = dense<10> : tensor<i32>}> {_global_shape = [#tf_type.shape<>]}
    // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[CONST_OUT]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs: mesh:CPU|x=4,y=1|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
    // CHECK-NEXT:     %[[NEG_OUT:.*]] = "tf.Neg"(%[[DTENSOR_LAYOUT_OUT]])
    // CHECK-NEXT:     %[[NEG_LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[NEG_OUT]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs: mesh:CPU|x=4,y=1|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {_mesh="CPU|x=4,y=1|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check that conflicting consumer layouts are merged via setting replicated
// layout to conflicting tensor dimension.
// CHECK-LABEL: func @main
func.func @main() {
   // CHECK:        "tf_device.cluster"()
   // CHECK-NEXT:     %[[CONST_OUT_1:.*]] = "tf.Const"()
   // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT_1:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_1]])
   // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
   // CHECK-NEXT:     %[[CONST_OUT_2:.*]] = "tf.Const"()
   // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT_2:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_2]])
   // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
   // CHECK-NEXT:     %[[TILE_OUT:.*]] = "tf.Tile"(%[[DTENSOR_LAYOUT_OUT_1]], %[[DTENSOR_LAYOUT_OUT_2]])
   // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT_3:.*]] = "tf.DTensorLayout"(%[[TILE_OUT]])
   // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
   %6, %7 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<10.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    %2 = "tf.Const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Tile"(%1, %2) {device = ""} : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<4x4xf32>

    %4 = "tf.Neg"(%3) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %6 = "tf.Identity"(%3) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %7 = "tf.DTensorLayout"(%6) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %5, %7 : tensor<4x4xf32>, tensor<4x4xf32>
  }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<2x2xi32>, tensor<2x2xi32>)
  func.return
}

// -----

// Check that constants with multiple consumers are cloned.
// CHECK-LABEL: func @main
func.func @main() {
   %6, %7 = "tf_device.cluster"() ({
    // CHECK:        "tf_device.cluster"()
    // CHECK-NEXT:     %[[CONST_OUT_1:.*]] = "tf.Const"() <{value = dense<10> : tensor<2x2xi32>}> {_global_shape = [#tf_type.shape<2x2>]}
    // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_1]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
    // CHECK-NEXT:     %[[CONST_OUT_2:.*]] = "tf.Const"() <{value = dense<10> : tensor<2x2xi32>}> {_global_shape = [#tf_type.shape<2x2>]}
    // CHECK-NEXT:     %[[DTENSOR_LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_2]])
    // CHECK-SAME:     layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
    %1 = "tf.Const"() {value = dense<10> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tf.Neg"(%1) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<2x2xi32>) -> tensor<2x2xi32>

    %4 = "tf.Identity"(%1) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %3, %5 : tensor<2x2xi32>, tensor<2x2xi32>
  }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<2x2xi32>, tensor<2x2xi32>)
  func.return
}

// -----

// Check that layout propagation of strided slice op with shrink axis attributes.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<3xi32> {tf._global_shape = #tf_type.shape<3>},
  %arg2: tensor<1xi32> {tf._global_shape = #tf_type.shape<1>},
  %arg3: tensor<1xi32> {tf._global_shape = #tf_type.shape<1>},
  %arg4: tensor<1xi32> {tf._global_shape = #tf_type.shape<1>}) -> (tensor<i32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0,op_input_1,op_input_2,op_input_3", outputs = "op_output_0"}} {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:    %[[DTENSOR_LAYOUT_1:.*]] = "tf.DTensorLayout"(%[[CONST_OUT]])
  // CHECK-SAME:    layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>
  // CHECK-NEXT:    %[[CONST_OUT_2:.*]] = "tf.Const"
  // CHECK-NEXT:    %[[DTENSOR_LAYOUT_2:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_2]])
  // CHECK-SAME:    layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>
  // CHECK-NEXT:    %[[CONST_OUT_3:.*]] = "tf.Const"
  // CHECK-NEXT:    %[[DTENSOR_LAYOUT_3:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_3]])
  // CHECK-NEXT:    %[[CONST_OUT_4:.*]] = "tf.Const"
  // CHECK-NEXT:    %[[DTENSOR_LAYOUT_4:.*]] = "tf.DTensorLayout"(%[[CONST_OUT_4]])
  // CHECK-SAME:    layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>
  // CHECK-NEXT:    "tf.StridedSlice"(%[[DTENSOR_LAYOUT_4]], %[[DTENSOR_LAYOUT_3]], %[[DTENSOR_LAYOUT_1]], %[[DTENSOR_LAYOUT_2]])
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {_global_shape = [#tf_type.shape<1>], value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = "tf.Const"() {_global_shape = [#tf_type.shape<1>], value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %3 = "tf.Const"() {_global_shape = [#tf_type.shape<3>], value = dense<[8, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
    %8 = "tf.StridedSlice"(%3, %2, %1, %1) {_global_shape = [#tf_type.shape<>], begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %9 = "tf.DTensorLayout"(%8) {_global_shape = [#tf_type.shape<>], global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs:scalar CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<i32>) -> tensor<i32>
    tf_device.return {_global_shape = []} %9 : tensor<i32>
  }) {_global_shape = [#tf_type.shape<>], _mesh = "CPU|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// Check that two consecutive DTensorLayoutOps with different layout is
// disallowed.
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<2xi32> {tf._global_shape = #tf_type.shape<1>}) -> (tensor<2xi32> {tf._global_shape = #tf_type.shape<>}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) { global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    // expected-error @+1 {{Found inconsistent layout}}
    %2 = "tf.DTensorLayout"(%1) { global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    tf_device.return %2 : tensor<2xi32>
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// Check that two multiple DTensorLayoutOps with are replaced by identity op.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<2xi32> {tf._global_shape = #tf_type.shape<1>}) -> (tensor<2xi32> {tf._global_shape = #tf_type.shape<>}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    "tf.Identity"
  // CHECK-NEXT:    "tf.DTensorLayout"
  // CHECK-NEXT:    "tf.Add"
  // CHECK-NEXT:    "tf.DTensorLayout"
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) { global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %2 = "tf.DTensorLayout"(%1) { global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %3 = "tf.Add"(%2, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    tf_device.return %3 : tensor<2xi32>
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// Check that einsum will produce replicated layouts when there is a conflict
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<2x2xf32> {tf._global_shape = #tf_type.shape<2x2>},
  %arg2: tensor<2x2xf32> {tf._global_shape = #tf_type.shape<2x2>}) -> (tensor<2x2xf32> {tf._global_shape = #tf_type.shape<2x2>}) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.DTensorLayout"
  // CHECK-NEXT: "tf.DTensorLayout"
  // CHECK-NEXT: "tf.Einsum"
  // CHECK-NEXT: "tf.DTensorLayout"
  // CHECK-SAME: sharding_specs:unsharded,unsharded
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) { global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "tf.DTensorLayout"(%arg2) { global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:y,x, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = "tf.Einsum"(%1, %2) {equation="ab,bc->ac"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    tf_device.return %3 : tensor<2x2xf32>
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// Check that conv2d uses input image layout as output layout.
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x32x32x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"},
           %arg2:tensor<8x3x3x3xf32>) {
  // CHECK-LABEL: func @main
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2D"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1>
  %0 = "tf_device.cluster"() ({
    %img_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1>} : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1>} : (tensor<8x3x3x3xf32>) -> tensor<8x3x3x3xf32>
    %conv = "tf.Conv2D"(%img_layout, %filter_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<8x3x3x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check that conv2d backprop uses grads as output layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<1x3x3x3xf32>,
           %arg2: tensor<8x32x32x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropInput"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %img_shape = "tf.Const"() { value=dense<[8,32,32,3]> : tensor<4xi32>} : () -> tensor<4xi32>
    %filter_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropInput"(%img_shape, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<4xi32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check Conv2DBackpropFilter output is replicated.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %input_img: tensor<2x9x9x1xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %grad: tensor<2x9x9x2xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}
  ) -> tensor<2x2x1x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropFilter"
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %filter_shape = "tf.Const"() { value = dense<[2, 2, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %input_layout = "tf.DTensorLayout"(%input_img) {global_shape = #tf_type.shape<2x9x9x1>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<2x9x9x1xf32>) -> tensor<2x9x9x1xf32>
    %grad_layout = "tf.DTensorLayout"(%grad) {global_shape = #tf_type.shape<2x9x9x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<2x9x9x2xf32>) -> tensor<2x9x9x2xf32>
    %2 = "tf.Conv2DBackpropFilter"(%input_layout, %filter_shape, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<2x9x9x1xf32>, tensor<4xi32>, tensor<2x9x9x2xf32>) -> tensor<2x2x1x2xf32>
    tf_device.return %2 : tensor<2x2x1x2xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<2x2x1x2xf32>
  func.return %0 : tensor<2x2x1x2xf32>
}
// -----


// Check inserted DTensorLayout carries shape for resource tensor.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}) -> (tensor<!tf_type.resource<tensor<2x4xf32>>> {tf._default_layout = "empty_layout", tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id", outputs = "op_output_0"}} {
  // CHECK:      %[[VAR_HANDLE_OUT:.*]] = "tf.VarHandleOp"
  // CHECK:      "tf.DTensorLayout"(%[[VAR_HANDLE_OUT]])
  // CHECK-SAME: global_shape = #tf_type.shape<2x4>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.VarHandleOp"() {_global_shape = [#tf_type.shape<>], allowed_devices = [], container = "", device = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<2x4xf32>>>
    tf_device.return %1 : tensor<!tf_type.resource<tensor<2x4xf32>>>
  }) {_mesh = "empty_mesh"} : () -> tensor<!tf_type.resource<tensor<2x4xf32>>>
  func.return %0 : tensor<!tf_type.resource<tensor<2x4xf32>>>
}

// -----

// Check that Relayout ops are properly inserted for WhileRegions if needed.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>  {},
           %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}) -> (tensor<4xf32>) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>}  : (tensor<i32>) -> tensor<i32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>}  : (tensor<4xf32>) -> tensor<4xf32>
    %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %6 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %8 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
    %12 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %16:2 = "tf.WhileRegion"(%4, %2) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
      %27 = "tf.Less"(%arg2, %6) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%27) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
      %27 = "tf.StatelessRandomNormal"(%10, %12) : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xf32>
      // CHECK:      "tf.AddV2"
      // CHECK-NEXT: "tf.AddV2"
      // CHECK-NEXT: "tf.DTensorLayout"
      // CHECK-NEXT: "tf.DTensorLayout"
      // CHECK-NEXT: "tf.Relayout"
      // CHECK-NEXT: "tf.DTensorLayout"
      // CHECK-NEXT: "tf.Yield"
      %33 = "tf.AddV2"(%arg3, %27) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %37 = "tf.AddV2"(%arg2, %8) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %40 = "tf.DTensorLayout"(%33) {global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4xf32>) -> tensor<4xf32>
      "tf.Yield"(%37, %40) : (tensor<i32>, tensor<4xf32>) -> ()
    }) {is_stateless = true, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
    // CHECK:      "tf.DTensorLayout"
    // CHECK-NEXT: "tf.Relayout"
    // CHECK-NEXT: "tf.DTensorLayout"
    // CHECK-NEXT: "tf.DTensorLayout"
    tf_device.return %16#1 : tensor<4xf32>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<4xf32>)
  func.return %0 : tensor<4xf32>
}

// -----

// Check that no Relayout ops are properly inserted for WhileRegions when not
// needed
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>  {},
           %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}) -> (tensor<4xf32>) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>}  : (tensor<i32>) -> tensor<i32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>}  : (tensor<4xf32>) -> tensor<4xf32>
    %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %6 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %8 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
    %12 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %16:2 = "tf.WhileRegion"(%4, %2) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
      %27 = "tf.Less"(%arg2, %6) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%27) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
      %27 = "tf.StatelessRandomNormal"(%10, %12) : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xf32>
      // CHECK:      "tf.AddV2"
      // CHECK-NEXT: "tf.DTensorLayout"
      // CHECK-NEXT: "tf.AddV2"
      // CHECK-NEXT: "tf.DTensorLayout"
      // CHECK-NEXT: "tf.Yield"
      %33 = "tf.AddV2"(%arg3, %27) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %37 = "tf.AddV2"(%arg2, %8) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%37, %33) : (tensor<i32>, tensor<4xf32>) -> ()
    }) {is_stateless = true, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
    // CHECK: "tf.DTensorLayout"
    // CHECK-NEXT: "tf.DTensorLayout"
    // CHECK-NEXT: tf_device.return
    tf_device.return %16#1 : tensor<4xf32>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<4xf32>)
  func.return %0 : tensor<4xf32>
}

// -----

// Check that RelayoutLike propagates the original Relayout input's layout to
// the output gradient.

// CHECK-LABEL: func @main
func.func @main(
    %arg0: tensor<i32> {},
    %arg1: tensor<8x8xf32> {
      tf._global_shape = #tf_type.shape<8x8>,
      tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    }) -> (tensor<8x8xf32> {tf._global_shape = #tf_type.shape<8x8>}) {
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<8x8>], value = dense<1.000000e+00> : tensor<8x8xf32>} : () -> tensor<8x8xf32>
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK:        %[[RELAYOUT_OUT:.*]] = "tf.Relayout"
    // CHECK-NEXT:   "tf.DTensorLayout"(%[[RELAYOUT_OUT]])
    // CHECK-SAME:       layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>
    %2 = "tf.Relayout"(%1) {global_shape = #tf_type.shape<8x8>, layout = "sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = "tf.Identity"(%cst) {_global_shape = [#tf_type.shape<8x8>]} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = "tf.AddN"(%2, %3) {_global_shape = [#tf_type.shape<8x8>]} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    // CHECK:        %[[RELAYOUT_GRAD_OUT:.*]] = "tf.RelayoutLike"
    // CHECK-NEXT:   "tf.DTensorLayout"(%[[RELAYOUT_GRAD_OUT]])
    // CHECK-SAME:       layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>
    %5 = "tf.RelayoutLike"(%4, %1) {_global_shape = [#tf_type.shape<8x8>]} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    tf_device.return %5 : tensor<8x8xf32>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// Check that the contracted dimension of the reduce op is set to any.
//
// We verify that this is correct because tf.Const's sharding specs were set to
// specs:x,y. Without the "kAny" flag it would have been set up to
// specs:x,unsharded due to the conflict between the consumers tf.Sum and tf.Neg
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:      %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[CONST_OUT]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %val = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %identity = "tf.Identity"(%val) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %negval = "tf.Neg"(%identity) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %valuelayout = "tf.DTensorLayout"(%negval) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %dimension = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
    %sum = "tf.Sum"(%identity, %dimension) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    %sumlayout = "tf.DTensorLayout"(%sum) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x1xi32>) -> tensor<2x1xi32>
    tf_device.return %sumlayout : tensor<2x1xi32>
  }) {_mesh = "|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check that layouts are propagated to then/else branches of the IfRegion op.
// CHECK-LABEL: func @main
func.func @main() {
  %0 = "tf_device.cluster"() ({
    // CHECK:      %[[IF_OUT:.*]]:2 = "tf.IfRegion"
    // CHECK-NEXT:   "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,y
    // CHECK-NEXT:   "tf.Sqrt"
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,y
    // CHECK:        "tf.Yield"
    // CHECK:        "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,y
    // CHECK:        "tf.Yield"
    // CHECK-NEXT: (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)
    // CHECK:      "tf.DTensorLayout"(%[[IF_OUT]]#1)
    // CHECK-SAME: sharding_specs:unsharded,y
    %predicate= "tf.Const"() { value = dense<0> :  tensor<i1>}: () ->  tensor<i1>
    %1:2 = "tf.IfRegion"(%predicate) ({
        %2 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %3 = "tf.Sqrt"(%2) {_global_shape = [#tf_type.shape<4x4>], device = ""} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %4 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %5 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%5, %4) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      },  {
        %6 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %7 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%7, %6) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      }) {_else_func_name = "cond_false_140", _global_shape = [#tf_type.shape<>, #tf_type.shape<4x4>], _lower_using_switch_merge = true, _read_only_resource_inputs = [], _then_func_name = "cond_true_130", device = "", is_stateless = true} : (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)

    tf_device.return %1#1: tensor<4x4xf64>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check that if conflicting layouts for outputs of IfRegion exists, then
// replicated layout is used.
// CHECK-LABEL: func @main
func.func @main() {
  %0 = "tf_device.cluster"() ({
    %predicate= "tf.Const"() { value = dense<0> :  tensor<i1>}: () ->  tensor<i1>
    // CHECK:      %[[IF_OUT:.*]]:2 = "tf.IfRegion"
    // CHECK-NEXT:   "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,y
    // CHECK-NEXT:   "tf.Sqrt"
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,unsharded
    // CHECK:        "tf.Yield"
    // CHECK:        "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,unsharded
    // CHECK:        "tf.Yield"
    // CHECK-NEXT: (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)
    // CHECK:      "tf.DTensorLayout"(%[[IF_OUT]]#1)
    // CHECK-SAME: sharding_specs:unsharded,unsharded
    %1:2 = "tf.IfRegion"(%predicate) ({
        %2 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %3 = "tf.Sqrt"(%2) {_global_shape = [#tf_type.shape<4x4>], device = ""} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %4 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %5 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%5, %4) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      },  {

        %6 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %8 = "tf.DTensorLayout"(%6) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %7 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%7, %8) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      }) {_else_func_name = "cond_false_140", _global_shape = [#tf_type.shape<>, #tf_type.shape<4x4>], _lower_using_switch_merge = true, _read_only_resource_inputs = [], _then_func_name = "cond_true_130", device = "", is_stateless = true} : (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)

    tf_device.return %1#1: tensor<4x4xf64>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check that if duplicate layouts for outputs of IfRegion exists, then
// replicated layout is used.
// CHECK-LABEL: func @main
func.func @main() {
  %0 = "tf_device.cluster"() ({
    %predicate= "tf.Const"() { value = dense<0> :  tensor<i1>}: () ->  tensor<i1>
    // CHECK:      %[[IF_OUT:.*]]:2 = "tf.IfRegion"
    // CHECK-NEXT:   "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,y
    // CHECK-NEXT:   "tf.Sqrt"
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,unsharded
    // CHECK:        "tf.Yield"
    // CHECK:        "tf.Const"()
    // CHECK-NEXT:   "tf.DTensorLayout"
    // CHECK-SAME:   sharding_specs:unsharded,unsharded
    // CHECK:        "tf.Yield"
    // CHECK-NEXT: (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)
    // CHECK:      "tf.DTensorLayout"(%[[IF_OUT]]#1)
    // CHECK-SAME: sharding_specs:unsharded,unsharded
    %1:2 = "tf.IfRegion"(%predicate) ({
        %2 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %3 = "tf.Sqrt"(%2) {_global_shape = [#tf_type.shape<4x4>], device = ""} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %4 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %5 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%5, %4) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      },  {
        %6 = "tf.Const"() { value = dense<0.0> : tensor<4x4xf64>}: () ->  tensor<4x4xf64>
        %8 = "tf.DTensorLayout"(%6) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x4xf64>) -> tensor<4x4xf64>
        %7 = "tf.Const"() { value = dense<0> : tensor<i1>}: () ->  tensor<i1>
        "tf.Yield"(%7, %8) {_global_shape = []} : (tensor<i1>, tensor<4x4xf64>) -> ()
      }) {_else_func_name = "cond_false_140", _global_shape = [#tf_type.shape<>, #tf_type.shape<4x4>], _lower_using_switch_merge = true, _read_only_resource_inputs = [], _then_func_name = "cond_true_130", device = "", is_stateless = true} : (tensor<i1>) -> (tensor<i1>, tensor<4x4xf64>)

    tf_device.return %1#1: tensor<4x4xf64>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check that the contracted dimension of the Matmul op is set to any.
//
// We verify that this is correct because tf.Const's sharding specs were set to
// specs:x,y. Without the "kAny" flag it would have been set up to
// specs:x,unsharded due to the conflict between the consumers tf.Matmul and
// tf.Neg.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[ID_OPERAND_2:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_2]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:unsharded,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[NEG:.*]] = "tf.Neg"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[NEG]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[MATMUL:.*]] = "tf.MatMul"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[MATMUL]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  %0 = "tf_device.cluster"() ({
    %operand_1 = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %operand_2 = "tf.Const"() { value = dense<[[1, 2],[3, 4], [5, 6], [7, 8], [9, 10], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]> : tensor<10x2xi32>}: () -> tensor<10x2xi32>
    %id_operand_1 = "tf.Identity"(%operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %id_operand_2 = "tf.Identity"(%operand_2) : (tensor<10x2xi32>) -> tensor<10x2xi32>
    %random_consumer_op = "tf.Neg"(%id_operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %random_consumer_op_layout = "tf.DTensorLayout"(%random_consumer_op) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %matmul = "tf.MatMul"(%id_operand_1, %id_operand_2) {transpose_a = false, transpose_b = false}: (tensor<2x10xi32>, tensor<10x2xi32>) -> tensor<2x2xi32>
    %matmul_layout = "tf.DTensorLayout"(%matmul) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %matmul_layout : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check that einsum will propagate sharding to its inputs.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[ID_OPERAND_2:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_2]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:unsharded,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[EINSUM:.*]] = "tf.Einsum"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[EINSUM]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  %0 = "tf_device.cluster"() ({
    %operand_1 = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %operand_2 = "tf.Const"() { value = dense<[[1, 2],[3, 4], [5, 6], [7, 8], [9, 10], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]> : tensor<10x2xi32>}: () -> tensor<10x2xi32>
    %id_operand_1 = "tf.Identity"(%operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %id_operand_2 = "tf.Identity"(%operand_2) : (tensor<10x2xi32>) -> tensor<10x2xi32>
    %einsum = "tf.Einsum"(%id_operand_1, %id_operand_2) {equation="ab,bc->ac"} : (tensor<2x10xi32>, tensor<10x2xi32>) -> tensor<2x2xi32>
    %einsum_layout = "tf.DTensorLayout"(%einsum) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %einsum_layout : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> tensor<2x2xi32>
  func.return
}

// -----

// Check that einsum will propagate the "any" flag to its inputs.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[ID_OPERAND_2:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_2]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:unsharded,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[NEG:.*]] = "tf.Neg"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[NEG]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[EINSUM:.*]] = "tf.Einsum"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[EINSUM]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  %0 = "tf_device.cluster"() ({
    %operand_1 = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %operand_2 = "tf.Const"() { value = dense<[[1, 2],[3, 4], [5, 6], [7, 8], [9, 10], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]> : tensor<10x2xi32>}: () -> tensor<10x2xi32>
    %id_operand_1 = "tf.Identity"(%operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %id_operand_2 = "tf.Identity"(%operand_2) : (tensor<10x2xi32>) -> tensor<10x2xi32>
    %random_consumer_op = "tf.Neg"(%id_operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %random_consumer_op_layout = "tf.DTensorLayout"(%random_consumer_op) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %einsum = "tf.Einsum"(%id_operand_1, %id_operand_2) {equation="ab,bc->ac"} : (tensor<2x10xi32>, tensor<10x2xi32>) -> tensor<2x2xi32>
    %einsum_layout = "tf.DTensorLayout"(%einsum) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %einsum_layout : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> tensor<2x2xi32>
  func.return
}

// -----

// Check that einsum will propagate the "any" flag to its inputs, with
// incompatible sharding.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[ID_OPERAND_2:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_2]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:unsharded,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[NEG:.*]] = "tf.Neg"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[NEG]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  // CHECK-NEXT:    %[[EINSUM:.*]] = "tf.Einsum"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[EINSUM]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|
  %0 = "tf_device.cluster"() ({
    %operand_1 = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %operand_2 = "tf.Const"() { value = dense<[[1, 2],[3, 4], [5, 6], [7, 8], [9, 10], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]> : tensor<10x2xi32>}: () -> tensor<10x2xi32>
    %id_operand_1 = "tf.Identity"(%operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %id_operand_2 = "tf.Identity"(%operand_2) : (tensor<10x2xi32>) -> tensor<10x2xi32>
    %random_consumer_op = "tf.Neg"(%id_operand_1) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %random_consumer_op_layout = "tf.DTensorLayout"(%random_consumer_op) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %einsum = "tf.Einsum"(%id_operand_1, %id_operand_2) {equation="ab,bc->ac"} : (tensor<2x10xi32>, tensor<10x2xi32>) -> tensor<2x2xi32>
    %einsum_layout = "tf.DTensorLayout"(%einsum) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,z, mesh:|x=2,y=2,z=2|*TPU>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %einsum_layout : tensor<2x2xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> tensor<2x2xi32>
  func.return
}

// -----

// Check resolution of conflicting consumer layouts when proposed specs are
// set to Any.
//
// We use the sum ops to generate "any" specs. These specs are passed to the
// layout propagation algorithm as consumer_layouts.
//
// In this test we have two consumer ops:
//   - Consumer_1 specs:any,x,
//   - Consumer_2 specs:x,any,
//
// The test verifies that the final propagated layout is:
//   - Proposed specs:unsharded,unsharded,
//
// avoiding the illegal scenario of specs:x,x,
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:unsharded,unsharded,  mesh:|x=2,y=2|
  %0 = "tf_device.cluster"() ({
    %val = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %identity = "tf.Identity"(%val) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %dimension_0 = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
    %sum_0 = "tf.Sum"(%identity, %dimension_0) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<1x10xi32>
    %sumlayout_0 = "tf.DTensorLayout"(%sum_0) {global_shape = #tf_type.shape<10>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1x10xi32>) -> tensor<1x10xi32>
    %dimension_1 = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
    %sum_1 = "tf.Sum"(%identity, %dimension_1) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    %sumlayout_1 = "tf.DTensorLayout"(%sum_1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x1xi32>) -> tensor<2x1xi32>
    tf_device.return %sumlayout_1 : tensor<2x1xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// This test verifies that when a consumer op is set to any, the proposed specs
// are not modified.
//
// We have two consumer ops:
//
//   - Consumer_1 specs:x,unsharded,
//   - Consumer_2 specs:any,unsharded,
//
// The test verifies that the final propagated layout is:
//   - Proposed specs:x,unsharded,
//
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|
  %0 = "tf_device.cluster"() ({
    %val = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %val_layout = "tf.DTensorLayout"(%val) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %identity = "tf.Identity"(%val_layout) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %consumer_op_1 = "tf.Neg"(%identity) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %consumer_op_1_layout = "tf.DTensorLayout"(%consumer_op_1) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %dimension_0 = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
    %consumer_op_2 = "tf.Sum"(%identity, %dimension_0) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<1x10xi32>
    tf_device.return %consumer_op_2 : tensor<1x10xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> (tensor<1x10xi32>)
  func.return
}

// -----

// This test verifies that when the proposed spec is set to any, and the
// consumer op is proposing a specific sharding, the latter is taken.
//
// We use the sum ops to generate "any" specs. These specs are passed to the
// layout propagation algorithm as consumer_layouts.
//
// In this test we have two consumer ops:
//
//   - Consumer_1 specs:any,unsharded,
//   - Consumer_2 specs:x,unsharded,
//
// The test verifies that the final propagated layout is:
//   - Proposed specs:x,unsharded,
//
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:         %[[ID_OPERAND_1:.*]] = "tf.Identity"
  // CHECK-NEXT:      "tf.DTensorLayout"(%[[ID_OPERAND_1]])
  // CHECK-SAME:      layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|
  %0 = "tf_device.cluster"() ({
    %val = "tf.Const"() { value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>}: () -> tensor<2x10xi32>
    %val_layout = "tf.DTensorLayout"(%val) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %identity = "tf.Identity"(%val_layout) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %dimension_0 = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
    %sum_0 = "tf.Sum"(%identity, %dimension_0) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<1x10xi32>
    %consumer_op_1 = "tf.Neg"(%identity) : (tensor<2x10xi32>) -> tensor<2x10xi32>
    %consumer_op_1_layout = "tf.DTensorLayout"(%consumer_op_1) {global_shape = #tf_type.shape<2x10>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x10xi32>) -> tensor<2x10xi32>
    tf_device.return %consumer_op_1_layout : tensor<2x10xi32>
  }) {_mesh = "|x=2,y=2,z=2|*TPU"} : () -> (tensor<2x10xi32>)
  func.return
}

// -----
// This test verifies that a graph intentionally made to trigger an
// infinite loop in the layout-propagation-v2 algorithm, should still converge
// and produce a layout for all ops in the graph.
//
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
  %arg1: tensor<16x16xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %arg2: tensor<16x16xf32> {tf._layout = "sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}
  ) -> tensor<16x16xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.Identity"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.MatMul"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.Identity"
  // CHECK: "tf.DTensorLayout"
  // CHECK: "tf.Relayout"
  // CHECK: "tf.DTensorLayout"
  %1 = "tf_device.cluster"() ( {
    %arg1_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<16x16>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %arg2_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<16x16>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = "tf.Identity"(%arg1_layout) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %3 = "tf.MatMul"(%arg1_layout, %2) {transpose_a = false, transpose_b = false}: (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = "tf.Identity"(%3) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    %5 = "tf.Relayout"(%4) {layout = "sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: (tensor<16x16xf32>) -> tensor<16x16xf32>
    %6 = "tf.Identity"(%5) : (tensor<16x16xf32>) -> tensor<16x16xf32>
    tf_device.return %6 : tensor<16x16xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<16x16xf32>
  func.return %1 : tensor<16x16xf32>
}

// -----

// Check the tf.RestoreV2Op's output layout is correctly inferred for single
// mesh cluster. The output layout should be the layout of the
// AssignVariable's resource layout on a 1:1 CPU mesh.
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<!tf_type.string>,
  %arg2: tensor<1x!tf_type.string>,
  %arg3: tensor<1x!tf_type.string>,
  %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>> {
    tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {

    // CHECK: "tf_device.cluster"
    // CHECK-NEXT: "tf.RestoreV2"
    // CHECK-NEXT: "tf.DTensorLayout"
    // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>
    "tf_device.cluster"() ({
      %6 = "tf.RestoreV2"(%arg1, %arg2, %arg3) {} : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> (tensor<4x8xf32>)
      "tf.AssignVariableOp"(%arg4, %6) {validate_shape = false} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
    func.return
}

// -----

// Check the tf.RestoreV2Op's output layout is correctly inferred for multi
// mesh cluster function. The output layout should be the layout of the
// AssignVariable's resource layout changed to the 1:1 CPU mesh.
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<!tf_type.string>,
  %arg2: tensor<1x!tf_type.string>,
  %arg3: tensor<1x!tf_type.string>,
  %arg4: tensor<*x!tf_type.resource<tensor<4x8xf32>>> {
    tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) {
    // CHECK: "tf_device.cluster"
    "tf_device.cluster"() ({
      %1 = "tf.DTensorRecv"() {key = "communication_key_|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", mesh = #dtensor.mesh<|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>, shape = #tf_type.shape<4x8>} : () -> tensor<4x8xf32>
      %2 = "tf.Relayout"(%1) {global_shape = #tf_type.shape<8x8>, layout = "sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
      %3 = "tf.Identity"(%2) : (tensor<4x8xf32>) -> tensor<4x8xf32>
      "tf.AssignVariableOp"(%arg4, %3) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<4x8xf32>>>, tensor<4x8xf32>) -> ()
      tf_device.return
    }) {_mesh="TPU|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : () -> (tensor<i32>, tensor<i32>)

    // CHECK: "tf_device.cluster"
    // CHECK-NEXT: "tf.RestoreV2"
    // CHECK-NEXT: "tf.DTensorLayout"
    // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>
    "tf_device.cluster"() ({
      %6 = "tf.RestoreV2"(%arg1, %arg2, %arg3) {} : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> (tensor<4x8xf32>)
      "tf.DTensorSend"(%6) {key = "communication_key_|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", target_mesh = #dtensor.mesh<|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<4x8xf32>) -> ()
      tf_device.return
    }) {_mesh="CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check Conv2DBackpropInputV2 forwards input layout to output layout.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32>,
                %arg2: tensor<1x3x3x3xf32>,
                %arg3: tensor<8x32x32x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropInputV2"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropInputV2"(%input_layout, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check Conv2DBackpropInputV2 forwards output layout to input layouts.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32>,
                %arg2: tensor<1x3x3x3xf32>,
                %arg3: tensor<8x32x32x3xf32>) {
  // CHECK:      %[[INPUT:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[INPUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[FILTER:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[FILTER]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[GRAD:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[GRAD]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %input_identity = "tf.Identity"(%input_layout) : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_identity = "tf.Identity"(%filter_layout) : (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_identity = "tf.Identity"(%grad_layout) : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropInputV2"(%input_identity, %filter_identity, %grad_identity) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv_layout : tensor<8x32x32x3xf32>
  }) {_mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check Conv3DBackpropInput forwards input layout to output layout.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32>,
                %arg2: tensor<1x3x3x3x3xf32>,
                %arg3: tensor<8x32x32x32x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3DBackpropInput"
  // CHECK-SAME: dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: data_format = "NDHWC"
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropInput"(%input_layout, %filter_layout, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check Conv3DBackpropInput forwards output layout to input layouts.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32>,
                %arg2: tensor<1x3x3x3x3xf32>,
                %arg3: tensor<8x32x32x32x3xf32>) {
  // CHECK:      %[[INPUT:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[INPUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[FILTER:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[FILTER]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[GRAD:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[GRAD]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %input_identity = "tf.Identity"(%input_layout) : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_identity = "tf.Identity"(%filter_layout) : (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_identity = "tf.Identity"(%grad_layout) : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropInput"(%input_identity, %filter_identity, %grad_identity) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv_layout : tensor<8x32x32x32x3xf32>
  }) {_mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check Conv2DBackpropFilterV2 forwards filter layout to output layout.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32>,
                %arg2: tensor<1x3x3x3xf32>,
                %arg3: tensor<8x32x32x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv2DBackpropFilterV2"
  // CHECK-SAME: data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropFilterV2"(%input_layout, %filter_layout, %grad_layout) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x3xf32>, tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check Conv2DBackpropFilterV2 forwards output layout to filter layouts.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x3xf32>,
                %arg2: tensor<1x3x3x4xf32>,
                %arg3: tensor<8x32x32x3xf32>) {
  // CHECK:      %[[INPUT:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[INPUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[FILTER:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[FILTER]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[GRAD:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[GRAD]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %input_identity = "tf.Identity"(%input_layout) : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %filter_identity = "tf.Identity"(%filter_layout) : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    %grad_identity = "tf.Identity"(%grad_layout) : (tensor<8x32x32x3xf32>) -> tensor<8x32x32x3xf32>
    %conv = "tf.Conv2DBackpropFilterV2"(%input_identity, %filter_identity, %grad_identity) {data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} :
      (tensor<8x32x32x3xf32>, tensor<1x3x3x4xf32>, tensor<8x32x32x3xf32>) -> tensor<1x3x3x4xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<1x3x3x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
    tf_device.return %conv_layout : tensor<1x3x3x4xf32>
  }) {_mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x3xf32>)
  func.return
}

// -----

// Check Conv3DBackpropFilter forwards input layout to output layout.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32>,
                %arg2: tensor<1x3x3x3x3xf32>,
                %arg3: tensor<8x32x32x32x3xf32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[CONV_OUT:.*]] = "tf.Conv3DBackpropFilter"
  // CHECK-SAME: dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]
  // CHECK-SAME: data_format = "NDHWC"
  // CHECK:      "tf.DTensorLayout"(%[[CONV_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3x3xf32>) -> tensor<1x3x3x3x3xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropFilter"(%input_layout, %filter_layout, %grad_layout) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x3xf32>, tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    tf_device.return %conv : tensor<8x32x32x32x3xf32>
  }) {_mesh="TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}

// -----

// Check Conv3DBackpropFilter forwards output layout to input layouts.
func.func @main(%arg0: tensor<1xi32>,
                %arg1: tensor<8x32x32x32x3xf32>,
                %arg2: tensor<1x3x3x3x4xf32>,
                %arg3: tensor<8x32x32x32x3xf32>) {
  // CHECK:      %[[INPUT:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[INPUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[FILTER:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[FILTER]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK:      %[[GRAD:.*]] = "tf.Identity"
  // CHECK:      "tf.DTensorLayout"(%[[GRAD]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  %0 = "tf_device.cluster"() ({
    %input_layout = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_layout = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<1x3x3x3x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3x4xf32>) -> tensor<1x3x3x3x4xf32>
    %grad_layout = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %input_identity = "tf.Identity"(%input_layout) : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %filter_identity = "tf.Identity"(%filter_layout) : (tensor<1x3x3x3x4xf32>) -> tensor<1x3x3x3x4xf32>
    %grad_identity = "tf.Identity"(%grad_layout) : (tensor<8x32x32x32x3xf32>) -> tensor<8x32x32x32x3xf32>
    %conv = "tf.Conv3DBackpropFilter"(%input_identity, %filter_identity, %grad_identity) {data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} :
      (tensor<8x32x32x32x3xf32>, tensor<1x3x3x3x4xf32>, tensor<8x32x32x32x3xf32>) -> tensor<1x3x3x3x4xf32>
    %conv_layout = "tf.DTensorLayout"(%conv) {global_shape = #tf_type.shape<8x32x32x32x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} :
      (tensor<1x3x3x3x4xf32>) -> tensor<1x3x3x3x4xf32>
    tf_device.return %conv_layout : tensor<1x3x3x3x4xf32>
  }) {_mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}: () -> (tensor<8x32x32x32x3xf32>)
  func.return
}
