// RUN: dtensor-opt %s -split-input-file -dtensor-lower-send-recv -verify-diagnostics | FileCheck %s

// Check that Data transfer from CPU to TPU is lowered correctly.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[DEVICE_ID:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Identity"
  // CHECK-NEXT:   %[[RECV_ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[RECV_SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[RECV_DEVICE_ID:.*]] = "tf.Reshape"(%[[DEVICE_ID]], %[[RECV_SIZE_TYPE]])
  // CHECK-NEXT:   %[[RECV_SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[RECV_DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[RECV_ID_TO_ORDINAL]], %[[RECV_DEVICE_ID]], %[[RECV_SLICE_SIZE]])
  // CHECK-NEXT:   %[[RECV_SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[RECV_DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[RECV_DEVICE_ORDINAL]], %[[RECV_SCALAR_TYPE]])
  // CHECK-NEXT:   %[[RECV_DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[RECV_DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[LAYOUT_OUT:.*]] = "tf.DTensorLayout"
  // CHECK-NEXT:   %[[SEND_ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[SEND_SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[SEND_DEVICE_ID:.*]] = "tf.Reshape"(%[[DEVICE_ID]], %[[SEND_SIZE_TYPE]])
  // CHECK-NEXT:   %[[SEND_SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[SEND_DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[SEND_ID_TO_ORDINAL]], %[[SEND_DEVICE_ID]], %[[SEND_SLICE_SIZE]])
  // CHECK-NEXT:   %[[SEND_SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[SEND_DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[SEND_DEVICE_ORDINAL]], %[[SEND_SCALAR_TYPE]])
  // CHECK-NEXT:   %[[SEND_DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[SEND_DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   "tf._XlaSendFromHostV2"(%[[LAYOUT_OUT]], %[[PROGRAM_KEY]], %[[SEND_DEVICE_ORDINAL_SCALAR_64]])
  // CHECK-NEXT:   %[[RECV_OUT:.*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_KEY]], %[[RECV_DEVICE_ORDINAL_SCALAR_64]])
  // CHECK-SAME:   key = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_2"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Identity"
  // CHECK-NEXT:   %[[TPU_RECV_OUT:.*]] = "tf.XlaRecvFromHost"()
  // CHECK-SAME:   key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
  // CHECK-NEXT:   %[[TPU_LAYOUT_OUT:.*]] = "tf.DTensorLayout"(%[[TPU_RECV_OUT]])
  // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
  // CHECK-NEXT:   "tf.XlaSendToHost"(%[[A_OUT]])
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> tensor<1xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> ()

    %2 = "tf.DTensorRecv"() {key = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_2", layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>, shape = #tf_type.shape<>} : () -> (tensor<1xi32>)
    "tf.B"(%2) : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()
  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, shape = #tf_type.shape<>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> tensor<1xi32>
    %2 = "tf.A"(%1) : (tensor<1xi32>) -> tensor<1xi32>
    "tf.DTensorSend"(%2) {key = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_2", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
  func.return
}

// -----

// Check that device id usages are added correctly.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[DEVICE_ID:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) -> tensor<1xi32> {
  // CHECK:       "tf_device.cluster"()
  // CHECK-NEXT:    "tf.Identity"(%[[DEVICE_ID]])
  // CHECK-NEXT:    "tf.XlaRecvFromHost"
  // CHECK-NEXT:    tf_device.return
  // CHECK-NEXT:  _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  %2 = "tf_device.cluster"() ({
    %0 = "tf.XlaRecvFromHost"() {_layout = ["sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"], key = "communication_key_0", shape = #tf_type.shape<1>} : () -> tensor<1xi32>
    tf_device.return %0 : tensor<1xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<1xi32>

  func.return %2 : tensor<1xi32>
}

