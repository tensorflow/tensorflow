// RUN: dtensor-opt %s -split-input-file -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check that Data transfer from CPU to TPU is lowered correctly.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
  // CHECK-NEXT:   %[[SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   %[[ZERO:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64]], %[[ZERO]])
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE]])
  // CHECK-NEXT:     %[[ZERO_2:.*]] = "tf.Const"
  // CHECK-SAME:     value = dense<0>
  // CHECK-NEXT:     "tf._XlaSendFromHostV2"(%[[CONST_OUT]], %[[PROGRAM_KEY]], %[[ZERO_2]])
  // CHECK-SAME:     key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<[0, 1, 2, 3]>
  // CHECK-NEXT:   %[[SIZE_TYPE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID_2:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE_2]])
  // CHECK-NEXT:   %[[SLICE_SIZE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_2:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL_2]], %[[DEVICE_ID_2]], %[[SLICE_SIZE_2]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_2:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL_2]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64_2:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR_2]])
  // CHECK-NEXT:   %[[ZERO_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE_2:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64_2]], %[[ZERO_2]])
  // CHECK-NEXT:   %[[IF_OUT:.*]] = "tf.IfRegion"(%[[PREDICATE_2]])
  // CHECK-NEXT:     %[[RECV_OUT:.*]] = "tf.XlaRecvFromHost"()
  // CHECK-SAME:      key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"
  // CHECK-NEXT:     "tf.Yield"(%[[RECV_OUT]])
  // CHECK:          %[[ZEROS_3:.*]] = "tf.Const"
  // CHECK-NEXT:     "tf.Yield"(%[[ZEROS_3]])
  // CHECK:       %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[OUTPUT:.*]] = "tf.DTensorAllReduce"(%[[IF_OUT]], %[[GROUP_ASSIGNMENT]])
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> tensor<1xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()
  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, shape = #tf_type.shape<>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> tensor<1xi32>
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
  func.return
}


// -----

// Check that Data transfer from TPU to CPU is lowered correctly.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<[0, 1, 2, 3]>
  // CHECK-NEXT:   %[[SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
  // CHECK-NEXT:   %[[SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   %[[ZERO:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64]], %[[ZERO]])
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE]])
  // CHECK-NEXT:     "tf.XlaSendToHost"(%[[CONST_OUT]])
  // CHECK-SAME:     key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[SIZE_TYPE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID_2:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE_2]])
  // CHECK-NEXT:   %[[SLICE_SIZE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_2:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL_2]], %[[DEVICE_ID_2]], %[[SLICE_SIZE_2]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_2:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL_2]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64_2:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR_2]])
  // CHECK-NEXT:   %[[RECV_OUT:.*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_KEY]], %[[DEVICE_ORDINAL_SCALAR_64_2]])
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> tensor<1xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>, shape = #tf_type.shape<>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> tensor<1xi32>

    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()
  func.return
}

// -----

// Check that tensor to send is converted to replicated layout before send.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK:        %[[ALL_GATHER_OUT:.*]] = "tf.DTensorAllGather"
  // CHECK-SAME:   output_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-NEXT:   %[[ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<[0, 1, 2, 3]>
  // CHECK-NEXT:   %[[SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
  // CHECK-NEXT:   %[[SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   %[[ZERO:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64]], %[[ZERO]])
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE]])
  // CHECK-NEXT:     "tf.XlaSendToHost"(%[[ALL_GATHER_OUT]])
  // CHECK-SAME:     key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[SIZE_TYPE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID_2:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE_2]])
  // CHECK-NEXT:   %[[SLICE_SIZE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_2:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL_2]], %[[DEVICE_ID_2]], %[[SLICE_SIZE_2]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_2:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL_2]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64_2:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR_2]])
  // CHECK-NEXT:   %[[RECV_OUT:.*]] = "tf._XlaRecvAtHostV2"(%[[PROGRAM_KEY]], %[[DEVICE_ORDINAL_SCALAR_64_2]])
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2xi32>) -> tensor<2xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<2xi32>) -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>, shape = #tf_type.shape<>} : () -> tensor<2xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<2xi32>) -> tensor<2xi32>

    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()
  func.return
}

// -----

// Check that send/recv to clusters with same mesh is disallowed.
func.func @main(%arg0: tensor<i32>) {

  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|*CPU>} : (tensor<1xi32>) -> tensor<1xi32>
    // expected-error @+1 {{Only use CopyToMesh to transfer data across different mesh cluster}}
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|*CPU>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=2|*CPU"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|*CPU>, shape = #tf_type.shape<>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=2|*CPU>} : (tensor<1xi32>) -> tensor<1xi32>

    tf_device.return
  }) {_mesh = "CPU|x=2|*CPU"} : () -> ()
  func.return
}

// -----

// Check that multi-mesh transfer between two non host clusters is disallowed.
func.func @main(%arg0: tensor<i32>) {

  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<1xi32>) -> tensor<1xi32>
    // expected-error @+1 {{f.CopyToMesh op must be used to send data from/to host mesh}}
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:GPU|x=2|0,1|0,1|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:GPU|x=2|*GPU>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:GPU|x=2|0,1|0,1|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:GPU|x=2|*GPU>, shape = #tf_type.shape<>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:GPU|x=2|*GPU>} : (tensor<1xi32>) -> tensor<1xi32>

    tf_device.return
  }) {_mesh = "GPU|x=2|*GPU"} : () -> ()
  func.return
}

// -----

// Check that send/recv between two CPUs works.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:   "tf._HostSend"(%[[CONST_OUT]])
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf._HostRecv"
  // CHECK-SAME:     tensor_name = "communication_key_sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1_0"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0>} : (tensor<1xi32>) -> tensor<1xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1_0", target_layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1>} : (tensor<1xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1_0", layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1>, shape = #tf_type.shape<1>} : () -> tensor<1xi32>
    %1 = "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1>} : (tensor<1xi32>) -> tensor<1xi32>
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:1"} : () -> ()
  func.return
}

// -----

// Check that Data transfer from CPU to GPU is lowered correctly.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[INPUT:.*]] = "tf.Const"
  // CHECK-NEXT:   "tf._HostSend"(%[[INPUT]])
  // CHECK-NEXT:   %[[ID_TO_ORDINAL:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[SIZE_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
  // CHECK-NEXT:   %[[SLICE_SIZE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR]])
  // CHECK-NEXT:   %[[ZERO:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64]], %[[ZERO]])
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[ID_TO_ORDINAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<[0, 1, 2, 3]>
  // CHECK-NEXT:   %[[SIZE_TYPE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ID_2:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE_2]])
  // CHECK-NEXT:   %[[SLICE_SIZE_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<1>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_2:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL_2]], %[[DEVICE_ID_2]], %[[SLICE_SIZE_2]])
  // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<>
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_2:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL_2]], %[[SCALAR_TYPE]])
  // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR_64_2:.*]] = "tf.Cast"(%[[DEVICE_ORDINAL_SCALAR_2]])
  // CHECK-NEXT:   %[[ZERO_2:.*]] = "tf.Const"
  // CHECK-SAME:   value = dense<0>
  // CHECK-NEXT:   %[[PREDICATE_2:.*]] = "tf.Equal"(%[[DEVICE_ORDINAL_SCALAR_64_2]], %[[ZERO_2]])
  // CHECK-NEXT:   %[[IF_OUT:.*]] = "tf.IfRegion"(%[[PREDICATE_2]])
  // CHECK-NEXT:     %[[RECV_OUT:.*]] = "tf._HostRecv"
  // CHECK-NEXT:     "tf.Yield"(%[[RECV_OUT]])
  // CHECK:          %[[ZEROS_3:.*]] = "tf.Const"
  // CHECK-NEXT:     "tf.Yield"(%[[ZEROS_3]])
  // CHECK:       %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[OUGPUT:.*]] = "tf.DTensorAllReduce"(%[[IF_OUT]], %[[GROUP_ASSIGNMENT]])
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<1.> : tensor<8x8xf32>} : () -> tensor<8x8xf32>
    %1 = "tf.DTensorLayout"(%0) {_global_shape = [#tf_type.shape<8x8>], global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3_0", target_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>} : (tensor<8x8xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()
  "tf_device.cluster"() ({
    %0 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3_0", layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>, shape = #tf_type.shape<8x8>} : () -> tensor<8x8xf32>
    %1 = "tf.DTensorLayout"(%0) {_global_shape = [#tf_type.shape<8x8>], global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    tf_device.return
  }) {_mesh = "GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> ()
  func.return
}
