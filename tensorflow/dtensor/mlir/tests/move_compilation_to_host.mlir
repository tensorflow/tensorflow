// RUN: dtensor-opt %s -split-input-file -dtensor-move-compilation-to-host -verify-diagnostics | FileCheck %s

// Check that TPU Compilation ops are moved to host computation functions and
// Send/Recv ops are inserted to transfer program key.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 656 : i32}}  {
  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<i32>,%arg1: tensor<4xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<f32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:       "tf.StatefulPartitionedCall"
    // CHECK-SAME:  _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK-SAME:  f = @_func_0
    // CHECK-NEXT:  "tf.StatefulPartitionedCall"
    // CHECK-SAME:  _mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:  f = @_func_1
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = [], _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_0} : (tensor<i32>, tensor<4xi32>) -> ()
    %0 = "tf.StatefulPartitionedCall"(%arg0) {_layout = ["sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], _mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config_proto = "", executor_type = "", f = @_func_1} : (tensor<i32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  // CHECK-LABEL: func private @_func_0
  // CHECK-SAME:  %[[ARG0:.*]]: tensor<i32>
  // CHECK-SAME:  %[[ARG1:.*]]: tensor<4xi32>
  func.func private @_func_0(%arg0: tensor<i32>, %arg1: tensor<4xi32>) {
    // CHECK-NEXT:   %[[ID_TO_ORDINAL:.*]] = "tf.Const"
    // CHECK-SAME:   value = dense<[0, 1]>
    // CHECK-NEXT:   %[[SIZE_TYPE:.*]] = "tf.Const"
    // CHECK-SAME:   value = dense<1>
    // CHECK-NEXT:   %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
    // CHECK-NEXT:   %[[SLICE_SIZE:.*]] = "tf.Const"
    // CHECK-SAME:   value = dense<1>
    // CHECK-NEXT:   %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
    // CHECK-NEXT:   %[[SCALAR_TYPE:.*]] = "tf.Const"
    // CHECK-SAME:   value = dense<>
    // CHECK-NEXT:   %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
    // CHECK:        %[[PROGRAM_KEY:.*]] = "tf.Case"(%[[DEVICE_ORDINAL_SCALAR]])
    // CHECK-NEXT: "tf_device.launch"()
    // CHECK-NEXT:   "tf.TPUExecute"(%[[ARG0]], %[[ARG1]], %[[PROGRAM_KEY]])
    // CHECK-NEXT:   tf_device.return
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%arg0, %arg1, %0#1) : (tensor<i32>, tensor<4xi32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  // CHECK-LABEL: func private @_func_1
  // CHECK-SAME:  %[[ARG0:.*]]: tensor<i32>
  func.func private @_func_1(%arg0: tensor<i32>) -> tensor<f32> {
    // CHECK:      %[[COMPILE_OUT:.*]]:2 = "tf_device.launch"()
    // CHECK-NEXT:   %[[COMPILATION_STATUS:.*]], %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlir"()
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:TPU:0"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device_incarnation = 0
    // CHECK-SAME:   tensor_name = "compilation_send_recv_key_0
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device_incarnation = 0
    // CHECK-SAME:   tensor_name = "compilation_send_recv_key_1
    // CHECK-NEXT:   tf_device.return %[[COMPILATION_STATUS]], %[[PROGRAM_KEY]]
    // CHECK-NEXT: device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-NEXT: "tf_device.launch"()
    // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUT]]#0)
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:localhost/replica:0/task:0/device:CPU:0"
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
    // CHECK-NEXT: %[[BROADCASTED_KEY:.*]] = "tf.Case"(%[[DEVICE_ORDINAL_SCALAR]])
    // CHECK-NEXT: "tf.Const"
    // CHECK-NEXT: "tf.Cast"
    // CHECK-NEXT: %[[MOD_OUT:.*]] = "tf.FloorMod"
    // CHECK-NEXT: "tf._XlaRecvAtHostV2"(%[[BROADCASTED_KEY]], %[[MOD_OUT]]
    // CHECK-NEXT: "tf.Sqrt"
    // CHECK-NEXT: "tf.Identity"
    // CHECK-NEXT: "tf.Identity"
    // CHECK-NEXT: return
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %2 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %3 = "tf.FloorMod"(%2, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%1, %3) {key = "communication_key_sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_0"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<f32>
    %5 = "tf.Sqrt"(%4) : (tensor<f32>) -> tensor<f32>
    %6 = "tf.Identity"(%5) : (tensor<f32>) -> tensor<f32>
    %7 = "tf.Identity"(%6) : (tensor<f32>) -> tensor<f32>
    func.return %7 : tensor<f32>
  }
}

// -----

// Check that TPU Compilation ops are moved to host computation functions and
// Send/Recv ops are inserted to transfer program key for
// TPUExecuteAndUpdateVariables op
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 656 : i32}}  {
  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<i32>,%arg1: tensor<*x!tf_type.resource<tensor<4xf32>>> {tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<f32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:       "tf.StatefulPartitionedCall"
    // CHECK-SAME:  _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK-SAME:  f = @_func_0
    // CHECK-NEXT:  "tf.StatefulPartitionedCall"
    // CHECK-SAME:  _mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:  f = @_func_1
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = [], _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_0} : (tensor<i32>, tensor<*x!tf_type.resource<tensor<4xf32>>>) -> ()
    %0 = "tf.StatefulPartitionedCall"(%arg0) {_layout = ["sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], _mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config_proto = "", executor_type = "", f = @_func_1} : (tensor<i32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  // CHECK-LABEL: func private @_func_0
  // CHECK-SAME:  %[[ARG0:.*]]: tensor<i32>
  // CHECK-SAME:  %[[ARG1:.*]]: tensor<*x!tf_type.resource<tensor<4xf32>>>
  func.func private @_func_0(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<4xf32>>>) {
    // CHECK-NEXT: %[[ID_TO_ORDINAL:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<[0, 1]>
    // CHECK-NEXT: %[[SIZE_TYPE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<1>
    // CHECK-NEXT: %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
    // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<1>
    // CHECK-NEXT: %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
    // CHECK-NEXT: %[[SCALAR_TYPE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<>
    // CHECK-NEXT: %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
    // CHECK:      %[[PROGRAM_KEY:.*]] = "tf.Case"(%[[DEVICE_ORDINAL_SCALAR]])
    // CHECK-NEXT: "tf_device.launch"()
    // CHECK-NEXT:   "tf.TPUExecuteAndUpdateVariables"(%[[ARG1]], %[[PROGRAM_KEY]])
    // CHECK-NEXT:   tf_device.return
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecuteAndUpdateVariables"(%arg1, %0#1) {device_var_reads_indices = [0], device_var_updates_indices = [-1]} : (tensor<*x!tf_type.resource<tensor<4xf32>>>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  // CHECK-LABEL: func private @_func_1
  // CHECK-SAME:  %[[ARG0:.*]]: tensor<i32>
  func.func private @_func_1(%arg0: tensor<i32>) -> tensor<f32> {
    // CHECK:      %[[COMPILE_OUT:.*]]:2 = "tf_device.launch"()
    // CHECK-NEXT:   %[[COMPILATION_STATUS:.*]], %[[PROGRAM_KEY:.*]] = "tf._TPUCompileMlir"()
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device_incarnation = 0
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:TPU:0"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device_incarnation = 0
    // CHECK-SAME:   tensor_name = "compilation_send_recv_key_0
    // CHECK-NEXT:   "tf._HostSend"(%[[PROGRAM_KEY]])
    // CHECK-SAME:   device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   recv_device = "/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK-SAME:   send_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-SAME:   send_device_incarnation = 0
    // CHECK-SAME:   tensor_name = "compilation_send_recv_key_1
    // CHECK-NEXT:   tf_device.return %[[COMPILATION_STATUS]], %[[PROGRAM_KEY]]
    // CHECK-NEXT: device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-NEXT: "tf_device.launch"()
    // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUT]]#0)
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:localhost/replica:0/task:0/device:CPU:0"
    // CHECK-NEXT: %[[ID_TO_ORDINAL:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<0>
    // CHECK-NEXT: %[[SIZE_TYPE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<1>
    // CHECK-NEXT: %[[DEVICE_ID:.*]] = "tf.Reshape"(%[[ARG0]], %[[SIZE_TYPE]])
    // CHECK-NEXT: %[[SLICE_SIZE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<1>
    // CHECK-NEXT: %[[DEVICE_ORDINAL:.*]] = "tf.Slice"(%[[ID_TO_ORDINAL]], %[[DEVICE_ID]], %[[SLICE_SIZE]])
    // CHECK-NEXT: %[[SCALAR_TYPE:.*]] = "tf.Const"
    // CHECK-SAME: value = dense<>
    // CHECK-NEXT: %[[DEVICE_ORDINAL_SCALAR:.*]] = "tf.Reshape"(%[[DEVICE_ORDINAL]], %[[SCALAR_TYPE]])
    // CHECK-NEXT: %[[BROADCASTED_KEY:.*]] = "tf.Case"(%[[DEVICE_ORDINAL_SCALAR]])
    // CHECK-NEXT: "tf.Const"
    // CHECK-NEXT: "tf.Cast"
    // CHECK-NEXT: %[[MOD_OUT:.*]] = "tf.FloorMod"
    // CHECK-NEXT: "tf._XlaRecvAtHostV2"(%[[BROADCASTED_KEY]], %[[MOD_OUT]]
    // CHECK-NEXT: "tf.Sqrt"
    // CHECK-NEXT: "tf.Identity"
    // CHECK-NEXT: "tf.Identity"
    // CHECK-NEXT: return
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %2 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %3 = "tf.FloorMod"(%2, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%1, %3) {key = "communication_key_sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_0"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<f32>
    %5 = "tf.Sqrt"(%4) : (tensor<f32>) -> tensor<f32>
    %6 = "tf.Identity"(%5) : (tensor<f32>) -> tensor<f32>
    %7 = "tf.Identity"(%6) : (tensor<f32>) -> tensor<f32>
    func.return %7 : tensor<f32>
  }
}

// -----

// Check that StatefulPartitionedCall op without mesh specification is
// disallowed.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 656 : i32}}  {
  func.func @main(%arg0: tensor<i32>,%arg1: tensor<4xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<f32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_0} : (tensor<i32>, tensor<4xi32>) -> ()
    // expected-error @+1 {{StatefulPartitionCall op must have `_mesh` attribute specified}}
    "tf.StatefulPartitionedCall"(%arg0) {config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_2} : (tensor<i32>) -> ()
    %0 = "tf.StatefulPartitionedCall"(%arg0) {_mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config_proto = "", executor_type = "", f = @_func_1} : (tensor<i32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  func.func private @_func_0(%arg0: tensor<i32>, %arg1: tensor<4xi32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%arg0, %arg1, %0#1) : (tensor<i32>, tensor<4xi32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_2(%arg0: tensor<i32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_1(%arg0: tensor<i32>) -> tensor<f32> {
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %2 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %3 = "tf.FloorMod"(%2, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%1, %3) {key = "communication_key_sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_0"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<f32>
    %5 = "tf.Identity"(%4) : (tensor<f32>) -> tensor<f32>
    func.return %5 : tensor<f32>
  }
}

// -----

// Check that multiple TPU cluster computations with same mesh is disallowed.
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 656 : i32}}  {
  func.func @main(%arg0: tensor<i32>,%arg1: tensor<4xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<f32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_0} : (tensor<i32>, tensor<4xi32>) -> ()
    // expected-error @+1 {{There should be exactly 1 function for each mesh in computation cluster}}
    "tf.StatefulPartitionedCall"(%arg0) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_2} : (tensor<i32>) -> ()
    %0 = "tf.StatefulPartitionedCall"(%arg0) {_mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config_proto = "", executor_type = "", f = @_func_1} : (tensor<i32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  func.func private @_func_0(%arg0: tensor<i32>, %arg1: tensor<4xi32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%arg0, %arg1, %0#1) : (tensor<i32>, tensor<4xi32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_2(%arg0: tensor<i32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_1(%arg0: tensor<i32>) -> tensor<f32> {
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %2 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %3 = "tf.FloorMod"(%2, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%1, %3) {key = "communication_key_sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_0"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<f32>
    %5 = "tf.Identity"(%4) : (tensor<f32>) -> tensor<f32>
    func.return %5 : tensor<f32>
  }
}

// -----

// Check that at multiple TPU computations are disallowed.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 656 : i32}}  {
  // expected-error @+1 {{Only 1 XLA cluster for DTensor computation is supported for now}}
  func.func @main(%arg0: tensor<i32>,%arg1: tensor<4xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<f32> {tf._global_shape = #tf_type.shape<>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func_0} : (tensor<i32>, tensor<4xi32>) -> ()
    "tf.StatefulPartitionedCall"(%arg0) {_mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:TPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:TPU:0", config_proto = "", executor_type = "", f = @_func_2} : (tensor<i32>) -> ()
    %0 = "tf.StatefulPartitionedCall"(%arg0) {_mesh = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config = "|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", config_proto = "", executor_type = "", f = @_func_1} : (tensor<i32>) -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  func.func private @_func_0(%arg0: tensor<i32>, %arg1: tensor<4xi32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%arg0, %arg1, %0#1) : (tensor<i32>, tensor<4xi32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_2(%arg0: tensor<i32>) {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "...", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    "tf_device.launch"() ({
      "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    func.return
  }

  func.func private @_func_1(%arg0: tensor<i32>) -> tensor<f32> {
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %2 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %3 = "tf.FloorMod"(%2, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%1, %3) {key = "communication_key_sharding_specs: mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_0"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<f32>
    %5 = "tf.Identity"(%4) : (tensor<f32>) -> tensor<f32>
    func.return %5 : tensor<f32>
  }
}

