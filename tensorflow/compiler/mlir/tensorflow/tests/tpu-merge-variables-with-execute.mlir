// RUN: tf-opt -split-input-file -verify-diagnostics -tf-tpu-merge-variables-with-execute %s | FileCheck %s

// Tests that the pass merges only variable reads/writes on the same device.

// CHECK-LABEL: func @merge_same_device_variables
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:.*]]: tensor<*x!tf_type.resource<tensor<16xf32>>>
func.func @merge_same_device_variables(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf_type.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg2: tensor<*x!tf_type.resource<tensor<16xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) {
  // CHECK-NEXT: %[[ID_0:.*]] = "tf.IdentityN"(%[[ARG_0]])
  %id0 = "tf.IdentityN"(%arg0) {device = "/job:localhost/replica:0/task:0/device:TPU:0"}
    : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<*x!tf_type.resource<tensor<32xf32>>>
  // CHECK-NEXT: %[[READ_2:.*]] = "tf.ReadVariableOp"(%[[ARG_2]])
  %read0 = "tf.ReadVariableOp"(%id0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>) -> tensor<64xf32>
  %read2 = "tf.ReadVariableOp"(%arg2) : (tensor<*x!tf_type.resource<tensor<16xf32>>>) -> tensor<16xf32>
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  %compile:2 = "tf_device.launch"() ({
      // CHECK: tf._TPUCompileMlir
      %0:2 = "tf._TPUCompileMlir"() {
        metadata = "",
        mlir_module = "module attributes {tf.versions = {producer = 888 : i32}} {\0A  func @main(%arg0: tensor<32xf32>, %arg1: tensor<64xf32>, %arg2: tensor<16xf32>) -> (tensor<32xf32>, tensor<16xf32>) {\0A    %0:2 = \22tf.A\22(%arg0, %arg1, %arg2) : (tensor<32xf32>, tensor<64xf32>, tensor<16xf32>) -> (tensor<32xf32>, tensor<16xf32>)\0A    return %0#0, %0#1 : tensor<32xf32>, tensor<16xf32>\0A  }\0A}"
      } : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
      tf_device.return %0#0, %0#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
    }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK: %[[EXE:.*]] = "tf_device.launch"
  // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
  // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[ID_0]], %[[ARG_1]], %[[READ_2]], %[[COMPILE]]#1)
  // CHECK-SAME: device_var_reads_indices = [0, 1],
  // CHECK-SAME: device_var_updates_indices = [0, -1]
  %execute:2 = "tf_device.launch"() ({
    %0:2 = "tf.TPUExecute"(%read0, %read1, %read2, %compile#1) {
      Targs = [tensor<32xf32>, tensor<64xf32>, tensor<16xf32>],
      Tresults = [tensor<32xf32>, tensor<16xf32>]}
      : (tensor<32xf32>, tensor<64xf32>, tensor<16xf32>, tensor<2x!tf_type.string>) -> (tensor<32xf32>, tensor<16xf32>)
    tf_device.return %0#0, %0#1 : tensor<32xf32>, tensor<16xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> (tensor<32xf32>, tensor<16xf32>)
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: })
  "tf.AssignVariableOp"(%id0, %execute#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_2]], %[[EXE]])
  "tf.AssignVariableOp"(%arg2, %execute#1) : (tensor<*x!tf_type.resource<tensor<16xf32>>>, tensor<16xf32>) -> ()
  // CHECK-NEXT: return
  func.return
}

// -----

// Tests that the pass do not check devices for replicated region.

// CHECK-LABEL: func @merge_replicated_variables
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>, %[[ARG_1:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>,
// CHECK-SAME: %[[ARG_2:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
func.func @merge_replicated_variables(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf_type.resource<tensor<32xf32>>>) {
  // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  %compile:2 = "tf_device.launch"() ({
    // CHECK: tf._TPUCompileMlir
    %0:2 = "tf._TPUCompileMlir"() {
      metadata = "",
      mlir_module = "module attributes {tf.versions = {producer = 888 : i32}} {\0A  func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> (tensor<32xf32>) {\0A    %0 = \22tf.A\22(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> (tensor<32xf32>)\0A    return %0 : tensor<32xf32>\0A  }\0A}"
    } : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %0#0, %0#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK: tf_device.replicate([%[[ARG_1]], %[[ARG_2]]] as %[[R_ARG:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>)
  tf_device.replicate([%arg1, %arg2] as %r: tensor<*x!tf_type.resource<tensor<32xf32>>>) {n = 2 : i32} {
    // CHECK-NEXT: "tf_device.launch"
    // CHECK-SAME: <{device = ""}>
    // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[READ_0]], %[[R_ARG]], %[[COMPILE]]#1)
    // CHECK-SAME: device_var_reads_indices = [1],
    // CHECK-SAME: device_var_updates_indices = [0]
    %read1 = "tf.ReadVariableOp"(%r) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %execute = "tf_device.launch"() ({
      %0 = "tf.TPUExecute"(%read0, %read1, %compile#1)
        : (tensor<32xf32>, tensor<32xf32>, tensor<2x!tf_type.string>) -> tensor<32xf32>
      tf_device.return %0 : tensor<32xf32>
    }) {device = ""} : () -> tensor<32xf32>
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: })
    "tf.AssignVariableOp"(%r, %execute) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    // CHECK-NEXT: tf_device.return
    tf_device.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: return
  func.return
}

// -----

// Tests that the pass do not merge reads/assigns if there are interfering
// accesses in between.

// CHECK-LABEL: func @interfering_accesses
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:.*]]: tensor<32xf32>
// CHECK-SAME: %[[ARG_4:.*]]: tensor<*x!tf_type.resource<tensor<8xf32>>>
// CHECK-SAME: %[[ARG_5:.*]]: tensor<*x!tf_type.resource<tensor<2xf32>>>
// CHECK-SAME: %[[ARG_6:.*]]: tensor<2xf32>
func.func @interfering_accesses(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf_type.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg2: tensor<32xf32>,
  %arg4: tensor<*x!tf_type.resource<tensor<8xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg5: tensor<*x!tf_type.resource<tensor<2xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg6: tensor<2xf32>) -> (tensor<8xf32>) {
  // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  // CHECK-NEXT: %[[READ_5:.*]] = "tf.ReadVariableOp"(%[[ARG_5]])
  %read5 = "tf.ReadVariableOp"(%arg5) : (tensor<*x!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[ARG_2]])
  "tf.AssignVariableOp"(%arg0, %arg2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_5]], %[[ARG_6]])
  "tf.AssignVariableOp"(%arg5, %arg6) : (tensor<*x!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>) -> tensor<64xf32>
  %read2 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  %compile:2 = "tf_device.launch"() ({
    // CHECK: tf._TPUCompileMlir
    %0:2 = "tf._TPUCompileMlir"() {
      metadata = "",
      mlir_module = "module attributes {tf.versions = {producer = 888 : i32}} {\0A  func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> (tensor<32xf32>) {\0A    %0 = \22tf.A\22(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> (tensor<32xf32>)\0A    return %0 : tensor<32xf32>\0A  }\0A}"
    } : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %0#0, %0#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK: %[[EXE:.*]]:2 = "tf_device.launch"
  // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
  // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[READ_0]], %[[ARG_1]], %[[ARG_4]], %[[READ_5]], %[[COMPILE]]#1)
  // CHECK-SAME: device_var_reads_indices = [1, 2],
  // CHECK-SAME: device_var_updates_indices = [1, -1]
  %execute:3 = "tf_device.launch"() ({
    %0:3 = "tf.TPUExecute"(%read0, %read1, %read2, %read5, %compile#1) {
      Targs = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>],
      Tresults = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>]}
      : (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>, tensor<2x!tf_type.string>)
        -> (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>)
    tf_device.return %0#0, %0#1, %0#2 : tensor<32xf32>, tensor<64xf32>, tensor<8xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>)
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: })
  "tf.AssignVariableOp"(%arg1, %execute#1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>, tensor<64xf32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#0)
  "tf.AssignVariableOp"(%arg0, %execute#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: %[[READ_3:.*]] = "tf.ReadVariableOp"(%[[ARG_4]])
  %read3 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_4]], %[[EXE]]#1)
  "tf.AssignVariableOp"(%arg4, %execute#2) : (tensor<*x!tf_type.resource<tensor<8xf32>>>, tensor<8xf32>) -> ()
  // CHECK-NEXT: return %[[READ_3]]
  func.return %read3 : tensor<8xf32>
}

// -----

// Tests that the pass does merge reads/assigns if there are resource accesses
// in between which are not interfering.
// For %arg0 and %arg1 both reads and assigns are merged, for %arg3 and %arg4
// only reads are merged.

// CHECK-LABEL: func @non_interfering_accesses
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:[a-z0-9]+]]: tensor<32xf32>
// CHECK-SAME: %[[ARG_3:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<8xf32>>>
// CHECK-SAME: %[[ARG_4:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<2xf32>>>
// CHECK-SAME: %[[ARG_5:[a-z0-9]+]]: tensor<2xf32>
// CHECK-SAME: %[[ARG_6:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_7:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<2xf32>>>
func.func @non_interfering_accesses(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf_type.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg2: tensor<32xf32>,
  %arg3: tensor<*x!tf_type.resource<tensor<8xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg4: tensor<*x!tf_type.resource<tensor<2xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg5: tensor<2xf32>,
  %arg6: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg7: tensor<*x!tf_type.resource<tensor<2xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<8xf32>) {
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %read5 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_6]], %[[ARG_2]])
  "tf.AssignVariableOp"(%arg6, %arg2) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_7]], %[[ARG_5]])
  "tf.AssignVariableOp"(%arg7, %arg5) : (tensor<*x!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>) -> tensor<64xf32>
  %read2 = "tf.ReadVariableOp"(%arg3) : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  %compile:2 = "tf_device.launch"() ({
    // CHECK: tf._TPUCompileMlir
    %0:2 = "tf._TPUCompileMlir"() {
      metadata = "",
      mlir_module = "module attributes {tf.versions = {producer = 888 : i32}} {\0A  func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> (tensor<32xf32>) {\0A    %0 = \22tf.A\22(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> (tensor<32xf32>)\0A    return %0 : tensor<32xf32>\0A  }\0A}"
    } : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %0#0, %0#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK: %[[EXE:.*]] = "tf_device.launch"
  // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
  // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[ARG_0]], %[[ARG_1]], %[[ARG_3]], %[[ARG_4]], %[[COMPILE]]#1)
  // CHECK-SAME: device_var_reads_indices = [0, 1, 2, 3],
  // CHECK-SAME: device_var_updates_indices = [0, 1, -1, -1]
  %execute:3 = "tf_device.launch"() ({
    %0:3 = "tf.TPUExecute"(%read0, %read1, %read2, %read5, %compile#1) {
      Targs = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>],
      Tresults = [tensor<32xf32>, tensor<64xf32>, tensor<8xf32>]}
      : (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>, tensor<2xf32>, tensor<2x!tf_type.string>)
        -> (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>)
    tf_device.return %0#0, %0#1, %0#2 : tensor<32xf32>, tensor<64xf32>, tensor<8xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> (tensor<32xf32>, tensor<64xf32>, tensor<8xf32>)
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: })
  // CHECK-NEXT: %[[READ:.*]] = "tf.ReadVariableOp"(%[[ARG_3]])
  %read3 = "tf.ReadVariableOp"(%arg3) : (tensor<*x!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_3]], %[[EXE]])
  "tf.AssignVariableOp"(%arg3, %execute#2) : (tensor<*x!tf_type.resource<tensor<8xf32>>>, tensor<8xf32>) -> ()
  "tf.AssignVariableOp"(%arg1, %execute#1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>, tensor<64xf32>) -> ()
  "tf.AssignVariableOp"(%arg0, %execute#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: return %[[READ]]
  func.return %read3 : tensor<8xf32>
}

// -----

// Tests that the pass does not merge for an execute node that has multiple
// inputs read from the same variable.

// CHECK-LABEL: func @do_not_merge_multi_read
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<!tf_type.string>
func.func @do_not_merge_multi_read(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<!tf_type.string>) {
  // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  // CHECK-NEXT: %[[READ_1:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
  %read1 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  // CHECK-NEXT: %[[EXE:.*]] = "tf_device.launch"
  // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
  // CHECK-NEXT: "tf.TPUExecute"(%[[READ_0]], %[[READ_1]], %[[ARG_1]])
  %execute = "tf_device.launch"() ({
    %0 = "tf.TPUExecute"(%read0, %read1, %arg1) {
      Targs = [tensor<32xf32>, tensor<32xf32>], Tresults = [tensor<32xf32>]}
      : (tensor<32xf32>, tensor<32xf32>, tensor<!tf_type.string>) -> (tensor<32xf32>)
    tf_device.return %0 : tensor<32xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<32xf32>
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: })
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]])
  "tf.AssignVariableOp"(%arg0, %execute) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: return
  func.return
}

// -----

// Tests that the pass do not merge for an execute node that has multiple
// outputs used to update a variable.

// CHECK-LABEL: func @do_not_merge_multi_assign
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<!tf_type.string>
func.func @do_not_merge_multi_assign(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<!tf_type.string>) {
  // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ARG_0]])
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  // CHECK-NEXT: %[[EXE:.*]]:2 = "tf_device.launch"
  // CHECK-SAME: <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
  // CHECK-NEXT: "tf.TPUExecute"(%[[READ_0]], %[[ARG_1]])
  %execute:2 = "tf_device.launch"() ({
    %0:2 = "tf.TPUExecute"(%read0, %arg1) {
      Targs = [tensor<32xf32>], Tresults = [tensor<32xf32>, tensor<32xf32>]}
      : (tensor<32xf32>, tensor<!tf_type.string>) -> (tensor<32xf32>, tensor<32xf32>)
    tf_device.return %0#0, %0#1 : tensor<32xf32>, tensor<32xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> (tensor<32xf32>, tensor<32xf32>)
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: })
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#0)
  "tf.AssignVariableOp"(%arg0, %execute#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[ARG_0]], %[[EXE]]#1)
  "tf.AssignVariableOp"(%arg0, %execute#1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  // CHECK-NEXT: return
  func.return
}

// -----

// Tests that the pass merges only variable reads/writes on the same device,
// with TPUExecutes in a tf_device.parallel_execute.

// CHECK-LABEL: func @parallel_execute
// CHECK-SAME: %[[ARG_0:.*]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_2:.*]]: tensor<!tf_type.string>
func.func @parallel_execute(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<*x!tf_type.resource<tensor<64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg2: tensor<!tf_type.string>) {
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>) -> tensor<64xf32>
  // CHECK-NOT: "tf.ReadVariableOp"
  // CHECK: "tf_device.parallel_execute"
  %pe:2 = "tf_device.parallel_execute"() ({
    // CHECK: "tf_device.launch"
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:0"
    %execute0 = "tf_device.launch"() ({
      // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[ARG_0]], %[[ARG_2]])
      %0 = "tf.TPUExecute"(%read0, %arg2) : (tensor<32xf32>, tensor<!tf_type.string>) -> tensor<32xf32>
      // CHECK-NEXT: tf_device.return
      tf_device.return %0 : tensor<32xf32>
    }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<32xf32>
    tf_device.return %execute0 : tensor<32xf32>
  }, {
    // CHECK: "tf_device.launch"
    // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:TPU:1"
    %execute1 = "tf_device.launch"() ({
      // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[ARG_1]], %[[ARG_2]])
      %1 = "tf.TPUExecute"(%read1, %arg2) : (tensor<64xf32>, tensor<!tf_type.string>) -> tensor<64xf32>
      // CHECK-NEXT: tf_device.return
      tf_device.return %1 : tensor<64xf32>
    }) {device = "/job:localhost/replica:0/task:0/device:TPU:1"} : () -> tensor<64xf32>
    tf_device.return %execute1 : tensor<64xf32>
  }) : () -> (tensor<32xf32>, tensor<64xf32>)
  // CHECK-NOT: "tf.AssignVariableOp"
  "tf.AssignVariableOp"(%arg0, %pe#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  "tf.AssignVariableOp"(%arg1, %pe#1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>, tensor<64xf32>) -> ()
  func.return
}

// -----

// Tests that the pass merges variable reads/writes for TPUExecutes in a
// tf_device.parallel_execute that is replicated (tf_device.replicate).

// CHECK-LABEL: func @replicated_parallel_execute
// CHECK-SAME: %[[ARG_0:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_1:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
// CHECK-SAME: %[[ARG_2:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_3:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
// CHECK-SAME: %[[ARG_4:[a-z0-9]+]]: tensor<!tf_type.string>
func.func @replicated_parallel_execute(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>,
  %arg2: tensor<*x!tf_type.resource<tensor<64xf32>>>,
  %arg3: tensor<*x!tf_type.resource<tensor<64xf32>>>,
  %arg4: tensor<!tf_type.string>) {
  // CHECK: tf_device.replicate
  // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<32xf32>>>
  // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]+]]: tensor<*x!tf_type.resource<tensor<64xf32>>>
  tf_device.replicate([%arg0, %arg1] as %ri0: tensor<*x!tf_type.resource<tensor<32xf32>>>,
                      [%arg2, %arg3] as %ri1: tensor<*x!tf_type.resource<tensor<64xf32>>>) {n = 2 : i32} {
    // CHECK-NOT: "tf.ReadVariableOp"
    %read0 = "tf.ReadVariableOp"(%ri0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
    %read1 = "tf.ReadVariableOp"(%ri1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>) -> tensor<64xf32>
    // CHECK: "tf_device.parallel_execute"
    %pe:2 = "tf_device.parallel_execute"() ({
      // CHECK: "tf_device.launch"
      %execute0 = "tf_device.launch"() ({
        // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[RI_0]], %[[ARG_4]])
        %0 = "tf.TPUExecute"(%read0, %arg4) : (tensor<32xf32>, tensor<!tf_type.string>) -> tensor<32xf32>
        // CHECK-NEXT: tf_device.return
        tf_device.return %0 : tensor<32xf32>
      }) {device = ""} : () -> tensor<32xf32>
      tf_device.return %execute0 : tensor<32xf32>
    }, {
      // CHECK: "tf_device.launch"
      %execute1 = "tf_device.launch"() ({
        // CHECK-NEXT: "tf.TPUExecuteAndUpdateVariables"(%[[RI_1]], %[[ARG_4]])
        %1 = "tf.TPUExecute"(%read1, %arg4) : (tensor<64xf32>, tensor<!tf_type.string>) -> tensor<64xf32>
        // CHECK-NEXT: tf_device.return
        tf_device.return %1 : tensor<64xf32>
      }) {device = ""} : () -> tensor<64xf32>
      tf_device.return %execute1 : tensor<64xf32>
    }) : () -> (tensor<32xf32>, tensor<64xf32>)
    // CHECK-NOT: "tf.AssignVariableOp"
    "tf.AssignVariableOp"(%ri0, %pe#0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
    "tf.AssignVariableOp"(%ri1, %pe#1) : (tensor<*x!tf_type.resource<tensor<64xf32>>>, tensor<64xf32>) -> ()
  }
  func.return
}

// -----

// Tests that resource variables not hoisted are flagged.

func.func @missing_read_write(
  %arg0: tensor<*x!tf_type.resource<tensor<32xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"},
  %arg1: tensor<!tf_type.string>) {
  %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %execute:2 = "tf_device.launch"() ({
    // expected-error @+1 {{resource that was neither read nor written to}}
    %0:2 = "tf.TPUExecute"(%arg0, %read0, %arg1)
      : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>, tensor<!tf_type.string>) -> (tensor<32xf32>, tensor<32xf32>)
    tf_device.return %0#0, %0#1 : tensor<32xf32>, tensor<32xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> (tensor<32xf32>, tensor<32xf32>)
  "tf.AssignVariableOp"(%arg0, %execute#1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>, tensor<32xf32>) -> ()
  func.return
}
