// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion | FileCheck %s

// Check SPMD for save op for replicated tensor only happens on device 0.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  "tf_device.cluster"() ({
    // CHECK:      "tf.Case"
    // CHECK-SAME: branches = [@tf.[[D0:.*]], @tf.[[D1:.*]]], is_stateless = false
    // CHECK:      func private @tf.[[D0]]
    // CHECK:      %[[CST:.*]] = "tf.Const"() <{value = dense<"_dev-0-of-2">
    // CHECK:      "tf.Add"(%arg0, %[[CST]])
    // CHECK:      ""
    // CHECK:      func private @tf.[[D1]]
    // CHECK:      "tf.NoOp"
    %0 = "tf.Const"() {value = dense<"/dev/null"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1 = "tf.Const"() {value = dense<"t1"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %2 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %3 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    "tf.SaveV2"(%0, %1, %2, %3) : (tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
  func.return
}

// -----

// Check SPMD for save op for sharded tensor.
// The following should generate a switch case on device id, and 2 save ops in each branch.
// One for device 0, another for device 1.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  "tf_device.cluster"() ({
    // CHECK:      tf.Case
    // CHECK-SAME: branches = [@tf.[[D0:.*]], @tf.[[D1:.*]]], is_stateless = false
    // CHECK:      func private @tf.[[D0]]
    // CHECK:      %[[CST:.*]] = "tf.Const"() <{value = dense<"_dev-0-of-2">
    // CHECK:      "tf.Add"(%arg0, %[[CST]])
    // CHECK:      "2 0,1"
    // CHECK:      func private @tf.[[D1]]
    // CHECK:      %[[CST:.*]] = "tf.Const"() <{value = dense<"_dev-1-of-2">
    // CHECK:      "tf.Add"(%arg0, %[[CST]])
    // CHECK:      "2 1,1"
    %0 = "tf.Const"() {value = dense<"/dev/null"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1 = "tf.Const"() {value = dense<"t1"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %2 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %3 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    "tf.SaveV2"(%0, %1, %2, %4) : (tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<2xi32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
  func.return
}

// -----

// Check MergeV2 only happens on Device 0 for DTensor Checkpointing V1 and
// a dtensor all-reduce is introduced to the graph.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  "tf_device.cluster"() ({
    // CHECK:      tf.DTensorAllReduce
    // CHECK:      tf.NotEqual
    // CHECK:      tf.If
    // CHECK-SAME: else_branch = @tf.[[ELSE:[a-zA-Z0-9_]*]]
    // CHECK-SAME: then_branch = @tf.[[THEN:[a-zA-Z0-9_]*]]
    // CHECK:      func private @tf.[[THEN]]
    // CHECK:      tf.NoOp
    // CHECK:      func private @tf.[[ELSE]]
    // CHECK:      "tf.Const"() <{value = dense<"_dev-0-of-2">
    // CHECK:      "tf.Add"
    // CHECK:      "tf.Const"() <{value = dense<"_dev-1-of-2">
    // CHECK:      "tf.Add"
    // CHECK:      "tf.Concat"
    // CHECK:      "tf.MergeV2Checkpoints"
    %0 = "tf.Const"() {value = dense<"/dev/null/device"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1 = "tf.Const"() {value = dense<"/dev/null/destination"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    "tf.MergeV2Checkpoints"(%0, %1) {allow_missing_files = true, delete_old_dirs = false} : (tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> ()
  func.return
}


// -----

// Check DTensorRestoreV2 does local restore with slice_spec.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<!tf_type.string> {
    tf._layout = "sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg2: tensor<2x!tf_type.string> {
    tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg3: tensor<2x!tf_type.string> {
    tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg4: tensor<8x2xf32> {
    tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg5: tensor<2x4xf32> {
    tf._layout = "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> (tensor<*xf32> , tensor<*xf32> ) {
    %0:2 = "tf_device.cluster"() ({
      // CHECK:      %[[CONDITION:.*]] = "tf.Equal"(%[[LOCAL_DEVICE_IDS:.*]], %arg0)
      // CHECK:      %[[IDX_TENSOR:.*]] = "tf.Where"(%[[CONDITION]])
      // CHECK-SAME: tensor<1x1xi64>
      // CHECK:      %[[BRANCH_IDX:.*]] = "tf.Cast"(%[[IDX_TENSOR]]
      // CHECK:      "tf.Reshape"(%[[BRANCH_IDX]]
      // CHECK-SAME: (tensor<1x1xi32>, tensor<0xi32>) -> tensor<i32>
      // CHECK:      tf.Case
      // CHECK-SAME: branches = [@tf.[[D0:.*]], @tf.[[D1:.*]]], is_stateless = false
      // CHECK:      func private @tf.[[D0]]
      // CHECK:      "tf.Const"() <{value = dense<["", "2 4 0,1:-"]>
      // CHECK:      func private @tf.[[D1]]
      // CHECK:      "tf.Const"() <{value = dense<["", "2 4 1,1:-"]>
      %1 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
      %2 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<["model/r/.ATTRIBUTES/VARIABLE_VALUE", "model/s/.ATTRIBUTES/VARIABLE_VALUE"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
      %3 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<"/dev/null/ckpt-0"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %4 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<>], global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %5 = "tf.DTensorLayout"(%2) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
      %6 = "tf.DTensorLayout"(%1) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
      %7:2 = "tf.DTensorRestoreV2"(%4, %5, %6) {_global_shape = [#tf_type.shape<*>, #tf_type.shape<*>], device = "", input_dtypes = [f32, f32],
        input_shapes=[#tf_type.shape<8x2>, #tf_type.shape<2x4>],
        input_layouts=["sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
          "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]} : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xf32>)
      %8 = "tf.DTensorLayout"(%7#0) {global_shape = #tf_type.shape<*>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<*xf32>) -> tensor<*xf32>
      %9 = "tf.DTensorLayout"(%7#1) {global_shape = #tf_type.shape<*>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<*xf32>) -> tensor<*xf32>
      tf_device.return %8, %9 : tensor<*xf32>, tensor<*xf32>
    }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> (tensor<*xf32>, tensor<*xf32>)
    func.return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
  }


// -----

// Check RestoreV2 does local restore with correct shape_and_slice spec.
// Restores a replicated 8x2 tensor and a x,unsharded 2x4 tensor.
// The expansion of a RestoreV2 should be the same expansion as a
// DTensorRestoreV2.
//
// To check correctness of the expansion, we just need to check that the
// correct `shape_and_slices` constant string is produced for each
// device_id function.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<!tf_type.string> {
    tf._layout = "sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg2: tensor<2x!tf_type.string> {
    tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg3: tensor<2x!tf_type.string> {
    tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg4: tensor<*x!tf_type.resource<tensor<8x2xf32>>> {
    tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg5: tensor<*x!tf_type.resource<tensor<2x4xf32>>> {
    tf._layout = "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> (tensor<8x2xf32> , tensor<2x4xf32> ) {
    %0:2 = "tf_device.cluster"() ({
      // CHECK:      %[[CONDITION:.*]] = "tf.Equal"(%[[LOCAL_DEVICE_IDS:.*]], %arg0)
      // CHECK:      %[[IDX_TENSOR:.*]] = "tf.Where"(%[[CONDITION]])
      // CHECK-SAME: tensor<1x1xi64>
      // CHECK:      %[[BRANCH_IDX:.*]] = "tf.Cast"(%[[IDX_TENSOR]]
      // CHECK:      "tf.Reshape"(%[[BRANCH_IDX]]
      // CHECK-SAME: (tensor<1x1xi32>, tensor<0xi32>) -> tensor<i32>
      // CHECK:      tf.Case
      // CHECK-SAME: branches = [@tf.[[D0:.*]], @tf.[[D1:.*]]], is_stateless = false
      // CHECK:      func private @tf.[[D0]]
      // CHECK:      "tf.Const"() <{value = dense<["", "2 4 0,1:-"]>
      // CHECK:      func private @tf.[[D1]]
      // CHECK:      "tf.Const"() <{value = dense<["", "2 4 1,1:-"]>
      %1 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
      %2 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<["model/r/.ATTRIBUTES/VARIABLE_VALUE", "model/s/.ATTRIBUTES/VARIABLE_VALUE"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
      %3 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<"/dev/null/ckpt-0"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %4 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<>], global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
      %5 = "tf.DTensorLayout"(%2) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
      %6 = "tf.DTensorLayout"(%1) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
      %7:2 = "tf.RestoreV2"(%4, %5, %6) {input_dtypes = [f32, f32]} : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<8x2xf32>, tensor<2x4xf32>)
      "tf.AssignVariableOp"(%arg4, %7#0) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<8x2xf32>>>, tensor<8x2xf32>) -> ()
      "tf.AssignVariableOp"(%arg5, %7#1) {validate_shape = true} : (tensor<*x!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
      %8 = "tf.DTensorLayout"(%7#0) {global_shape = #tf_type.shape<*>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<8x2xf32>) -> tensor<8x2xf32>
      %9 = "tf.DTensorLayout"(%7#1) {global_shape = #tf_type.shape<*>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
      tf_device.return %8, %9 : tensor<8x2xf32>, tensor<2x4xf32>
    }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> (tensor<8x2xf32>, tensor<2x4xf32>)
    func.return %0#0, %0#1 : tensor<8x2xf32>, tensor<2x4xf32>
  }
