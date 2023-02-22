// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check layout propagation for tf.VarHandleOp followed by Relayout.
func.func @main()  -> (tensor<2xi32>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.VarHandleOp"()
  // CHECK-SAME:      _layout = ["sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
  // CHECK:   "tf.ReadVariableOp"
  // CHECK-SAME:      _layout = ["sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
  %0 = "tf_device.cluster"() ({
    %0 = "tf.VarHandleOp"() {_global_shape = [#tf_type.shape<>], allowed_devices = [], container = "", device = "", shared_name = ""} : () -> tensor<!tf_type.resource<tensor<2xi32>>>
    %1 = "tf.Relayout"(%0) {_global_shape = [#tf_type.shape<>], device = "", layout = "sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}
    : (tensor<!tf_type.resource<tensor<2xi32>>>) -> tensor<!tf_type.resource<tensor<2xi32>>>
    %2 = "tf.ReadVariableOp"(%0) {_global_shape = [#tf_type.shape<2>], device = ""} : (tensor<!tf_type.resource<tensor<2xi32>>>) -> tensor<2xi32>
    tf_device.return %2 : tensor<2xi32>
  }) {_mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<2xi32>)
  func.return %0 : tensor<2xi32>
}

// -----


// Check layout propagation for tf.VarHandleOp without a Relayout.
func.func @main()  -> (tensor<!tf_type.resource<tensor<2xi32>>>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.VarHandleOp"()
  // CHECK-SAME:      _layout = ["empty_layout"]
  %0 = "tf_device.cluster"() ({
    %0 = "tf.VarHandleOp"() {_global_shape = [#tf_type.shape<>], allowed_devices = [], container = "", device = "", shared_name = ""} : () -> tensor<!tf_type.resource<tensor<2xi32>>>
    tf_device.return %0 : tensor<!tf_type.resource<tensor<2xi32>>>
  }) {_mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<!tf_type.resource<tensor<2xi32>>>)
  func.return %0 : tensor<!tf_type.resource<tensor<2xi32>>>
}