// RUN: dtensor-opt %s -split-input-file -dtensor-cluster-function-conversion -verify-diagnostics | FileCheck %s

func.func @check_cluster_func_without_layout_disallowed() {
  %1 = "tf.A"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> tensor<i32>
  %2 = "tf.B"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> tensor<i32>
  // expected-error @+1 {{requires _mesh attribute}}
  %3 = "tf_device.cluster_func"(%1, %2) {func = @main_func1} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return
}

func.func @main_func1(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_layouts_retvals_attached_in_layout_op
func.func @check_layouts_retvals_attached_in_layout_op() -> tensor<i32> {
  // CHECK-NOT:       "tf_device.cluster_func"()
  // CHECK:           %[[SPC_OUT:.*]] = "tf.StatefulPartitionedCall"()
  // CHECK-SAME:      _layout = ["sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:      config = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf_device.cluster_func"() {func = @single_in_out, _mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @single_in_out() -> (tensor<i32>) {
  %0 = "tf.Const"() {_layout = ["sharding_specs:scalar, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @check_layouts_retval_attached_with_multi_in_op
func.func @check_layouts_retval_attached_with_multi_in_op(%arg0: tensor<i64>, %arg1: tensor<1xf32> {tf._layout = "sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3" }, %arg2: tensor<1xf32> {tf._layout = "mesh:CPU,x=2,y=2 layout:scalar" }) -> tensor<1xf32> {
  // CHECK-NOT:       "tf_device.cluster_func"()
  // CHECK-NEXT:      %[[SPC_OUT:.*]] = "tf.StatefulPartitionedCall"(%arg1, %arg2)
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:      config = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf_device.cluster_func"(%arg1, %arg2) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", func = @multi_in} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

func.func @multi_in(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Add"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @check_input_resource_layouts_attached_in_call_op
func.func @check_input_resource_layouts_attached_in_call_op() -> tensor<i32> {
  // CHECK-NOT:       "tf_device.cluster_func"()
  // CHECK:           %[[SPC_OUT:.*]] = "tf.StatefulPartitionedCall"()
  // CHECK-SAME:      _inferred_resource_indices = dense<1> : vector<1xi32>
  // CHECK-SAME:      _inferred_resource_layouts
  // CHECK-SAME:      "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:      config = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf_device.cluster_func"() {func = @single_in_out, _mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", _inferred_resource_indices = dense<1> : vector<1xi32>,
    _inferred_resource_layouts = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

func.func @single_in_out() -> (tensor<i32>) {
  %0 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_nested_stateful_partitioned_call
func.func @check_nested_stateful_partitioned_call() -> (tensor<i32>, tensor<i32>) {
  // CHECK-NOT:       "tf_device.cluster_func"()
  // CHECK:           %[[SPC_OUT:.*]] = "tf.StatefulPartitionedCall"()
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  // CHECK-SAME:      "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:      config = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0:2 = "tf_device.cluster_func"() {func = @nested_stateful_partitioned_call, _mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)
  func.return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

func.func @nested_stateful_partitioned_call() -> (tensor<i32>, tensor<i32>) {
  %0:2 = "tf.StatefulPartitionedCall()"() {config = "", config_proto = "", executor_type = "", f = @nested_cluster_func, _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> (tensor<i32>, tensor<i32>)
  func.return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

func.func @nested_cluster_func() -> (tensor<i32>, tensor<i32>) {
  %0:2 = "tf_device.cluster_func"() {func = @nested_func, _mesh="|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>, tensor<i32>)
  func.return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

func.func @nested_func() -> (tensor<i32>, tensor<i32>) {
   %0 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
   %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
  func.return %0, %1 : tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: func @check_var_handle_op_skip_compilation
func.func @check_var_handle_op_skip_compilation() -> tensor<!tf_type.resource<tensor<i32>>> {
  // CHECK-NOT:       "tf_device.cluster_func"()
  // CHECK:           %[[SPC_OUT:.*]] = "tf.StatefulPartitionedCall"()
  // CHECK-SAME:      _layout = ["sharding_specs: mesh:TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"]
  // CHECK-SAME:      _skip_xla_compilation = true
  // CHECK-SAME:      config = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
  %0 = "tf_device.cluster_func"() {func = @var_handle_op, _mesh="TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  func.return %0 : tensor<!tf_type.resource<tensor<i32>>>
}

func.func @var_handle_op() -> (tensor<!tf_type.resource<tensor<i32>>>) {
  %0 = "tf.VarHandleOp"() {_layout = ["sharding_specs:scalar, mesh:TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"], container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  func.return %0 : tensor<!tf_type.resource<tensor<i32>>>
}
