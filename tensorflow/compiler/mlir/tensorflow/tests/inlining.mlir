// RUN: tf-opt %s -inline='default-pipeline=''' | FileCheck %s

func private @simple_callee() -> tensor<2xi32>  {
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}

func private @simple_callee_with_noinline() -> tensor<2xi32> attributes {tf._noinline = true} {
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}

// Test that simple TF operations can be inlined.

// CHECK-LABEL: func @inline_simple(
func @inline_simple() -> tensor<2xi32> {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: return %[[CST]]
  %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee} : () -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Test that functions with 'tf._noinline' are not inlined.

// CHECK-LABEL: func @dont_inline_func_with_noinline_attribute(
func @dont_inline_func_with_noinline_attribute() -> tensor<2xi32> {
  // CHECK-NEXT: %[[PARTITIONED_CALL:.*]] = "tf.PartitionedCall"
  // CHECK-NEXT: return %[[PARTITIONED_CALL]]
  %result = "tf.PartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee_with_noinline} : () -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Test that TPUPartitionedCallOp is not inlined.


// CHECK-LABEL: func @dont_inline_tpu_partitioned_call(
func @dont_inline_tpu_partitioned_call() -> tensor<2xi32> {
  // CHECK-NEXT: %[[ORDINAL:.*]] = "tf.TPUOrdinalSelector"
  // CHECK-NEXT: %[[PARTITIONED_CALL:.*]] = "tf.TPUPartitionedCall"(%[[ORDINAL]])
  // CHECK-NEXT: return %[[PARTITIONED_CALL]]
  %0 = "tf.TPUOrdinalSelector"() {device = ""} : () -> tensor<?xi32>
  %result = "tf.TPUPartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @simple_callee} : (tensor<?xi32>) -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Check that TF call operations can be inlined, even when the shape of the
// argument or result is different than the called function.

func private @inline_shape_cast_callee(%arg : tensor<*xi32>) -> tensor<*xi32>  {
  return %arg : tensor<*xi32>
}

func private @custom_callee() -> tensor<2xi32>  {
  %0 = "tf.CustomTFOp"() : () -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// Test that unregistered user-defined custom TF operations can not be inlined
// when there are duplicated cases.

// CHECK-LABEL: func @dont_inline_custom_on_duplicated_cases(
func @dont_inline_custom_on_duplicated_cases() -> tensor<2xi32> {
  // CHECK-NEXT: "tf.PartitionedCall"
  // CHECK-NEXT: "tf.PartitionedCall"
  // CHECK-NEXT: return
  %0 = "tf.PartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @custom_callee} : () -> tensor<2xi32>
  %1 = "tf.PartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @custom_callee} : () -> tensor<2xi32>
  return %1: tensor<2xi32>
}

// CHECK-LABEL: func @inline_shape_cast(
// CHECK-SAME:                          %[[ARG:.*]]: tensor<2xi32>
func @inline_shape_cast(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: %[[ARG_CAST:.*]] = "tf.Cast"(%[[ARG]]) {Truncate = false} : (tensor<2xi32>) -> tensor<*xi32>
  // CHECK-NEXT: %[[RESULT_CAST:.*]] = "tf.Cast"(%[[ARG_CAST]]) {Truncate = false} : (tensor<*xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[RESULT_CAST]]
  %result = "tf.PartitionedCall"(%arg) {config = "", config_proto = "", executor_type = "", f = @inline_shape_cast_callee} : (tensor<2xi32>) -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Test that functions can be inlined into tf_device regions.

// CHECK-LABEL: func @inline_simple_tf_device_region(
func @inline_simple_tf_device_region() -> tensor<2xi32> {
  // CHECK-NEXT: "tf_device.cluster"()
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: tf_device.return %[[CST]]
  %cluster_result = "tf_device.cluster"() ( {
    %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee} : () -> tensor<2xi32>
    tf_device.return %result : tensor<2xi32>
  }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> (tensor<2xi32>)
  return %cluster_result : tensor<2xi32>
}


// Check that functions can be inlined into islands.

func private @inline_into_island_multi_block_callee() -> tensor<2xi32>  {
  br ^bb1

^bb1:
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: func @inline_into_island(
func @inline_into_island() -> (tensor<2xi32>, tensor<2xi32>) {
  %0:2 = tf_executor.graph {
    %1:3 = tf_executor.island {
      // Single block regions may be inlined.
      // CHECK: %[[CST:.*]] = "tf.Const"
      %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee} : () -> tensor<2xi32>

      // Multi block regions may not.
      // CHECK-NEXT: %[[CALL:.*]] = "tf.StatefulPartitionedCall"
      %result_2 = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @inline_into_island_multi_block_callee} : () -> tensor<2xi32>

      // CHECK-NEXT: tf_executor.yield %[[CST]], %[[CALL]]
      tf_executor.yield %result, %result_2 : tensor<2xi32>, tensor<2xi32>
    }
    tf_executor.fetch %1#1, %1#1 : tensor<2xi32>, tensor<2xi32>
  }
  return %0#1, %0#1 : tensor<2xi32>, tensor<2xi32>
}

// Test that stateful TF ops that don't have do not duplicate trait can be
// inlined.

func private @simple_callee_var() -> tensor<2xi32>  {
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<2xi32>>>
  "tf.AssignVariableOp"(%0, %cst) {device = ""} : (tensor<!tf_type.resource<tensor<2xi32>>>, tensor<2xi32>) -> ()
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: func @inline_simple_var(
func @inline_simple_var() -> tensor<2xi32> {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[VAR:.*]] = "tf.VarHandleOp"
  // CHECK-NEXT: "tf.AssignVariableOp"(%[[VAR]], %[[CST]]
  // CHECK-NEXT: return %[[CST]]
  %result = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee_var} : () -> tensor<2xi32>
  return %result : tensor<2xi32>
}

// Test that simple TF operations can be inlined with devices assigned.

func private @simple_callee_with_devices() -> (tensor<2xi32>, tensor<2xf32>)  {
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  %cst_gpu = "tf.Const"() { value = dense<2.> : tensor<2xf32>, device = "/GPU:0" } : () -> tensor<2xf32>
  return %cst, %cst_gpu : tensor<2xi32>, tensor<2xf32>
}
// CHECK-LABEL: func @inline_simple_with_devices(
func @inline_simple_with_devices() -> tensor<2xi32> {
  // CHECK-DAG: %[[CST:.*]] = "tf.Const"{{.*}}CPU{{.*}}i32
  // CHECK-DAG: "tf.Const"{{.*}}GPU{{.*}}f32
  // CHECK: return %[[CST]]
  %result:2 = "tf.StatefulPartitionedCall"() {config = "", config_proto = "", executor_type = "", f = @simple_callee_with_devices, device = "/CPU:0"} : () -> (tensor<2xi32>, tensor<2xf32>)
  return %result#0 : tensor<2xi32>
}
