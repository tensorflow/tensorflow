// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-sharding-identification | FileCheck %s

// Tests empty cluster func. Empty input/output sharding configuration
// attributes must be added.
// CHECK-LABEL: func @check_sharding_attrs_exists_for_empty_cluster_func
func.func @check_sharding_attrs_exists_for_empty_cluster_func() {
  "tf_device.cluster_func"() {func = @empty_func, step_marker_location = ""} : () -> ()
  // CHECK: input_sharding_configuration = []
  // CHECK: output_sharding_configuration = []
  func.return
}

// CHECK-LABEL: func @empty_func() {
func.func @empty_func() {
  func.return
}

// -----

// Tests with a block argument inputs/outputs with no xla sharding op attached
// gets default maximal(0) sharding configuration.
// CHECK-LABEL: func @check_default_sharding_for_block_arg_inputs_outputs
func.func @check_default_sharding_for_block_arg_inputs_outputs(%arg0: tensor<*xi32>) {
  "tf_device.cluster_func"(%arg0) {func = @func_without_sharding, step_marker_location = ""} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  func.return
}

// CHECK-LABEL: func @func_without_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
func.func @func_without_sharding(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  func.return %arg0 : tensor<*xi32>
}

// -----

// Tests with a inputs/outputs with no xla sharding op attached gets
// default maximal(0) sharding configuration.
// CHECK-LABEL: func @check_default_sharding_for_inputs_outputs
func.func @check_default_sharding_for_inputs_outputs(%arg0: tensor<*xi32>) {
  "tf_device.cluster_func"(%arg0) {func = @func_without_sharding, step_marker_location = ""} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  func.return
}

// CHECK-LABEL: func @func_without_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
func.func @func_without_sharding(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.A"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// Tests with a input arg connected to XlaSharding op.
// CHECK-LABEL: func @check_sharding_for_input_correctly_identified
func.func @check_sharding_for_input_correctly_identified(%arg0: tensor<*xi32>) {
  "tf_device.cluster_func"(%arg0) {func = @inputs_with_sharding_func, step_marker_location = ""} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  func.return
}

// CHECK-LABEL: func @inputs_with_sharding_func
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
func.func @inputs_with_sharding_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.A"(%0) : (tensor<*xi32>) -> (tensor<*xi32>)
  func.return %1 : tensor<*xi32>
}

// -----

// Tests with sharding is correctly parsed for multiple inputs/outputs.
// CHECK-LABEL: func @check_sharding_for_multiple_inputs_outputs
func.func @check_sharding_for_multiple_inputs_outputs(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_sharding, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  func.return
}

// CHECK-LABEL: func @func_with_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {mhlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {mhlo.sharding = "\0D\0E\0F"})
func.func @func_with_sharding(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06", sharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %2, %3 = "tf.A"(%0, %1) : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %4 = "tf.XlaSharding"(%2) { _XlaSharding = "\0A\0B\0C", sharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0D\0E\0F", sharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  func.return %4, %5 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following an identity op.
// CHECK-LABEL: func @check_sharding_after_identity
func.func @check_sharding_after_identity(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_sharding_after_identity, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  func.return
}

// CHECK-LABEL: func @func_with_sharding_after_identity
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {mhlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {mhlo.sharding = "\0D\0E\0F"})
func.func @func_with_sharding_after_identity(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.Identity"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %2 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06", sharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %3, %4 = "tf.A"(%1, %2) : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0A\0B\0C", sharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0D\0E\0F", sharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  func.return %5, %6 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following a ReadVariable op.
// CHECK-LABEL: func @check_sharding_after_read_variable
func.func @check_sharding_after_read_variable(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_sharding_after_read_variable, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  func.return
}

// CHECK-LABEL: func @func_with_sharding_after_read_variable
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*x!tf_type.resource<tensor<32xf32>>> {mhlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*x!tf_type.resource<tensor<32xf32>>> {mhlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {mhlo.sharding = "\0D\0E\0F"})
func.func @func_with_sharding_after_read_variable(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>, %arg1: tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<32xf32>) -> tensor<32xf32>
  %2 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %3 = "tf.Identity"(%2) : (tensor<32xf32>) -> tensor<32xf32>
  %4 = "tf.XlaSharding"(%3) { _XlaSharding = "\04\05\06", sharding = "\04\05\06" } : (tensor<32xf32>) -> tensor<32xf32>
  %5, %6 = "tf.A"(%1, %3) : (tensor<32xf32>, tensor<32xf32>) -> (tensor<*xi32>, tensor<*xi1>)
  %7 = "tf.XlaSharding"(%5) { _XlaSharding = "\0A\0B\0C", sharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %8 = "tf.XlaSharding"(%6) { _XlaSharding = "\0D\0E\0F", sharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  func.return %7, %8 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following an identity op and cast op.
// CHECK-LABEL: func @check_sharding_after_cast_op
func.func @check_sharding_after_cast_op(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_sharding_after_cast, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  func.return
}

// CHECK-LABEL: func @func_with_sharding_after_cast
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {mhlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {mhlo.sharding = "\0D\0E\0F"})
func.func @func_with_sharding_after_cast(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.Identity"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.Cast"(%0) : (tensor<*xi32>) -> tensor<*xi1>
  %2 = "tf.XlaSharding"(%1) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi1>) -> tensor<*xi1>
  %3 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06", sharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %4, %5 = "tf.A"(%2, %3) : (tensor<*xi1>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0A\0B\0C", sharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %7 = "tf.XlaSharding"(%5) { _XlaSharding = "\0D\0E\0F", sharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  func.return %6, %7 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests that input sharding inside a functional op is parsed correctly.
// CHECK-LABEL: func @check_sharding_inside_functional_op
func.func @check_sharding_inside_functional_op(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.cluster_func"(%arg0, %arg1) {func = @func_with_device_training_loop, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  func.return
}

// CHECK-LABEL: func @func_with_device_training_loop
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {mhlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {mhlo.sharding = "\0D\0E\0F"})
func.func @func_with_device_training_loop(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %1:2 = "tf.StatefulPartitionedCall"(%arg0){f= @func_body, config="", config_proto="", executor_type=""}
         : (tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
  %2 = "tf.PartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_func_body} : (tensor<*xi1>) -> (tensor<i32>)
  %3, %4 = "tf.A"(%1#0, %2) : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, tensor<*xi1>)

  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0A\0B\0C", sharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0D\0E\0F", sharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>

  func.return %5, %6 : tensor<*xi32> , tensor<*xi1>
}

// CHECK-LABEL: func @func_body
func.func @func_body(%arg0: tensor<*xi32>)-> (tensor<*xi32>, tensor<*xi1>) {
  %1 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %2, %3 = "tf.C"(%1) : (tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
  func.return %2, %3 : tensor<*xi32> , tensor<*xi1>
}

// CHECK-LABEL: func @pcall_func_body
func.func @pcall_func_body(%arg0: tensor<*xi1>) -> tensor<i32> {
  %1 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\04\05\06", sharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %2 = "tf.D"(%1) : (tensor<*xi1>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

// -----

// Tests that output sharding inside a functional op is parsed correctly.

// CHECK-LABEL: func @check_sharding_inside_functional_op
func.func @check_sharding_inside_functional_op(%arg0: tensor<*xi32>) {
  "tf_device.cluster_func"(%arg0) {func = @cluster_func, step_marker_location = ""} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\01\02\03"]
  func.return
}

func.func @cluster_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.PartitionedCall"(%arg0) {f= @func_body, config="", config_proto="", executor_type=""} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

func.func @func_body(%arg0: tensor<*xi32>)-> tensor<*xi32> {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03", sharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.Identity"(%0) : (tensor<*xi32>) -> (tensor<*xi32>)
  func.return %1 : tensor<*xi32>
}

// -----

// Tests partitioned data inputs/outputs are set correctly (via XLA SPMD) is
// enabled. Non replicated inputs/outputs should have shardings set to be
// replicate sharding ("").

// CHECK-LABEL: func @partitioned_input_output
func.func @partitioned_input_output(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %0 = "tf.TPUPartitionedInput"(%arg0) {_XlaSharding = "\01\02\03", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\01\02\03", ""]
  // CHECK-SAME: output_sharding_configuration = ["", "\04\05\06"]
  %1:2 = "tf_device.cluster_func"(%0, %arg1) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  %2 = "tf.TPUPartitionedOutput"(%1#1) {_XlaSharding = "\04\05\06", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %1#0, %2 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xi32> {mhlo.sharding = "\01\02\03"}, {{.+}}: tensor<*xi32> {mhlo.sharding = ""})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = ""}, tensor<*xi32> {mhlo.sharding = "\04\05\06"})
func.func @cluster_func(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  func.return %arg0, %arg1 : tensor<*xi32>, tensor<*xi32>
}

// -----

// Tests partitioned variables (via XLA SPMD) propagates shardings correctly.

// CHECK-LABEL: func @partitioned_variable
func.func @partitioned_variable(%arg0: tensor<!tf_type.resource<tensor<*xf32>>>) {
  %0 = "tf.TPUPartitionedInput"(%arg0) {_XlaSharding = "\01\02\03", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<*xf32>>>) -> tensor<!tf_type.resource<tensor<*xf32>>>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\01\02\03"]
  // CHECK-SAME: output_sharding_configuration = []
  "tf_device.cluster_func"(%1) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xf32>) -> ()
  func.return
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xf32> {mhlo.sharding = "\01\02\03"})
func.func @cluster_func(%arg0: tensor<*xf32>) {
  func.return
}

// -----

// Tests that device variable sharding defaults to xla.OpSharding
// { type : MAXIMAL
//   tile_assignment_dimensions: [ 1 ]
//   tile_assignment_devices   : [ 0 ]
// }

// CHECK-LABEL: func @maximal_device_variable
func.func @maximal_device_variable(%arg0: tensor<*x!tf_type.resource<tensor<*xf32>>>) {
   tf_device.replicate(%arg0 as %arg1: tensor<*x!tf_type.resource<tensor<*xf32>>>)
     {_mirrored_variable_indices = [0], n = 2 : i32} {
     %0 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
     // CHECK:      tf_device.cluster_func
     // CHECK-SAME: input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"]
     "tf_device.cluster_func"(%0) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xf32>) -> ()
     tf_device.return
  }
  func.return
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
func.func @cluster_func(%arg0: tensor<*xf32>) {
  func.return
}

// -----

// Tests that device variable sharding for an implicitly capture device variable
// defaults to REPLICATE.

// CHECK-LABEL: func @replicated_device_variable
func.func @replicated_device_variable(%arg0: tensor<*x!tf_type.resource<tensor<*xf32>>>) {
   %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<*xf32>
   tf_device.replicate()
     {n = 2 : i32} {
     // CHECK:      tf_device.cluster_func
     // CHECK-SAME: input_sharding_configuration = [""]
     "tf_device.cluster_func"(%0) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xf32>) -> ()
     tf_device.return
  }
  func.return
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xf32> {mhlo.sharding = ""})
func.func @cluster_func(%arg0: tensor<*xf32>) {
  func.return
}

// -----

// Tests partitioned inputs/outputs with no sharding (via XLA SPMD) defaults to
// replicate sharding ("").

// CHECK-LABEL: func @partitioned_input_output
func.func @partitioned_input_output(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.TPUPartitionedInput"(%arg0) {partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = [""]
  // CHECK-SAME: output_sharding_configuration = [""]
  %1 = "tf_device.cluster_func"(%0) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xi32>) -> tensor<*xi32>
  %2 = "tf.TPUPartitionedOutput"(%1) {partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %2 : tensor<*xi32>
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xi32> {mhlo.sharding = ""})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = ""})
func.func @cluster_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  func.return %arg0 : tensor<*xi32>
}

// -----

// Tests output sharding of unpartitioned resource write takes on same sharding
// as unpartitioned resource.

// CHECK-LABEL: func @partitioned_input_output
func.func @partitioned_input_output(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  %0 = "tf.TPUPartitionedInput"(%arg0) {_XlaSharding = "\01\02\03", partition_dim = -1 : i64} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<!tf_type.resource<tensor<f32>>>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = []
  // CHECK-SAME: output_sharding_configuration = ["\01\02\03"]
  %1 = "tf_device.cluster_func"() {func = @cluster_func, use_spmd_for_xla_partitioning = true} : () -> tensor<f32>
  "tf.AssignVariableOp"(%0, %1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  func.return
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: -> (tensor<f32> {mhlo.sharding = "\01\02\03"})
func.func @cluster_func() -> tensor<f32> {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Tests if any of inputs have MAXIMAL sharding, then revert to MPMD.

// CHECK-LABEL: func @partitioned_input_maximal_sharding_revert_mpmd
func.func @partitioned_input_maximal_sharding_revert_mpmd(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %0 = "tf.TPUPartitionedInput"(%arg0) {_XlaSharding = "\08\01\1A\01\01\22\01\00", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"]
  // CHECK-SAME: output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\04\05\06"]
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  %1:2 = "tf_device.cluster_func"(%0, %arg1) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  %2 = "tf.TPUPartitionedOutput"(%1#1) {_XlaSharding = "\04\05\06", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %1#0, %2 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, {{.+}}: tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<*xi32> {mhlo.sharding = "\04\05\06"})
func.func @cluster_func(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  func.return %arg0, %arg1 : tensor<*xi32>, tensor<*xi32>
}

// -----

// Tests if any of outputs have MAXIMAL sharding, then revert to MPMD.

// CHECK-LABEL: func @partitioned_output_maximal_sharding_revert_mpmd
func.func @partitioned_output_maximal_sharding_revert_mpmd(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\04\05\06", "\08\01\1A\01\01\22\01\00"]
  // CHECK-SAME: output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"]
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  %0 = "tf.TPUPartitionedInput"(%arg0) {_XlaSharding = "\04\05\06", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  %1:2 = "tf_device.cluster_func"(%0, %arg1) {func = @cluster_func, use_spmd_for_xla_partitioning = true} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  %2 = "tf.TPUPartitionedOutput"(%1#1) {_XlaSharding = "\08\01\1A\01\01\22\01\00", partition_dim = -1 : i64} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %1#0, %2 : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: func @cluster_func
// CHECK-SAME: ({{.+}}: tensor<*xi32> {mhlo.sharding = "\04\05\06"}, {{.+}}: tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<*xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"})
func.func @cluster_func(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  func.return %arg0, %arg1 : tensor<*xi32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: func @check_propagation_downwards_with_alias
func.func @check_propagation_downwards_with_alias(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\01\02\03", "\04\05\06"]
  // CHECK-SAME: output_sharding_configuration = ["\04\05\06", "\01\02\03"]
  "tf_device.cluster_func"(%arg0, %arg1) {
      func = @func,
      use_spmd_for_xla_partitioning = false
  } : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return
}

// CHECK-LABEL: func @func
// CHECK-SAME: %arg0: tensor<*xi32> {mhlo.sharding = "\01\02\03"
// CHECK-SAME: %arg1: tensor<*xi32> {mhlo.sharding = "\04\05\06"
// CHECK-SAME: ->{{.*}}mhlo.sharding = "\04\05\06"{{.*}}mhlo.sharding = "\01\02\03"
func.func @func(%arg0: tensor<*xi32> {tf.aliasing_output = 1 : i64},
           %arg1: tensor<*xi32> {tf.aliasing_output = 0 : i64}) -> (tensor<*xi32>, tensor<*xi32>) {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03"} : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06"} : (tensor<*xi32>) -> tensor<*xi32>
  // flip order
  %2 = "tf.A"(%1) : (tensor<*xi32>) -> (tensor<*xi32>)
  %3 = "tf.B"(%0) : (tensor<*xi32>) -> (tensor<*xi32>)
  func.return %2, %3 : tensor<*xi32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: func @check_arg_sharding_errors
func.func @check_arg_sharding_errors(%arg0: tensor<1x2x3xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"]
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  "tf_device.cluster_func"(%arg0) {func = @func} : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  func.return
}

func.func @func(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
  // Use a four dimension sharding (devices=[1,1,1,1]0)
  // Since the input tensor only has three dimensions, we expect this to fail.
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\08\03\1A\04\01\01\01\01\22\01\00" } : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  %1 = "tf.A"(%0) : (tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>)
  func.return %1: tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @check_retval_sharding_errors
func.func @check_retval_sharding_errors(%arg0: tensor<1x2x3xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"]
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  "tf_device.cluster_func"(%arg0) {func = @func} : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  func.return
}

func.func @func(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
  %0 = "tf.A"(%arg0) : (tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>)
  // Use a four dimension sharding (devices=[1,1,1,1]0)
  // Since the output tensor only has three dimensions, we expect this to fail.
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\08\03\1A\04\01\01\01\01\22\01\00" } : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  func.return %1: tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @check_propagation_upwards_when_spmd_for_xla_is_true
func.func @check_propagation_upwards_when_spmd_for_xla_is_true(%arg0: tensor<*xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = ["\01\02\03"]
  // CHECK-SAME: output_sharding_configuration = [""]
  "tf_device.cluster_func"(%arg0) {
      func = @func,
      use_spmd_for_xla_partitioning = true
  } : (tensor<*xi32>) -> tensor<*xi32>
  func.return
}

func.func @func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03"} : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.A"(%0) : (tensor<*xi32>) -> (tensor<*xi32>)
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @check_propagation_downwards_when_spmd_for_xla_is_true
func.func @check_propagation_downwards_when_spmd_for_xla_is_true(%arg0: tensor<*xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: input_sharding_configuration = [""]
  // CHECK-SAME: output_sharding_configuration = ["\01\02\03"]
  "tf_device.cluster_func"(%arg0) {
      func = @func,
      use_spmd_for_xla_partitioning = true
  } : (tensor<*xi32>) -> tensor<*xi32>
  func.return
}

func.func @func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.A"(%arg0) : (tensor<*xi32>) -> (tensor<*xi32>)
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\01\02\03"} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @check_propagation_downwards_through_ops
func.func @check_propagation_downwards_through_ops(%arg0: tensor<4xf32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: output_sharding_configuration = ["\01\02\03"]
  "tf_device.cluster_func"(%arg0) {
      func = @func,
      use_spmd_for_xla_partitioning = false
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return
}

// CHECK-LABEL: func @func
// CHECK-SAME: ->{{.*}}mhlo.sharding = "\01\02\03"
func.func @func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = "tf.Const"() {value = dense<23.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03"} : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "tf.AddV2"(%0, %cst) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  %2 = "tf.AddV2"(%cst, %1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  %3 = "tf.Mul"(%2, %cst) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  %4 = "tf.Mul"(%cst, %3) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  %5 = "tf.Sub"(%4, %cst) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  %6 = "tf.Sub"(%cst, %5) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  func.return %6 : tensor<4xf32>
}

// -----
// CHECK-LABEL: func @check_propagation_for_output_sharding_from_tf_matmul
// CHECK:      tf_device.cluster_func
// CHECK-SAME: input_sharding_configuration = ["", ""]
// CHECK-SAME: output_sharding_configuration = ["\08\03\1A\02\02\01\22\02\00\01"]
func.func @check_propagation_for_output_sharding_from_tf_matmul(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %0 = "tf_device.cluster_func"(%arg0, %arg1) {func = @_func, use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  %1:2 = "tf.TPUPartitionedOutput"(%0) {device = "", partition_dim = 0 : i64} : (tensor<2x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  return %1#0, %1#1 : tensor<1x2xf32>, tensor<1x2xf32>
}
func.func @_func(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.MatMul"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01"} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----
// CHECK-LABEL: func @check_propagation_for_output_sharding_from_tf_matmul_following_by_identity_op
// CHECK:      tf_device.cluster_func
// CHECK-SAME: input_sharding_configuration = ["", ""]
// CHECK-SAME: output_sharding_configuration = ["\08\03\1A\02\02\01\22\02\00\01"]
func.func @check_propagation_for_output_sharding_from_tf_matmul_following_by_identity_op(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %0 = "tf_device.cluster_func"(%arg0, %arg1) {func = @_func, use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  %1:2 = "tf.TPUPartitionedOutput"(%0) {device = "", partition_dim = 0 : i64} : (tensor<2x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  return %1#0, %1#1 : tensor<1x2xf32>, tensor<1x2xf32>
}
func.func @_func(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.MatMul"(%arg0, %arg1) {_XlaSharding = "\08\03\1A\02\02\01\22\02\00\01"} : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  %1 = "tf.Identity"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
