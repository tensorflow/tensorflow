// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-sharding-identification | FileCheck %s --dump-input=fail

// Tests empty launch func. Empty input/output sharding configuration
// attributes must be added.
// CHECK-LABEL: func @check_sharding_attrs_exists_for_empty_launch_func
func @check_sharding_attrs_exists_for_empty_launch_func() {
  "tf_device.launch_func"() {device = "", func = @empty_func, step_marker_location = ""} : () -> ()
  // CHECK: input_sharding_configuration = []
  // CHECK: output_sharding_configuration = []
  return
}

// CHECK-LABEL: func @empty_func() {
func @empty_func() {
  return
}

// -----

// Tests with a block argument inputs/outputs with no xla sharding op attached
// gets default maximal(0) sharding configuration.
// CHECK-LABEL: func @check_default_sharding_for_block_arg_inputs_outputs
func @check_default_sharding_for_block_arg_inputs_outputs(%arg0: tensor<*xi32>) {
  "tf_device.launch_func"(%arg0) {device = "", func = @func_without_sharding, step_marker_location = ""} : (tensor<*xi32>) -> ()
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  return
}

// CHECK-LABEL: func @func_without_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"})
func @func_without_sharding(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  return %arg0 : tensor<*xi32>
}

// -----

// Tests with a inputs/outputs with no xla sharding op attached gets
// default maximal(0) sharding configuration.
// CHECK-LABEL: func @check_default_sharding_for_inputs_outputs
func @check_default_sharding_for_inputs_outputs(%arg0: tensor<*xi32>) {
  "tf_device.launch_func"(%arg0) {device = "", func = @func_without_sharding, step_marker_location = ""} : (tensor<*xi32>) -> ()
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  return
}

// CHECK-LABEL: func @func_without_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"})
func @func_without_sharding(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.A"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

// Tests with a input arg connected to XlaSharding op.
// CHECK-LABEL: func @check_sharding_for_input_correctly_identified
func @check_sharding_for_input_correctly_identified(%arg0: tensor<*xi32>) {
  "tf_device.launch_func"(%arg0) {device = "", func = @inputs_with_sharding_func, step_marker_location = ""} : (tensor<*xi32>) -> ()
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\08\01\1A\01\01\22\01\00"]
  return
}

// CHECK-LABEL: func @inputs_with_sharding_func
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\01\02\03"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"})
func @inputs_with_sharding_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.A"(%0) : (tensor<*xi32>) -> (tensor<*xi32>)
  return %1 : tensor<*xi32>
}

// -----

// Tests with sharding is correctly parsed for multiple inputs/outputs.
// CHECK-LABEL: func @check_sharding_for_multiple_inputs_outputs
func @check_sharding_for_multiple_inputs_outputs(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.launch_func"(%arg0, %arg1) {device = "", func = @func_with_sharding, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
return
}

// CHECK-LABEL: func @func_with_sharding
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {xla_hlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {xla_hlo.sharding = "\0D\0E\0F"})
func @func_with_sharding(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %2, %3 = "tf.A"(%0, %1) : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %4 = "tf.XlaSharding"(%2) { _XlaSharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  return %4, %5 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following an identity op.
// CHECK-LABEL: func @check_sharding_after_identity
func @check_sharding_after_identity(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.launch_func"(%arg0, %arg1) {device = "", func = @func_with_sharding_after_identity, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  return
}

// CHECK-LABEL: func @func_with_sharding_after_identity
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {xla_hlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {xla_hlo.sharding = "\0D\0E\0F"})
func @func_with_sharding_after_identity(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.Identity"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %2 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %3, %4 = "tf.A"(%1, %2) : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  return %5, %6 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following a ReadVariable op.
// CHECK-LABEL: func @check_sharding_after_read_variable
func @check_sharding_after_read_variable(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.launch_func"(%arg0, %arg1) {device = "", func = @func_with_sharding_after_read_variable, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  return
}

// CHECK-LABEL: func @func_with_sharding_after_read_variable
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*x!tf.resource<tensor<32xf32>>> {xla_hlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*x!tf.resource<tensor<32xf32>>> {xla_hlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {xla_hlo.sharding = "\0D\0E\0F"})
func @func_with_sharding_after_read_variable(%arg0: tensor<*x!tf.resource<tensor<32xf32>>>, %arg1: tensor<*x!tf.resource<tensor<32xf32>>>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %1 = "tf.XlaSharding"(%0) { _XlaSharding = "\01\02\03" } : (tensor<32xf32>) -> tensor<32xf32>
  %2 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<32xf32>>>) -> tensor<32xf32>
  %3 = "tf.Identity"(%2) : (tensor<32xf32>) -> tensor<32xf32>
  %4 = "tf.XlaSharding"(%3) { _XlaSharding = "\04\05\06" } : (tensor<32xf32>) -> tensor<32xf32>
  %5, %6 = "tf.A"(%1, %3) : (tensor<32xf32>, tensor<32xf32>) -> (tensor<*xi32>, tensor<*xi1>)
  %7 = "tf.XlaSharding"(%5) { _XlaSharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %8 = "tf.XlaSharding"(%6) { _XlaSharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  return %7, %8 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests with input sharding following an identity op and cast op.
// CHECK-LABEL: func @check_sharding_after_cast_op
func @check_sharding_after_cast_op(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.launch_func"(%arg0, %arg1) {device = "", func = @func_with_sharding_after_cast, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  return
}

// CHECK-LABEL: func @func_with_sharding_after_cast
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {xla_hlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {xla_hlo.sharding = "\0D\0E\0F"})
func @func_with_sharding_after_cast(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %0 = "tf.Identity"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  %1 = "tf.Cast"(%0) : (tensor<*xi32>) -> tensor<*xi1>
  %2 = "tf.XlaSharding"(%1) { _XlaSharding = "\01\02\03" } : (tensor<*xi1>) -> tensor<*xi1>
  %3 = "tf.XlaSharding"(%arg1) { _XlaSharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %4, %5 = "tf.A"(%2, %3) : (tensor<*xi1>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %7 = "tf.XlaSharding"(%5) { _XlaSharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>
  return %6, %7 : tensor<*xi32> , tensor<*xi1>
}

// -----

// Tests that input sharding inside a functional op is parsed correctly.
// CHECK-LABEL: func @check_sharding_inside_functional_op
func @check_sharding_inside_functional_op(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.launch_func"(%arg0, %arg1) {device = "", func = @func_with_device_training_loop, step_marker_location = ""} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>)
  // CHECK: input_sharding_configuration
  // CHECK-SAME: ["\01\02\03", "\04\05\06"]
  // CHECK: output_sharding_configuration
  // CHECK-SAME: ["\0A\0B\0C", "\0D\0E\0F"]
  return
}

// CHECK-LABEL: func @func_with_device_training_loop
// CHECK-SAME: (%{{[a-z0-9]+}}: tensor<*xi32> {xla_hlo.sharding = "\01\02\03"}, %{{[a-z0-9]+}}: tensor<*xi1> {xla_hlo.sharding = "\04\05\06"})
// CHECK-SAME: -> (tensor<*xi32> {xla_hlo.sharding = "\0A\0B\0C"}, tensor<*xi1> {xla_hlo.sharding = "\0D\0E\0F"})
func @func_with_device_training_loop(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi1>) {
  %1:2 = "tf.StatefulPartitionedCall"(%arg0){f= @func_body, config="", config_proto="", executor_type=""}
         : (tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
  %2 = "tf.PartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_func_body} : (tensor<*xi1>) -> (tensor<i32>)
  %3, %4 = "tf.A"(%1#0, %2) : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>, tensor<*xi1>)

  %5 = "tf.XlaSharding"(%3) { _XlaSharding = "\0A\0B\0C" } : (tensor<*xi32>) -> tensor<*xi32>
  %6 = "tf.XlaSharding"(%4) { _XlaSharding = "\0D\0E\0F" } : (tensor<*xi1>) -> tensor<*xi1>

  return %5, %6 : tensor<*xi32> , tensor<*xi1>
}

// CHECK-LABEL: func @func_body
func @func_body(%arg0: tensor<*xi32>)-> (tensor<*xi32>, tensor<*xi1>) {
  %1 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\01\02\03" } : (tensor<*xi32>) -> tensor<*xi32>
  %2, %3 = "tf.C"(%1) : (tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
  return %2, %3 : tensor<*xi32> , tensor<*xi1>
}

// CHECK-LABEL: func @pcall_func_body
func @pcall_func_body(%arg0: tensor<*xi1>) -> tensor<i32> {
  %1 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\04\05\06" } : (tensor<*xi1>) -> tensor<*xi1>
  %2 = "tf.D"(%1) : (tensor<*xi1>) -> (tensor<i32>)
  return %2 : tensor<i32>
}
