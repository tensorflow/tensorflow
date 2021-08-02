// RUN: tf-tfrt-opt -pass-pipeline='tfrt-lower-tf-savedmodel' -split-input-file -verify-diagnostics %s

module attributes {tf_saved_model.semantics} {

"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<i32>, value = dense<0> : tensor<i32>} : () -> ()

// Test error handling in a simple scenario.

// expected-error @+1 {{'builtin.func' op failed to promote resource variables}}
func @main(
    %arg0: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y})
      -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_basic"]} {

  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{'tf.AssignVariableOp' op unsupported users of resource variables}}
  "tf.AssignVariableOp"(%arg0, %0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  %1 = "tf.ReadVariableOp"(%arg0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  return %1 : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<i32>, value = dense<0> : tensor<i32>} : () -> ()

func private @callee(
  %arg0: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>
{
    return %arg0: tensor<!tf_type.resource<tensor<i32>>>
}

// Test error handling during recursive processing that involes a call operation.

// expected-error @+1 {{'builtin.func' op failed to promote resource variables}}
func @main(
    %arg0: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y})
      -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_basic"]} {

  %res = "tf.StatefulPartitionedCall"(%arg0)
    {config = "", config_proto = "", executor_type = "", f = @callee} :
    (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<!tf_type.resource<tensor<i32>>>)

  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{'tf.AssignVariableOp' op unsupported users of resource variables}}
  "tf.AssignVariableOp"(%res, %0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  %1 = "tf.ReadVariableOp"(%arg0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  return %1 : tensor<i32>
}

}

// -----
