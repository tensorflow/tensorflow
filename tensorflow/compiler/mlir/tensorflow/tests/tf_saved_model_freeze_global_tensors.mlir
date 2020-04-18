// RUN: tf-opt -verify-diagnostics -tf-saved-model-freeze-global-tensors -split-input-file %s | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics} {

  // Test case: Basic freezing.

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  // CHECK: func @f()
  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    return
  }
}

// -----


module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non-bound inputs.
  // The pass shouldn't do anything in this case.

  // CHECK: func @f(%arg0: tensor<!tf.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  func @f(%arg0: tensor<!tf.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Fail if mutable global tensors are found.

  // expected-error @+1 {{is not immutable}}
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Fail if bound input user is not ReadVariableOp

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // expected-error @+1 {{could not rewrite use of immutable bound input}}
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> ()
    return
  }

  func @f_callee(%arg0: tensor<!tf.resource<tensor<f32>>>) {
    return
  }
}

// -----

// expected-error @+1 {{could not freeze all global tensors in the module}}
module attributes {tf_saved_model.semantics} {

  // Test case: Fail if some global tensor ops remain

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func @f(%arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v"}, %arg2: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v2"})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: "tf.Const"()
    %0 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>

    // CHECK: "tf.Const"()
    %1 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }
}
