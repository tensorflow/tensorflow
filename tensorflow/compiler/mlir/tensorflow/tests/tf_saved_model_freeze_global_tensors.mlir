// RUN: tf-opt -verify-diagnostics -tf-saved-model-freeze-global-tensors -split-input-file %s | FileCheck %s

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: Basic freezing.

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  // CHECK: func @f()
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non-bound inputs.
  // The pass shouldn't do anything in this case.

  // CHECK: func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Fail if mutable global tensors are found.

  // expected-error @+1 {{is not immutable}}
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    func.return
  }

}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: success if bound input user's only none ReadVariableOp instance
  // is call.

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<21.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @f_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    // CHECK: "tf.Const"() {value = dense<2.100000e+01> : tensor<f32>}
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: success if bound input user's only none ReadVariableOp instance
  // is call with read inside function

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<32.0> : tensor<f32> } : () -> ()

  func.func @g(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["g"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @g_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @g_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() {value = dense<3.200000e+01> : tensor<f32>}
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: fail if bound input user's only none ReadVariableOp instance
  // is call with write inside function

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @f_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    %0 = "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // expected-error @+1 {{immutable bound input}}
    "tf.AssignAddVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    func.return
  }
}

// -----

// expected-error @+1 {{could not freeze all global tensors in the module}}
module attributes {tf_saved_model.semantics} {

  // Test case: Fail if some global tensor ops remain

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<2.0> : tensor<f32> } : () -> ()

  func.func @f(%arg1: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v"}, %arg2: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v2"})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK-DAG: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    %0 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>

    // CHECK-DAG: "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>}
    %1 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

// Test running the pass on a module that does not have
// tf_saved_model.semantics.
// CHECK-LABEL: module
module {}
