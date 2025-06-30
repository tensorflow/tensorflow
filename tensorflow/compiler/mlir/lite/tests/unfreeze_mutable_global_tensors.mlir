// RUN: litert-opt -unfreeze-mutable-global-tensors -split-input-file %s | FileCheck %s

module attributes {tf_saved_model.semantics} {

  // Test case: Move mutable global tensors to session initializer. Create a session initializer

  // CHECK-NOT: "tf_saved_model.global_tensor"() <{
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %c0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    func.return
  }
// CHECK: module attributes {tf_saved_model.semantics} {
// CHECK:   "tf_saved_model.session_initializer"() <{initializers = [@NoOp]}> : () -> ()
// CHECK:   func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
// CHECK:     %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %cst = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:     "tf.AssignVariableOp"(%0, %cst) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @f() attributes {tf_saved_model.exported_names = ["f"]} {
// CHECK:     %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:     "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Move mutable global tensors to session initializer. Do not create a new session initializer

  // CHECK-NOT: "tf_saved_model.global_tensor"() <{
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v2", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  "tf_saved_model.session_initializer"() <{initializers = [@NoOp]}> : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
    %cst = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}> : () -> tensor<f32>
    "tf.AssignVariableOp"(%0, %cst) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    return
  }
  func.func @f2(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v2})
  attributes {tf_saved_model.exported_names = ["f2"]} {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
    %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    "tf.AssignVariableOp"(%arg0, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    return
  }
// CHECK: module attributes {tf_saved_model.semantics} {
// CHECK:   "tf_saved_model.session_initializer"() <{initializers = [@NoOp]}> : () -> ()
// CHECK:   func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"], tf_saved_model.initializer_type = "init_op"} {
// CHECK:     %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v2"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %cst = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:     "tf.AssignVariableOp"(%0, %cst) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     %1 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %cst_0 = "tf.Const"() <{value = dense<4.200000e+01> : tensor<f32>}> : () -> tensor<f32>
// CHECK:     "tf.AssignVariableOp"(%1, %cst_0) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func @f2() attributes {tf_saved_model.exported_names = ["f2"]} {
// CHECK:     %0 = "tf.VarHandleOp"() <{container = "", shared_name = "v2"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %1 = "tf.VarHandleOp"() <{container = "", shared_name = "v"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
// CHECK:     %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:     "tf.AssignVariableOp"(%1, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
}
