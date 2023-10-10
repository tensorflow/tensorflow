// RUN: tf-opt %s --allow-unregistered-dialect --tf-saved-model-lower-globals-to-mlprogram --split-input-file | FileCheck %s

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global private mutable [[V:@.+]](dense<1.000000e+00> : tensor<1xf32>) : tensor<?xf32>
  // CHECK: func.func @f
  // CHECK: [[T:%.+]] = ml_program.global_load [[V]]
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<?xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global private mutable [[V:@.+]](dense<1.000000e+00> : tensor<1xf32>) : tensor<?xf32>
  // CHECK: func.func @f(%arg0: tensor<?xf32>
  // CHECK: ml_program.global_store [[V]] = %arg0 : tensor<?xf32>
  // CHECK-NEXT: return
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.AssignVariableOp"(%arg1, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL:module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global private mutable [[V:@.+]](dense<1.000000e+00> : tensor<1xf32>) : tensor<?xf32>
  // CHECK: func.func @f(%arg0: tensor<?xf32>
  // CHECK: ml_program.global_store [[V]] = %arg0 : tensor<?xf32>
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    cf.br ^bb1(%arg1 : tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%r: tensor<!tf_type.resource<tensor<?xf32>>>):
    "tf.AssignVariableOp"(%r, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global private mutable [[V:@.+]](dense<1.000000e+00> : tensor<1xf32>) : tensor<?xf32>
  // CHECK: func.func @f(%arg0: tensor<?xf32>
  // CHECK: ml_program.global_store [[V]]
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}) attributes {tf_saved_model.exported_names = ["f"]} {
    cf.br ^bb1(%arg1, %arg1, %arg1 : tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%0: tensor<!tf_type.resource<tensor<?xf32>>>, %1: tensor<!tf_type.resource<tensor<?xf32>>>, %2: tensor<!tf_type.resource<tensor<?xf32>>>):
    "tf.AssignVariableOp"(%0, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    cf.br ^bb1(%1, %2, %0 : tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>, tensor<!tf_type.resource<tensor<?xf32>>>)
  }
}
