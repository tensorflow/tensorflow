// RUN: tf-opt %s --tf-strip-saved-module-metadata --split-input-file | FileCheck %s

// CHECK-LABEL: module
// CHECK-NOT: tf_saved_model
module attributes {tf_saved_model.semantics} {
  // CHECK: tf_saved_model.global_tensor
  // CHECK-NOT: tf_saved_model
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: return
    return %arg0 : tensor<!tf_type.resource<tensor<?xf32>>>
  }
}
