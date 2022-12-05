// RUN: tf-opt --allow-unregistered-dialect --split-input-file --tf-saved-model-lower-variable-ops-to-mlprogram %s | FileCheck %s

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global {{.*}} @vars.v
  // CHECK: ml_program.global {{.*}} @vars.bar
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<10xf32>, value = dense<[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  func.func @read(%arg0: tensor<!tf_type.resource<tensor<10xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<10xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["read"]} {
    // CHECK: %[[v:.*]] = ml_program.global_load @vars.v : tensor<10xf32>
    // CHECK: %[[bar:.*]] = ml_program.global_load @vars.bar : tensor<1xf32>
    // CHECK: %[[sum:.*]] = "tf.Add"(%[[v]], %[[bar]])
    // CHECK: return %[[sum]]
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "bar"} : () -> tensor<!tf_type.resource<tensor<1xf32>>>
    %2 = "tf.ReadVariableOp"(%1) : (tensor<!tf_type.resource<tensor<1xf32>>>) -> tensor<1xf32>
    %3 = "tf.Add"(%0, %2) : (tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global
  // CHECK-NOT: mutable
  // CHECK-SAME: vars.v
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  func.func @read_twice() -> (tensor<10xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["read_twice"]} {
    // CHECK: %[[v1:.*]] = ml_program.global_load @vars.v : tensor<10xf32>
    // CHECK: %[[v2:.*]] = ml_program.global_load @vars.v : tensor<10xf32>
    // CHECK: "tf.Mul"(%[[v1]], %[[v2]])
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    %3 = "tf.Mul"(%1, %2) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global{{.*}}mutable{{.*}}@vars.v
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  func.func @assign_twice(%arg0: tensor<!tf_type.resource<tensor<10xf32>>> {tf_saved_model.bound_input = @v})
  -> ()
  attributes {tf_saved_model.exported_names = ["assign_twice"]} {
    // CHECK: ml_program.global_store @vars.v
    // CHECK: ml_program.global_store @vars.v
    %0 = "tf.Const"() {type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32>} : () -> tensor<10xf32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
    %2 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<1xf32>>>
    "tf.AssignVariableOp"(%2, %0) : (tensor<!tf_type.resource<tensor<1xf32>>>, tensor<10xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global{{.*}}mutable [[V:@.+]]({{.*}}) : tensor<3xf32>
  // CHECK: func.func @assign_inserts_cast_if_necessary(%arg0: tensor<?xf32>
  // CHECK: ml_program.global_load [[V]]
  // CHECK: [[C:%.*]] = "tf.Cast"(%arg0
  // CHECK: ml_program.global_store [[V]] = [[C]]
  func.func @assign_inserts_cast_if_necessary(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<3xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>
    %2 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<?xf32>>>
    "tf.AssignVariableOp"(%2, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global{{.*}}[[V:@.+]]({{.*}}) : tensor<3xf32>
  // CHECK: func.func @read_inserts_cast_if_necessary(%arg0: tensor<?xf32>
  // CHECK: [[V1:%.*]] = ml_program.global_load
  // CHECK: [[V2:%.*]] = ml_program.global_load
  // CHECK: "tf.Cast"([[V2]]
  func.func @read_inserts_cast_if_necessary(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<3xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>
    %2 = "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<?xf32>>>
    %3 = "tf.ReadVariableOp"(%2) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK: ml_program.global
  // CHECK-NOT: mutable
  // CHECK-SAME: vars.v(dense<[0{{.*}}1{{.*}}2{{.*}}3{{.*}}4{{.*}}5{{.*}}6{{.*}}7{{.*}}8{{.*}}9
  "tf_saved_model.global_tensor"() {
      is_mutable, sym_name = "v", type = tensor<10xf32>,
          value = dense<[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  func.func @preserves_constants(%arg0: tensor<!tf_type.resource<tensor<10xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<10xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["read"]} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}
