
// RUN: tf-opt %s -allow-unregistered-dialect --tf-localize-var-handles --split-input-file | FileCheck %s

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<10xf32>, value = dense<[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  // CHECK-LABEL: @read_from_global
  func.func @read_from_global(%arg0: tensor<!tf_type.resource<tensor<10xf32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["read_from_global"]} {
    // CHECK: [[name:%.*]] = "tf.VarHandleOp"
    // CHECK: "tf.ReadVariableOp"([[name]])
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<10xf32>, value = dense<[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32> } : () -> ()
  // CHECK-LABEL: @assign_from_global
  func.func @assign_from_global(%arg0: tensor<!tf_type.resource<tensor<10xf32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["assign_from_global"]} {
    // CHECK: [[name:%.*]] = "tf.VarHandleOp"
    // CHECK: "tf.AssignVariableOp"([[name]]
    %0 = "tf.Const"() {type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32>} : () -> tensor<10xf32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: @read_resource
  func.func private @read_resource(%arg0: tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32> {
    // CHECK: [[name:%.*]] = "tf.VarHandleOp"
    // CHECK: "tf.ReadVariableOp"([[name]]
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }

  // CHECK-LABEL: @read_using_function
  func.func @read_using_function()
  attributes {tf_saved_model.exported_names = ["read_using_function"]} {
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "foo"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    %1 = func.call @read_resource(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<10xf32>
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: @assign_resource
  func.func private @assign_resource(%arg0: tensor<!tf_type.resource<tensor<10xf32>>>) {
    // CHECK: [[name:%.*]] = "tf.VarHandleOp"
    // CHECK: "tf.AssignVariableOp"([[name]]
    %0 = "tf.Const"() {type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32>} : () -> tensor<10xf32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
    return
  }

  // CHECK-LABEL: @assign_using_function
  func.func @assign_using_function()
  attributes {tf_saved_model.exported_names = ["assign_using_function"]} {
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "foo"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    func.call @assign_resource(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: @handles_gaps_in_dataflow
  func.func @handles_gaps_in_dataflow()
  attributes {tf_saved_model.exported_names = ["handles_gaps_in_dataflow"]} {
    // CHECK: VarHandleOp
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "foo"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    %1 = "tf.Const"() {type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32>} : () -> tensor<10xf32>
    // CHECK: [[handle:%.*]] = "foo.bar"
    %2 = "foo.bar"(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> tensor<!tf_type.resource<tensor<10xf32>>>
    // CHECK-NOT: AssignVariableOp
    // CHECK: "tf.AssignVariableOp"([[handle]]
    "tf.AssignVariableOp"(%2, %1) : (tensor<!tf_type.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module
module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: @assign_resource
  func.func private @assign_resource(%arg0: tensor<!tf_type.resource<tensor<10xf32>>>) {
    // CHECK-NOT: VarHandleOp
    // CHECK: "tf.AssignVariableOp"(
    %0 = "tf.Const"() {type = tensor<10xf32>, value = dense<[0.,10.,2.,3.,4.,5.,6.,7.,8.,9.]> : tensor<10xf32>} : () -> tensor<10xf32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<10xf32>>>, tensor<10xf32>) -> ()
    return
  }

  // CHECK-LABEL: @handles_ambiguous_var_handles
  func.func @handles_ambiguous_var_handles()
  attributes {tf_saved_model.exported_names = ["handles_ambiguous_var_handles"]} {
    // CHECK: VarHandleOp
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "foo"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    func.call @assign_resource(%0) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> ()
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "foo"} : () -> tensor<!tf_type.resource<tensor<10xf32>>>
    func.call @assign_resource(%1) : (tensor<!tf_type.resource<tensor<10xf32>>>) -> ()
    return
  }
}

// -----

// CHECK-LABEL: module @handles_iterators
module @handles_iterators attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: @get_next
  func.func private @get_next(%arg0: tensor<!tf_type.resource>) -> tensor<200x10xf32> {
    // CHECK: %0 = "tf.Iterator"
    // CHECK-SAME: shared_name = "foo_iterator"
    // CHECK: "tf.IteratorGetNext"(%0)
    %0 = "tf.IteratorGetNext"(%arg0) : (tensor<!tf_type.resource>) -> tensor<200x10xf32>
    return %0 : tensor<200x10xf32>
  }

  // CHECK-LABEL: @main
  func.func @main()
  attributes {tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.Iterator"() {container = "", output_shapes = [#tf_type.shape<200x10>], output_types = [f32], shared_name = "foo_iterator"} : () -> tensor<!tf_type.resource>
    %1 = func.call @get_next(%0) : (tensor<!tf_type.resource>) -> tensor<200x10xf32>
    return
  }
}
