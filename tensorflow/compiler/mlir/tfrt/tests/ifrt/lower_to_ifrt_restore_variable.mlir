// RUN: tf-tfrt-opt -split-input-file -verify-diagnostics -lower-to-ifrt-restore-variable %s | FileCheck %s

// -----
// single variable

// CHECK-LABEL:   func.func @restore_single() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]])
// CHECK-SAME:        {restored_dtypes = [f32]}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @restore_single() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}

// -----
// single variable: VarHandleOp is before RestoreV2

// CHECK-LABEL:   func.func @varhandle_before_restore() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]])
// CHECK-SAME:        {restored_dtypes = [f32]}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @varhandle_before_restore() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}


// -----
// multiple variables

// CHECK-LABEL:   func.func @restore_multiple() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     [[HANDLEZ:%.*]] = "tf.VarHandleOp"() <{container = "x", shared_name = "z"}> : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]], [[HANDLEZ]])
// CHECK-SAME:        {restored_dtypes = [f32, f32]}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @restore_multiple() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %0:2 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<3x1xf32>, tensor<1x3xf32>)
    %1 = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    %2 = "tf.VarHandleOp"() <{container = "x", shared_name = "z"}> : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
    "tf.AssignVariableOp"(%2, %0#1) : (tensor<!tf_type.resource<tensor<1x3xf32>>>, tensor<1x3xf32>) -> ()
    return
  }
}

// -----
// Restored variable is not assigned with a name is an error.

module {
  func.func @unassigned_restore_return_error() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    //expected-error@below {{'tf.RestoreV2' op expects 2 VarHandleOps, but got 1}}
    %0:2 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<3x1xf32>, tensor<1x3xf32>)
    %1 = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}

