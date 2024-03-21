// RUN: tf-tfrt-opt -tf-restore-prune-identity %s | FileCheck %s


// CHECK-LABEL:   func.func @prune_identity_in_restoration
func.func @prune_identity_in_restoration() {
  %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
  %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  // CHECK: [[TENSOR:%.*]] = "tf.RestoreV2"
  %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
  // CHECK-NEXT: [[HANDLE:%.*]] = "tf.VarHandleOp"
  %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  // CHECK-NOT: "tf.Identity"
  %2 = "tf.Identity"(%0) : (tensor<3x1xf32>) -> tensor<3x1xf32>
  // CHECK-NEXT: "tf.AssignVariableOp"([[HANDLE]], [[TENSOR]])
  "tf.AssignVariableOp"(%1, %2) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL:   func.func @identity_remains_without_restore_op
func.func @identity_remains_without_restore_op() {
  %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
  // CHECK: tf.Identity
  %1 = "tf.Identity"(%cst) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
  return
}



