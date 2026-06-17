// RUN: tf-tfrt-opt -tf-restore-pruning %s | FileCheck %s

// CHECK-LABEL:   func.func @prune_unused_restore
func.func @prune_unused_restore() {
  %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
  %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  // CHECK-NOT: tf.RestoreV2
  %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
  %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  return
}


// CHECK-LABEL: func.func @used_restore_remains
func.func @used_restore_remains() {
  %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
  %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  // CHECK: tf.RestoreV2
  %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
  %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
  return
}
