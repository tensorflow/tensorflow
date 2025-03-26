// RUN: tf-opt %s -split-input-file -tf-saved-model-asset-sinking='saved-model-dir=foo/bar' | FileCheck %s

// CHECK-LABEL: module @asset
module @asset attributes {tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() {initializers = [@init]} : () -> ()

  // CHECK-NOT: "tf_saved_model.asset"
  "tf_saved_model.asset"() {filename = "assets/test0.txt", sym_name = "asset0"} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/test1.txt", sym_name = "asset1"} : () -> ()

  // CHECK: func @init()
  func.func @init(%arg0: tensor<!tf_type.string> {tf_saved_model.bound_input = @asset0}, %arg1: tensor<!tf_type.string> {tf_saved_model.bound_input = @asset1}) attributes {tf_saved_model.exported_names = ["init"]} {
    // CHECK-DAG: %[[ASSET0:.*]] = "tf.Const"() <{value = dense<"foo/bar/assets/test0.txt"> : tensor<!tf_type.string>}>
    // CHECK-DAG: %[[ASSET1:.*]] = "tf.Const"() <{value = dense<"foo/bar/assets/test1.txt"> : tensor<!tf_type.string>}>

    // CHECK: %[[VAR0:.*]] = "tf.VarHandleOp"()
    %0 = "tf.VarHandleOp"() {container = "", shared_name = "var0"} : () -> tensor<!tf_type.resource<tensor<!tf_type.string>>>

    // CHECK: "tf.AssignVariableOp"(%[[VAR0]], %[[ASSET0]])
    "tf.AssignVariableOp"(%0, %arg0) : (tensor<!tf_type.resource<tensor<!tf_type.string>>>, tensor<!tf_type.string>) -> ()

    // CHECK: %[[VAR1:.*]] = "tf.VarHandleOp"()
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<!tf_type.string>>>

    // CHECK: "tf.AssignVariableOp"(%[[VAR1]], %[[ASSET1]])
    "tf.AssignVariableOp"(%1, %arg1) : (tensor<!tf_type.resource<tensor<!tf_type.string>>>, tensor<!tf_type.string>) -> ()

    func.return
  }
}
