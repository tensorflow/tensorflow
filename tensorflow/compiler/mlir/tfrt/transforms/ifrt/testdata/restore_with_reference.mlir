module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.session_initializer"() <{initializers = [@"save/restore_all_1"]}> : () -> ()
  func.func @"save/restore_all_1"() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_save/restore_all_1"], tf_saved_model.initializer_type = "restore_op"} {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %0:2 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<i64>, tensor<1x3xf32>)
    %1 = "tf.VariableV2"() <{container = "x", shape = #tf_type.shape<>, shared_name = "y"}> : () -> tensor<!tf_type.int64ref>
    %dummy = "tf.Assign"(%1, %0#0) : (tensor<!tf_type.int64ref>, tensor<i64>) ->  tensor<!tf_type.int64ref>
    %2 = "tf.VarHandleOp"() <{container = "x", shared_name = "z"}> : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
    "tf.AssignVariableOp"(%2, %0#1) : (tensor<!tf_type.resource<tensor<1x3xf32>>>, tensor<1x3xf32>) -> ()
    return
  }
}
