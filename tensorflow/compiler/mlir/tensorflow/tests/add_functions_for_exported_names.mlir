// RUN: tf-opt %s --tf-saved-model-add-functions-for-exported-names --split-input-file | FileCheck %s

// CHECK-LABEL @export_three
module @export_three attributes {tf_saved_model.semantics} {
  // CHECK: func @a
  // CHECK-SAME: i32
  // CHECK-SAME: f32
  // CHECK-SAME: tensor<i32>
  // CHECK: [[tuple:%.*]]:2 = call @f
  // CHECK: return [[tuple]]#0, [[tuple]]#1
  // CHECK-SAME: tensor<i8>
  // CHECK-SAME: tensor<i16>
  // CHECK: func @b
  // CHECK: func @c
  // CHECK: func private @f
  // CHECK-NOT: exported_names
  // CHECK: tf.Const
  // CHECK: tf.Const
  // CHECK: return
  func.func @f(%a: i32         {tf_saved_model.index_path = [0]},
               %b: f32         {tf_saved_model.index_path = [0]},
               %c: tensor<i32> {tf_saved_model.index_path = [0]})
               ->( tensor<i8>  {tf_saved_model.index_path = [0]},
                   tensor<i16> {tf_saved_model.index_path = [0]})
                    attributes {tf_saved_model.exported_names = ["a", "b", "c"]} {
    %0 = "tf.Const"() {value = dense<42> : tensor<i8>} : () -> tensor<i8>
    %1 = "tf.Const"() {value = dense<4242> : tensor<i16>} : () -> tensor<i16>
    return %0, %1: tensor<i8>, tensor<i16>
  }
}

// -----

// CHECK-LABEL: @export_as_self
module @export_as_self attributes {tf_saved_model.semantics} {
  // CHECK: func @foobar
  // CHECK: func @baz
  // CHECK: func private @foobar_internal
  func.func @foobar()
    attributes {tf_saved_model.exported_names = ["foobar", "baz"]}
  {
    func.return
  }
}
