// RUN: tf-opt %s --tf-saved-model-convert-session-initializer-to-function --split-input-file | FileCheck %s

// CHECK-LABEL: simple_initializer
// CHECK-NOT: tf_saved_model.session_initializer
// CHECK: func @session_initializer
// CHECK: call @init1
module @simple_initializer attributes {tf_saved_model.semantics} {
"tf_saved_model.session_initializer"() {initializers = [@init1]} : () -> ()
func.func @init1() attributes {tf_saved_model.exported_names = ["init1"]} {
  %0 = "tf.Const"() {value = dense<42> : tensor<1xi64>} : () -> tensor<1xi64>
  return
}
}

// -----

// CHECK-LABEL: with_initializer_type
// CHECK-NOT: tf_saved_model.session_initializer
// CHECK: func @session_initializer
// CHECK: call @init1
module @with_initializer_type attributes {tf_saved_model.semantics} {
"tf_saved_model.session_initializer"() {initializers = [@init1]} : () -> ()
func.func @init1() attributes {tf_saved_model.exported_names = ["init1"], tf_saved_model.initializer_type = "init_op"} {
  %0 = "tf.Const"() {value = dense<42> : tensor<1xi64>} : () -> tensor<1xi64>
  return
}
}

// -----

// CHECK-LABEL: multiple_initializers
// CHECK-NOT: tf_saved_model.session_initializer
// CHECK: func @session_initializer
// CHECK: call @init1
// CHECK: call @init2
// CHECK: call @init3
module @multiple_initializers attributes {tf_saved_model.semantics} {
"tf_saved_model.session_initializer"() {initializers = [@init1, @init2, @init3]} : () -> ()
func.func @init1() attributes {tf_saved_model.exported_names = ["init1"]} {
  %0 = "tf.Const"() {value = dense<42> : tensor<1xi64>} : () -> tensor<1xi64>
  return
}
func.func @init2() attributes {tf_saved_model.exported_names = ["init2"]} {
  %0 = "tf.Const"() {value = dense<43> : tensor<1xi64>} : () -> tensor<1xi64>
  return
}
func.func @init3() attributes {tf_saved_model.exported_names = ["init3"]} {
  %0 = "tf.Const"() {value = dense<44> : tensor<1xi64>} : () -> tensor<1xi64>
  return
}
}
