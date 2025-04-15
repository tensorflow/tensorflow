// RUN: tf-opt %s -split-input-file -tf-xla-call-module-serialization -tf-xla-call-module-deserialization | FileCheck %s

// Tests that running xla-call-module-serialization followed by
// xla-call-module-deserialization preserves the original module.
//
// Note that function names may be different, but arguments, attributes,
// results, and function body should be the same.

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func @main
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi32>, %[[ARG1:.*]]: tensor<10xi32>
  func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    // CHECK:      %[[RESULT:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME:   Sout = [#tf_type.shape<?>]
    // CHECK-NOT:    function_list
    // CHECK-SAME:   module = ""
    // CHECK-SAME:   platforms = []
    // CHECK-SAME:   version = 5
    // CHECK-SAME:   _entry_function = @main_0
    // CHECK-SAME:   _stablehlo_module_attrs = {}
    // CHECK-SAME:   _stablehlo_version = "0.12.0"

    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @main_0, _stablehlo_version = "0.12.0", module = "", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    // CHECK: return %[[RESULT]]
    func.return %0 : tensor<10xi32>
  }

  // CHECK-LABEL: func private @_tf_func
  func.func private @_tf_func(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) {
    // CHECK: tf.StreamResults

    // StreamResults is a pseudo op in this test.
    "tf.StreamResults"(%arg0, %arg1) : (tensor<?xi32>, tensor<*xi32>) -> ()
    func.return
  }

  // CHECK-LABEL: func private @main_0
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}
  // CHECK-SAME:    %[[ARG1:.*]]: tensor<*xi32>)
  // CHECK-SAME:    (tensor<?xi32> {jax.result_info = ""})
  // CHECK-SAME:    attributes {_from_xla_call_module}
  func.func private @main_0(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
    // CHECK:      stablehlo.custom_call @tf.call_tf_function(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: {
    // CHECK-SAME:  api_version = 2 : i32,
    // CHECK-SAME:  has_side_effect = true,
    // CHECK-SAME:  tf.backend_config = {called_func = @_tf_func}
    // CHECK-SAME: }
    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    // CHECK: call @f
    %arg2 = func.call @f(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    return %arg2 : tensor<?xi32>
  }

  // CHECK-LABEL: func private @f
  // CHECK:    attributes {_from_xla_call_module}
  func.func private @f(%arg0: tensor<?xi32>) -> (tensor<?xi32>) attributes {_from_xla_call_module} {
    return %arg0 : tensor<?xi32>
  }
}
