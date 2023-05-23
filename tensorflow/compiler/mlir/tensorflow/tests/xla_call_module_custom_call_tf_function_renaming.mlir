// RUN: tf-opt %s -split-input-file -tf-xla-call-module-custom-call-tf-function-renaming | FileCheck %s

// Tests that if stablehlo.custom_call refers to a renamed TF host callback
// function, the tf-xla-call-module-custom-call-tf-renaming pass will update
// stablehlo.custom_call's caller_name attribute to the new TF function name.

module {
  // CHECK-LABEL: func private @__tf_host_callback
  func.func private @__tf_host_callback(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) attributes {tf._original_func_name = "original_tf"} {
    "tf.StreamResults"(%arg0, %arg1) : (tensor<?xi32>, tensor<*xi32>) -> ()
    func.return
  }

  // CHECK-LABEL: @_stablehlo_f
  func.func private @_stablehlo_f(%arg0: tensor<?xi32>) -> (tensor<?xi32>) attributes {_from_xla_call_module} {
    return %arg0 : tensor<?xi32>
  }

  // CHECK-LABEL: @_stablehlo_main_0
  func.func private @_stablehlo_main_0(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
    // CHECK: stablehlo.custom_call
    // CHECK-SAME: caller_name = @__tf_host_callback
    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {caller_name = "original_tf"}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    %arg2 = func.call @_stablehlo_f(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    return %arg2 : tensor<?xi32>
  }

  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_0, function_list = [@__tf_host_callback], module = "", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    func.return %0 : tensor<10xi32>
  }
}
