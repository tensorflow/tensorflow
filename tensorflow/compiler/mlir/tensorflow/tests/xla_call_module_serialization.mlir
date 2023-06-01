// RUN: tf-opt %s -split-input-file -tf-xla-call-module-serialization | FileCheck %s

// Tests that stablehlo functions called by XlaCallModuleOp in the top-level
// module can be serialized into bytecode and embedded in XlaCallModuleOp's
// `module` attribute.

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func private @_tf_func
  func.func private @_tf_func(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) {
    // CHECK: tf.StreamResults

    // StreamResults is a pseudo op in this test.
    "tf.StreamResults"(%arg0, %arg1) : (tensor<?xi32>, tensor<*xi32>) -> ()
    func.return
  }

  // CHECK-NOT: @_stablehlo_f
  func.func private @_stablehlo_f(%arg0: tensor<?xi32>) -> (tensor<?xi32>) attributes {_from_xla_call_module} {
    return %arg0 : tensor<?xi32>
  }

  // CHECK-NOT: @_stablehlo_main_0
  func.func private @_stablehlo_main_0(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    %arg2 = func.call @_stablehlo_f(%arg0) : (tensor<?xi32>) -> (tensor<?xi32>)
    return %arg2 : tensor<?xi32>
  }

  // CHECK-LABEL: func @main
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi32>, %[[ARG1:.*]]: tensor<10xi32>
  func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    // CHECK:      %[[RESULT:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME:   Sout = [#tf_type.shape<?>]
    // CHECK-SAME:   dim_args_spec = []
    // CHECK-NOT:    _entry_function
    // CHECK-NOT:    _stablehlo_module_attrs
    // CHECK-SAME:   function_list = [@_tf_func]
    // CHECK-SAME:   module = "ML\EFR{{.*}}"

    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_0, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, module = "", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    // CHECK: return %[[RESULT]]
    func.return %0 : tensor<10xi32>
  }
}
