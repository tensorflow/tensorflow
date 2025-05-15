// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-xla-call-module-to-call | FileCheck %s

// -----

// Tests composite tf.XlaCallModule is converted to func.call.

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1x1024xf32>) -> tensor<1x3xf32> {
    // CHECK: call @composite_dot_general_fn_1
    // CHECK-SAME: (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    // CHECK-NOT: tf.XlaCallModule
    %0 = "tf.Const"() <{value = dense<0.5> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
    %2 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
  // CHECK-LABEL: func.func private @composite_dot_general_fn_1
  // CHECK-SAME: -> tensor<1x3xf32>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
