// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-unwrap-xla-call-module-op | FileCheck %s

// Tests if XlaCallModule op without quantizable trait that calls function with
// '_from_xla_call_module' trait is unwrapped.
// Tests if XlaCallModule op with quantizable trait is not unwrapped.
// Tests if XlaCallModule op without quantizable trait that calls function
// without '_from_xla_call_module' trait is not unwrapped.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1682 : i32}, tf_saved_model.semantics} {
  // CHECK-LABEL: @main_00
  // CHECK: %[[ARG0:.*]]: tensor<10x1x1024xf32>
  func.func private @main_00(%arg0: tensor<10x1x1024xf32>) -> tensor<6x5xf32> attributes {tf._original_func_name = "main_0"} {
    %0 = "tf.Const"() <{value = dense<1.000000e+00> : tensor<10x1024x3xf32>}> : () -> tensor<10x1024x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<10x1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    %2 = "tf.XlaCallModule"(%1) <{Sout = [#tf_type.shape<3x10>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @main_0, _stablehlo_version = "1.0.0", _stablehlo_module_attrs = {}, device = ""} : (tensor<10x1x3xf32>) -> tensor<3x10xf32>
    %3 = "tf.XlaCallModule"(%2) <{Sout = [#tf_type.shape<6x5>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = ["CPU"], version = 9 : i64}> {_entry_function = @main_1, _stablehlo_version = "1.0.0", _stablehlo_module_attrs = {}, device = ""} : (tensor<3x10xf32>) -> tensor<6x5xf32>
    return %3 : tensor<6x5xf32>
  }
  // CHECK: %[[CST:.*]] = "tf.Const"()
  // CHECK-NEXT: %[[CALL1:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[CST]])
  // CHECK-SAME: _entry_function = @composite_dot_general_fn_1
  // CHECK-SAME: _tfl_quant_trait = "fully_quantizable"
  // CHECK-NOT: "tf.XlaCallModule"
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[CALL1]] : (tensor<10x1x3xf32>) -> tensor<3x10xf32>
  // CHECK-NEXT: %[[CALL2:.*]] = "tf.XlaCallModule"(%[[RESHAPE]])
  // CHECK-SAME: _entry_function = @main_1
  // CHECK-NOT:  _tfl_quant_trait = "fully_quantizable"
  // CHECK-NEXT: return %[[CALL2]]

  // CHECK: @composite_dot_general_fn_1
  func.func private @composite_dot_general_fn_1(%arg0: tensor<10x1x1024xf32>, %arg1: tensor<10x1024x3xf32>) -> tensor<10x1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}} : (tensor<10x1x1024xf32>, tensor<10x1024x3xf32>) -> tensor<10x1x3xf32>
    return %0 : tensor<10x1x3xf32>
  }
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general
  // CHECK-NEXT: return %[[DOT]]

  // CHECK: @main_0
  func.func private @main_0(%arg0: tensor<10x1x3xf32>) -> tensor<3x10xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.reshape %arg0 : (tensor<10x1x3xf32>) -> tensor<3x10xf32>
    return %0 : tensor<3x10xf32>
  }
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape
  // CHECK-NEXT: return %[[RESHAPE]]

  // CHECK: @main_1
  func.func private @main_1(%arg0: tensor<3x10xf32>) -> tensor<6x5xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x10xf32>) -> tensor<6x5xf32>
    return %0 : tensor<6x5xf32>
  }
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape
  // CHECK-NEXT: return %[[RESHAPE]]
}
