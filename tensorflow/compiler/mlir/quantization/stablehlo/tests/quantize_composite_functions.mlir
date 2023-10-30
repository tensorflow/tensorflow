// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions | FileCheck %s

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_dot_general(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<3x3xf32>} : () -> tensor<3x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
// Checks that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK-LABEL: func.func private @quantize_dot_general
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<3x3xi8>} : () -> tensor<3x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.*]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<3x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.*]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

  func.func private @composite_dot_general_fn(%arg0: tensor<1x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// Checks that the entry function is quantized for dot_general. Quantized
// dot_general outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_dot_general_fn(%[[ARG_2:.*]]: tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_3:.*]]: tensor<3x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL_0:.*]] = stablehlo.dot_general %[[ARG_2]], %[[ARG_3]], contracting_dims = [1] x [0] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<3x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.*]] = stablehlo.uniform_quantize %[[DOT_GENERAL_0]] : (tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests error when there are no corresponding entry function to quantize
// (@composite_dot_general_fn).

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @error_when_no_entry_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<3x3xf32>} : () -> tensor<3x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// expected-error @+2 {{Failed to find a valid entry function}}
// expected-error @+1 {{'tf.XlaCallModule' op operand #0 must be variadic of tensor of tf.dtype values}}
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
}

// -----

// Tests that XlaCallModule op is not quantized without the quantfork.stats ops.

module attributes {tf_saved_model.semantics} {
  func.func private @not_quantized_without_stats(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<3x3xf32>} : () -> tensor<3x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
// Check that "tf.Const" is converted to stablehlo.constant. XlaCallModule is
// not quantized.

// CHECK-LABEL: func.func private @not_quantized_without_stats
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<3.000000e-01> : tensor<3x3xf32>
// CHECK: %[[XLA_CALL_MODULE_0:.*]] = "tf.XlaCallModule"(%[[ARG_1]], %[[CONST_0]]) <{{{.*}}}> {{{.*_entry_function = @composite_dot_general_fn.*}}} : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[XLA_CALL_MODULE_0]]

  func.func private @composite_dot_general_fn(%arg0: tensor<1x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// Check that the composite_dot_general_fn is untouched.

// CHECK: func.func private @composite_dot_general_fn(%[[ARG_2:.*]]: tensor<1x3xf32>, %[[ARG_3:.*]]: tensor<3x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[ARG_2]], %[[ARG_3]]
// CHECK: return %[[DOT_GENERAL]]
}

// -----

// Tests that a fusion pattern for dot_general is not yet supported. Further op
// coverage will be provided in the future.
// TODO - b/307620428: Increase op coverage to cover this test case.

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @dot_general_fn_fusion_not_quantized(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<3x3xf32>} : () -> tensor<3x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// expected-error @+1 {{'tf.XlaCallModule' op operand #0 must be variadic of tensor of tf.dtype values}}
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }

  func.func private @composite_dot_general_fn(%arg0: tensor<1x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x3xf32>, tensor<3x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}
