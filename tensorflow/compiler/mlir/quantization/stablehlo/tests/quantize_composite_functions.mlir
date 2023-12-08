// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions | FileCheck %s

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_dot_general(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
// Checks that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK-LABEL: func.func private @quantize_dot_general
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.*]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.*]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// Checks that the entry function is quantized for dot_general. Quantized
// dot_general outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_dot_general_fn(%[[ARG_2:.*]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL_0:.*]] = stablehlo.dot_general %[[ARG_2]], %[[ARG_3]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.*]] = stablehlo.uniform_quantize %[[DOT_GENERAL_0]] : (tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that fused bias pattern is properly quantized.

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_dot_general_with_bias(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_bias_fn, _original_entry_function = "composite_dot_general_with_bias_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }

// CHECK-LABEL: func.func private @quantize_dot_general_with_bias
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<1x3xi32>} : () -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.*]] = call @quantized_dot_general_with_bias_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<1x3x!quant.uniform<i32:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.*]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

// CHECK: func.func private @quantized_dot_general_with_bias_fn(%[[ARG_2:.*]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_3:.*]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_4:.*]]: tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_dot_general_with_bias_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
// CHECK: %[[DOT_GENERAL_0:.*]] = stablehlo.dot_general %[[ARG_2]], %[[ARG_3]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i32:f32, 8.3371932554046126E-6>>
// CHECK: %[[ADD_0:.*]] = stablehlo.add %[[DOT_GENERAL_0]], %[[ARG_4]] : tensor<1x3x!quant.uniform<i32:f32, 8.3371932554046126E-6>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.*]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>

}

// -----

// Tests that fused bias pattern with dynamic shape is not quantized.
// TODO: b/307620428 - Add support for fused bias with dynamic shapes.

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_dot_general_with_bias_dynamic(%arg0: tensor<?x2xf32>) -> tensor<?x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<3xf32>} : () -> tensor<3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<?x2xf32>) -> tensor<?x2xf32>
    // expected-error@+1 {{'tf.XlaCallModule' op operand #0 must be variadic of tensor of tf.dtype values, but got}}
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {Sout = [#tf_type.shape<?x3>], _entry_function = @composite_dot_general_with_bias_dynamic_fn, _original_entry_function = "composite_dot_general_with_bias_dynamic_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<?x2xf32>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    return %2 : tensor<?x3xf32>
  }

  func.func private @composite_dot_general_with_bias_dynamic_fn(%arg0: tensor<?x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<3xf32>) -> tensor<?x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x3xf32>) -> tensor<?x3xf32>
    %1 = shape.shape_of %0 : tensor<?x3xf32> -> tensor<2xindex>
    %2 = stablehlo.dynamic_broadcast_in_dim %arg2, %1, dims = [1] : (tensor<3xf32>, tensor<2xindex>) -> tensor<?x3xf32>
    %3 = stablehlo.add %0, %2 : tensor<?x3xf32>
    return %3 : tensor<?x3xf32>
  }
}

// -----

// Tests error when there are no corresponding entry function to quantize
// (@composite_dot_general_fn).

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @error_when_no_entry_function(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
// expected-error @+2 {{Failed to find a valid entry function}}
// expected-error @+1 {{'tf.XlaCallModule' op operand #0 must be variadic of tensor of tf.dtype values}}
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
}

// -----

// Tests that XlaCallModule op is not quantized without the quantfork.stats ops.

module attributes {tf_saved_model.semantics} {
  func.func private @not_quantized_without_stats(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
// Check that "tf.Const" is converted to stablehlo.constant. XlaCallModule is
// not quantized.

// CHECK-LABEL: func.func private @not_quantized_without_stats
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
// CHECK: %[[XLA_CALL_MODULE_0:.*]] = "tf.XlaCallModule"(%[[ARG_1]], %[[CONST_0]]) <{{{.*}}}> {{{.*_entry_function = @composite_dot_general_fn.*}}} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[XLA_CALL_MODULE_0]]

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// Check that the composite_dot_general_fn is untouched.

// CHECK: func.func private @composite_dot_general_fn(%[[ARG_2:.*]]: tensor<1x2xf32>, %[[ARG_3:.*]]: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[ARG_2]], %[[ARG_3]]
// CHECK: return %[[DOT_GENERAL]]
}

// -----

// Test basic convolution is quantized.

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_convolution(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3x4x2>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64, _entry_function = @composite_convolution_fn, _original_entry_function = "composite_convolution_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// Checks that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK-LABEL: func.func private @quantize_convolution
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.*]] = call @quantized_convolution_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.*]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

  func.func private @composite_convolution_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
// Checks that the entry function is quantized for convolution. Quantized
// convolution outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_convolution_fn(%[[ARG_2:.*]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_3:.*]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[CONVOLUTION_0:.*]] = stablehlo.convolution(%[[ARG_2]], %[[ARG_3]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.*]] = stablehlo.uniform_quantize %[[CONVOLUTION_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that fused bias pattern is properly quantized.

module attributes {tf_saved_model.semantics} {
// The following pattern does not converge because of a bug in QuantizePass.
// TODO - b/305469508: Fix the QuantizePass to avoid this warning.
// expected-warning @+1 {{Failed to converge pattern at QuantizePass.}}
  func.func private @quantize_convolution_with_bias(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x3x4x2xf32>} : () -> tensor<1x3x4x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {Sout = [#tf_type.shape<1x3x4x2>], _entry_function = @composite_convolution_with_bias_fn, _original_entry_function = "composite_convolution_with_bias_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }

// CHECK-LABEL: func.func private @quantize_convolution_with_bias
// CHECK-SAME: (%[[ARG_1:.*]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant() {value = dense<{{.*}}> : tensor<1x3x4x2xi32>} : () -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.*]] = call @quantized_convolution_with_bias_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.*]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

// CHECK: func.func private @quantized_convolution_with_bias_fn(%[[ARG_2:.*]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_3:.*]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_4:.*]]: tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_convolution_with_bias_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3x4x2xf32>
    return %1 : tensor<1x3x4x2xf32>
  }
// CHECK: %[[CONVOLUTION_0:.*]] = stablehlo.convolution(%[[ARG_2]], %[[ARG_3]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[ADD_0:.*]] = stablehlo.add %[[CONVOLUTION_0]], %[[ARG_4]] : tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.*]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
}
