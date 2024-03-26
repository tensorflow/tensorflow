// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions=enable-full-int-quantization=true | FileCheck --check-prefix=CHECK-FULL-INT %s

// Tests that a basic `stablehlo.add` and a fused `stablehlo.dot_general`
// are properly quantized.

module attributes {tf_saved_model.semantics} {
// CHECK-FULL-INT: func.func private @quantize_add_fn(%[[ARG:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
  func.func private @quantize_add_fn(%arg: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst_0 = "tf.Const"() {value = dense<1.00000000e-1> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
    %cst_1 = "tf.Const"() {value = dense<1.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %0 = "quantfork.stats"(%arg) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst_0) {Sout = [#tf_type.shape<1x2>], _entry_function = @composite_add_fn, _original_entry_function = "composite_add_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %3 = "quantfork.stats"(%2) {layerStats = dense<[5.00000000e-6, 6.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = "tf.XlaCallModule"(%3, %cst_1) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _original_entry_function = "composite_dot_general_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantfork.stats"(%4) {layerStats = dense<[5.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %5 : tensor<1x3xf32>
  }
// CHECK-FULL-INT: %[[CONST:.+]] = stablehlo.constant() {value = dense<127> : tensor<1x2xi8>} : () -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: %[[CONST_0:.+]] = stablehlo.constant() {value = dense<127> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>> 
// CHECK-FULL-INT: %[[UNIFORM_QUANTIZE:.+]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: %[[CALL:.+]] = call @quantized_add_fn(%[[UNIFORM_QUANTIZE]], %[[CONST]]) : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: %[[UNIFORM_DEQUANTIZE:.+]] = stablehlo.uniform_dequantize %[[CALL]] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2xf32>
// CHECK-FULL-INT: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[UNIFORM_DEQUANTIZE]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-FULL-INT: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

// CHECK-FULL-INT: func.func private @quantized_add_fn(%[[ARG_0:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_add_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
// CHECK-FULL-INT: %[[ADD:.+]] = stablehlo.add %arg0, %arg1 : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: return %[[ADD]] : tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-FULL-INT: func.func private @quantized_dot_general_fn(%[[ARG_0:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_1:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK-FULL-INT: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1,{{.*}}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK-FULL-INT: %[[UNIFORM_QUANTIZE:.+]] = stablehlo.uniform_quantize %[[DOT_GENERAL]] : (tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-FULL-INT: return %[[UNIFORM_QUANTIZE]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
}
