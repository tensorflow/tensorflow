// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions | FileCheck %s
// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics \
// RUN:     -stablehlo-quantize-composite-functions=enable-per-channel-quantized-weight=false | FileCheck --check-prefix=CHECK-PER-TENSOR %s

// Tests that basic dot_general is properly quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
// Checks that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK: func.func private @quantize_dot_general_fn(%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

// CHECK-PER-TENSOR: func.func private @quantize_dot_general_fn(%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// Checks that the entry function is quantized for dot_general. Quantized
// dot_general outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_dot_general_fn(%[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[DOT_GENERAL_0]] : (tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_dot_general_fn(%[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[DOT_GENERAL_0]] : (tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that `stablehlo.dot_general` with `batching_dim` is quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_batch_per_tensor_quantized_fn(%arg0: tensor<2x2x2xf32>) -> tensor<2x2x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x2x3xf32>} : () -> tensor<2x2x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<2x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
    return %2 : tensor<2x2x3xf32>
  }
// CHECK: func.func private @quantize_dot_general_batch_per_tensor_quantized_fn(%[[ARG_0:.+]]: tensor<2x2x2xf32>) -> tensor<2x2x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x2x3xi8>}> : () -> tensor<2x2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<2x2x2xf32>) -> tensor<2x2x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) {_quantization_method = "static_range_ptq { }"} : (tensor<2x2x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<2x2x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<2x2x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<2x2x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<2x2x3xf32>

  func.func private @composite_dot_general_fn(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2x2x3xf32>) -> tensor<2x2x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x2x2xf32>, tensor<2x2x3xf32>) -> tensor<2x2x3xf32>
    return %0 : tensor<2x2x3xf32>
  }
}

// -----

// Tests that fused pattern for dot_general + bias is properly quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_with_bias_same_shape_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_with_bias_same_shape_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_with_bias_same_shape_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
// CHECK: func.func private @quantize_dot_general_with_bias_same_shape_fn(%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>
// CHECK: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x3xi32>}> : () -> tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_with_bias_same_shape_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>, tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

// CHECK-PER-TENSOR: func.func private @quantize_dot_general_with_bias_same_shape_fn(%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x3xi32>}> : () -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_dot_general_with_bias_same_shape_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<1x3x!quant.uniform<i32:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

  func.func private @composite_dot_general_with_bias_same_shape_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
// CHECK: func.func private @quantized_dot_general_with_bias_same_shape_fn(%[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>) -> tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[DOT_GENERAL_0]], %[[ARG_3]] : tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_dot_general_with_bias_same_shape_fn(%[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[DOT_GENERAL_0]], %[[ARG_3]] : tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>

}

// -----

// Tests that fused pattern for dot_general + bias with dynamic batch dimension
// is properly quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_dot_general_with_bias_dynamic_fn(%arg0: tensor<?x2xf32>) -> tensor<?x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<3xf32>} : () -> tensor<3xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<?x2xf32>) -> tensor<?x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {Sout = [#tf_type.shape<?x3>], _entry_function = @composite_dot_general_with_bias_dynamic_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_with_bias_dynamic_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<?x2xf32>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<?x3xf32>) -> tensor<?x3xf32>
    return %2 : tensor<?x3xf32>
  }
// CHECK: func.func private @quantize_dot_general_with_bias_dynamic_fn(%[[ARG_0:.+]]: tensor<?x2xf32>) -> tensor<?x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32:0, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x2xf32>) -> tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_with_bias_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) {_quantization_method = "static_range_ptq { }"} : (tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>, tensor<3x!quant.uniform<i32:f32:0, {{.*}}>) -> tensor<?x3x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3xf32>

// CHECK-PER-TENSOR: func.func private @quantize_dot_general_with_bias_dynamic_fn(%[[ARG_0:.+]]: tensor<?x2xf32>) -> tensor<?x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x2xf32>) -> tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_dot_general_with_bias_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]]) {_quantization_method = "static_range_ptq { }"} : (tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<3x!quant.uniform<i32:f32, {{.*}}>) -> tensor<?x3x!quant.uniform<i8:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3xf32>

  func.func private @composite_dot_general_with_bias_dynamic_fn(%arg0: tensor<?x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<3xf32>) -> tensor<?x3xf32> attributes {_from_xla_call_module} {
      %cst_0 = stablehlo.constant dense<2> : tensor<1xi32>
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x3xf32>) -> tensor<?x3xf32>
      %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
      %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
      %3 = stablehlo.concatenate %2, %cst_0, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [1] : (tensor<3xf32>, tensor<2xi32>) -> tensor<?x3xf32>
      %5 = stablehlo.add %0, %4 : tensor<?x3xf32>
      return %5 : tensor<?x3xf32>
    }
}
// CHECK: func.func private @quantized_dot_general_with_bias_dynamic_fn(%[[ARG_1:.+]]: tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>, %[[ARG_3:.+]]: tensor<3x!quant.uniform<i32:f32:0, {{.*}}>>) -> tensor<?x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[CONST_2:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<?x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[DOT_GENERAL_0]], dim = 0 : (tensor<?x3x!quant.uniform<i32:f32:1, {{.*}}>)
// CHECK: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [1] : (tensor<3x!quant.uniform<i32:f32:0, {{.*}}>>, tensor<2xi32>) -> tensor<?x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[DOT_GENERAL_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x!quant.uniform<i32:f32:1, {{.*}}>>)
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<?x3x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_dot_general_with_bias_dynamic_fn(%[[ARG_1:.+]]: tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<3x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<?x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[CONST_2:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK-PER-TENSOR: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]], contracting_dims = [1] x [0] : (tensor<?x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<?x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[DOT_GENERAL_0]], dim = 0 : (tensor<?x3x!quant.uniform<i32:f32, {{.*}}>)
// CHECK-PER-TENSOR: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
// CHECK-PER-TENSOR: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [1] : (tensor<3x!quant.uniform<i32:f32, {{.*}}>>, tensor<2xi32>) -> tensor<?x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[DOT_GENERAL_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x!quant.uniform<i32:f32, {{.*}}>>)
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<?x3x!quant.uniform<i8:f32, {{.*}}>>

// -----

// Tests that basic convolution is properly quantized. It is per-channel
// quantized unless `enable-per-channel-quantized-weight=false`, according to
// `_quantization_method` with an `input_quantized_types` and explicit
// `dimension_specs`.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_fn(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {
        Sout = [#tf_type.shape<1x3x4x2>],
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64,
        _entry_function = @composite_conv_fn,
        _stableghlo_version = "1.0.0",
        _original_entry_function = "composite_conv_fn",
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _stablehlo_module_attrs = {},
        _tfl_quant_trait = "fully_quantizable",
        device = ""
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// Check that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK: func.func private @quantize_conv_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]])
// CHECK-PER-TENSOR-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

  func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    return %0 : tensor<1x3x4x2xf32>
  }
// Checks that the entry function is quantized for convolution. Quantized
// convolution outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_conv_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[CONVOLUTION_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[CONVOLUTION_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that basic convolution is properly quantized. In this example, the
// convolution is always per-tensor quantized (even if
// enable-per-channel-quantized-weights=true), according to
// `_quantization_method`.

// CHECK-LABEL: quantize_conv_fn_per_tensor
func.func @quantize_conv_fn_per_tensor(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> {
  %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
  %1 = "tf.XlaCallModule"(%0, %cst) {
      Sout = [#tf_type.shape<1x3x4x2>],
      dim_args_spec = [],
      disabled_checks = [],
      has_token_input_output = false,
      module = "",
      platforms = [],
      version = 5 : i64,
      _entry_function = @composite_conv_fn,
      _stablehlo_version = "1.0.0",
      _original_entry_function = "composite_conv_fn",
      _quantization_method = "static_range_ptq { }",
      _stablehlo_module_attrs = {},
      _tfl_quant_trait = "fully_quantizable",
      device = ""
    } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
  return %2 : tensor<1x3x4x2xf32>
}
// Check that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.

// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32>
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

func.func private @composite_conv_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
  return %0 : tensor<1x3x4x2xf32>
}
// Checks that the entry function is quantized for convolution. Quantized
// convolution outputs an i32 quantized tensor, followed by requantization to
// i8 quantized tensor.

// CHECK: func.func private @quantized_conv_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[CONVOLUTION_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// -----

// Tests that fused pattern for convolution + bias is properly quantized.

// Checks that fused functions with 1D bias is properly quantized.
// The 1D bias should be broadcasted in dims [3], where it initially has
// `quantizedDimension=0`, but has `quantizedDimension=3` after broadcasting.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_with_bias_1d_fn(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<2xf32>} : () -> tensor<2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {
        Sout = [#tf_type.shape<1x3x4x2>],
        _entry_function = @composite_conv_with_bias_1d_fn,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_conv_with_bias_1d_fn",
        _stablehlo_module_attrs = {},
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _tfl_quant_trait = "fully_quantizable",
        device = "",
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<1x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// CHECK: func.func private @quantize_conv_with_bias_1d_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<47978> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {8.3371932554046126E-6,8.3371932554046126E-6}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_with_bias_1d_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>, tensor<2x!quant.uniform<i32:f32:0, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, 0.0027450979924669452:-128>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_with_bias_1d_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_with_bias_1d_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-PER-TENSOR-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<2x!quant.uniform<i32:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

  func.func private @composite_conv_with_bias_1d_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2xf32>) -> tensor<1x3x4x2xf32>
    %1 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = stablehlo.add %1, %0 : tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// CHECK: func.func private @quantized_conv_with_bias_1d_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>, %[[ARG_3:.+]]: tensor<2x!quant.uniform<i32:f32:0, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[BROADCAST_IN_DIM:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<2x!quant.uniform<i32:f32:0, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[BROADCAST_IN_DIM]] : tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_with_bias_1d_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[BROADCAST_IN_DIM:.+]] = stablehlo.broadcast_in_dim %[[ARG_3]]
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[BROADCAST_IN_DIM]] : tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Checks that fused functions with 4D bias is properly quantized.
// The 4D bias should be braoadcasted in dims [0, 1, 2, 3], where it
// already has `quantizedDimension=3`.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_with_bias_fn(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x1x1x2xf32>} : () -> tensor<1x1x1x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {
        Sout = [#tf_type.shape<1x3x4x2>],
        _entry_function = @composite_conv_with_bias_fn,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_conv_with_bias_fn",
        _stablehlo_module_attrs = {},
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _tfl_quant_trait = "fully_quantizable",
        device = "",
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64
      } : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<1x3x4x2xf32>) -> tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// CHECK: func.func private @quantize_conv_with_bias_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_with_bias_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>, tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_with_bias_fn(%[[ARG_0:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_with_bias_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-PER-TENSOR-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3x4x2xf32>

  func.func private @composite_conv_with_bias_fn(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<1x1x1x2xf32>) -> tensor<1x3x4x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>) -> tensor<1x3x4x2xf32>
    %1 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x4x2xf32>
    %2 = stablehlo.add %1, %0 : tensor<1x3x4x2xf32>
    return %2 : tensor<1x3x4x2xf32>
  }
// CHECK: func.func private @quantized_conv_with_bias_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK: %[[BROADCAST_IN_DIM:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<1x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[BROADCAST_IN_DIM]] : tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_with_bias_fn(%[[ARG_1:.+]]: tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR: %[[BROADCAST_IN_DIM:.+]] = stablehlo.broadcast_in_dim %arg2
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<1x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[BROADCAST_IN_DIM]] : tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<1x3x4x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_1]] : tensor<1x3x4x2x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that fused pattern for convolution + bias with dynamic batch dimension
// is properly quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_with_bias_dynamic_fn(%arg0: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x1x1x2xf32>} : () -> tensor<1x1x1x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {
        Sout = [#tf_type.shape<1x3x4x2>],
        _entry_function = @composite_conv_with_bias_dynamic_fn,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_conv_with_bias_dynamic_fn",
        _stablehlo_module_attrs = {},
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _tfl_quant_trait = "fully_quantizable",
        device = "",
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64
      } : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 7.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x2xf32>) -> tensor<?x3x4x2xf32>
    return %2 : tensor<?x3x4x2xf32>
  }
// CHECK: func.func private @quantize_conv_with_bias_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_with_bias_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>, tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_with_bias_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_with_bias_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-PER_TENSOR-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>, tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

  func.func private @composite_conv_with_bias_dynamic_fn(%arg0: tensor<?x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32> attributes {_from_xla_call_module} {
    %cst_0 = stablehlo.constant dense<3> : tensor<1xi32>
    %cst_1 = stablehlo.constant dense<4> : tensor<1xi32>
    %cst_2 = stablehlo.constant dense<2> : tensor<1xi32>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x4x2xf32>
    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?x3x4x2xf32>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.concatenate %2, %cst_0, %cst_1, %cst_2, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>, tensor<4xi32>) -> tensor<?x3x4x2xf32>
    %5 = stablehlo.add %0, %4 : tensor<?x3x4x2xf32>
    return %5 : tensor<?x3x4x2xf32>
  }
}
// CHECK: func.func private @quantized_conv_with_bias_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>)
// CHECK: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>)
// CHECK: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_with_bias_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>)
// CHECK-PER-TENSOR: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-PER-TENSOR: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>)
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>>

// -----

// Tests that fused pattern for convolution + bias + relu with
// dynamic batch dimension is properly quantized.

// Note that this checks for identical condition as
// quantize_conv_with_bias_dynamic_fn, omitting stablehlo.maximum.
// This is because activation clipping which includes 0.0f can be simply
// omitted from the graph as the lifted function's out_scale and out_zp are
// already calculated based on the clipped distribution.
// Note that the resulting scale and zero point should be calculated based on
// clipped range [0, r_max].

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_with_bias_and_relu_dynamic_fn(%arg0: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x1x1x2xf32>} : () -> tensor<1x1x1x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {
        Sout = [#tf_type.shape<1x3x4x2>],
        _entry_function = @composite_conv_with_bias_and_relu_dynamic_fn,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_conv_with_bias_and_relu_dynamic_fn",
        _stablehlo_module_attrs = {},
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _tfl_quant_trait = "fully_quantizable",
        device = "",
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64
      } : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[0.00000000e-6, 8.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x2xf32>) -> tensor<?x3x4x2xf32>
    return %2 : tensor<?x3x4x2xf32>
  }
// CHECK: func.func private @quantize_conv_with_bias_and_relu_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_with_bias_and_relu_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>, tensor<1x1x1x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_with_bias_and_relu_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_with_bias_and_relu_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-PER-TENSOR-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>, tensor<1x1x1x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

  func.func private @composite_conv_with_bias_and_relu_dynamic_fn(%arg0: tensor<?x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32> attributes {_from_xla_call_module} {
    %cst_0 = stablehlo.constant dense<3> : tensor<1xi32>
    %cst_1 = stablehlo.constant dense<4> : tensor<1xi32>
    %cst_2 = stablehlo.constant dense<2> : tensor<1xi32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x4x2xf32>
    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?x3x4x2xf32>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.concatenate %2, %cst_0, %cst_1, %cst_2, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>, tensor<4xi32>) -> tensor<?x3x4x2xf32>
    %5 = stablehlo.add %0, %4 : tensor<?x3x4x2xf32>
    %6 = stablehlo.clamp %cst_3, %5, %cst_4 : (tensor<f32>, tensor<?x3x4x2xf32>, tensor<f32>) -> tensor<?x3x4x2xf32>
    return %6 : tensor<?x3x4x2xf32>
  }
}
// CHECK: func.func private @quantized_conv_with_bias_and_relu_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>
// CHECK: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>)
// CHECK: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>
// CHECK: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_with_bias_and_relu_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>
// CHECK-PER-TENSOR: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>)
// CHECK-PER-TENSOR: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-PER-TENSOR: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0031372549487095253:-128>>

// -----

// Tests that fused pattern for convolution + bias + relu6 with
// dynamic batch dimension is properly quantized.

// Note that this checks for identical condition as
// quantize_conv_with_bias_dynamic_fn, omitting stablehlo.clamp.
// This is because activation clipping which includes 0.0f can be simply
// omitted from the graph as the lifted function's out_scale and out_zp are
// already calculated based on the clipped distribution.
// Note that the resulting scale and zero point should be calculated based on
// clipped range [0, r_max].

module attributes {tf_saved_model.semantics} {
  func.func private @quantize_conv_with_bias_and_relu6_dynamic_fn(%arg0: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {value = dense<4.00000000e-1> : tensor<1x1x1x2xf32>} : () -> tensor<1x1x1x2xf32>
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[6.00000000e-6, 9.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3xf32>
    %1 = "tf.XlaCallModule"(%0, %cst, %cst_0) {
        Sout = [#tf_type.shape<1x3x4x2>],
        _entry_function = @composite_conv_with_bias_and_relu6_dynamic_fn,
        _stablehlo_version = "1.0.0",
        _original_entry_function = "composite_conv_with_bias_and_relu6_dynamic_fn",
        _stablehlo_module_attrs = {},
        // Per-channel quantization at dimension 3 for input index 1.
        _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
        _tfl_quant_trait = "fully_quantizable",
        device = "",
        dim_args_spec = [],
        disabled_checks = [],
        has_token_input_output = false,
        module = "",
        platforms = [],
        version = 5 : i64
      } : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[5.00000000e-6, 6.00000000e-1]> : tensor<2xf32>} : (tensor<?x3x4x2xf32>) -> tensor<?x3x4x2xf32>
    return %2 : tensor<?x3x4x2xf32>
  }
// CHECK: func.func private @quantize_conv_with_bias_and_relu6_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>
// CHECK-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_conv_with_bias_and_relu6_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-SAME: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>, tensor<1x1x1x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

// CHECK-PER-TENSOR: func.func private @quantize_conv_with_bias_and_relu6_dynamic_fn(%[[ARG_0:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x4x2xf32> attributes {tf._original_func_name = "main_0"}
// CHECK-PER-TENSOR-DAG: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>
// CHECK-PER-TENSOR-DAG: %[[CONST_1:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1x1x1x2xi32>}> : () -> tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<?x3x4x3xf32>) -> tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[CALL_0:.+]] = call @quantized_conv_with_bias_and_relu6_dynamic_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]], %[[CONST_1]])
// CHECK-PER-TENSOR: {_quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}"} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>, tensor<1x1x1x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>
// CHECK-PER-TENSOR: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<?x3x4x2xf32>
// CHECK-PER-TENSOR: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<?x3x4x2xf32>

  func.func private @composite_conv_with_bias_and_relu6_dynamic_fn(%arg0: tensor<?x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<1x1x1x2xf32>) -> tensor<?x3x4x2xf32> attributes {_from_xla_call_module} {
    %cst_0 = stablehlo.constant dense<3> : tensor<1xi32>
    %cst_1 = stablehlo.constant dense<4> : tensor<1xi32>
    %cst_2 = stablehlo.constant dense<2> : tensor<1xi32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_4 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x4x2xf32>
    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?x3x4x2xf32>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.concatenate %2, %cst_0, %cst_1, %cst_2, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg2, %3, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>, tensor<4xi32>) -> tensor<?x3x4x2xf32>
    %5 = stablehlo.add %0, %4 : tensor<?x3x4x2xf32>
    %6 = stablehlo.clamp %cst_3, %5, %cst_4 : (tensor<f32>, tensor<?x3x4x2xf32>, tensor<f32>) -> tensor<?x3x4x2xf32>
    return %6 : tensor<?x3x4x2xf32>
  }
}
// CHECK: func.func private @quantized_conv_with_bias_and_relu6_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.0023622048182750312,0.0023622048182750312}>>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>
// CHECK: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>)
// CHECK: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32:3, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32:3, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32:3, {8.3371932554046126E-6,8.3371932554046126E-6}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>
// CHECK: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>

// CHECK-PER-TENSOR: func.func private @quantized_conv_with_bias_and_relu6_dynamic_fn(%[[ARG_1:.+]]: tensor<?x3x4x3x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_2:.+]]: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, {{.*}}>>, %[[ARG_3:.+]]: tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
// CHECK-PER-TENSOR-DAG: %[[CONST_2:.+]] = stablehlo.constant dense<3> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_3:.+]] = stablehlo.constant dense<4> : tensor<1xi32>
// CHECK-PER-TENSOR-DAG: %[[CONST_4:.+]] = stablehlo.constant dense<2> : tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONVOLUTION_0:.+]] = stablehlo.convolution(%[[ARG_1]], %[[ARG_2]]) {{.*}} : (tensor<?x3x4x3x!quant.uniform<i8:f32, 0.0035294116712084002:-128>>, tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>
// CHECK-PER-TENSOR: %[[GET_DIMENSION_SIZE_0:.+]] = stablehlo.get_dimension_size %[[CONVOLUTION_0]], dim = 0 : (tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>)
// CHECK-PER-TENSOR: %[[RESHAPE_0:.+]] = stablehlo.reshape %[[GET_DIMENSION_SIZE_0]] : (tensor<i32>) -> tensor<1xi32>
// CHECK-PER-TENSOR: %[[CONCATENATE_0:.+]] = stablehlo.concatenate %[[RESHAPE_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
// CHECK-PER-TENSOR: %[[DYNAMIC_BROADCAST_IN_DIM_0:.+]] = stablehlo.dynamic_broadcast_in_dim %[[ARG_3]], %[[CONCATENATE_0]], dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32, {{.*}}>>, tensor<4xi32>) -> tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[ADD_0:.+]] = stablehlo.add %[[CONVOLUTION_0]], %[[DYNAMIC_BROADCAST_IN_DIM_0]] : tensor<?x3x4x2x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-PER-TENSOR: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ADD_0]] : (tensor<?x3x4x2x!quant.uniform<i32:f32, 8.3371932554046126E-6>>) -> tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>
// CHECK-PER-TENSOR: return %[[UNIFORM_QUANTIZE_0]] : tensor<?x3x4x2x!quant.uniform<i8:f32, 0.0023529412699680704:-128>>

// -----

// Tests that XlaCallModule op is not quantized and converted to func.call without the quantfork.stats ops.

module attributes {tf_saved_model.semantics} {
  func.func private @not_quantized_without_stats_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<3.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %1 = "tf.XlaCallModule"(%arg0, %cst) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _stableghlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable",   device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
// Check that "tf.Const" is converted to stablehlo.constant. XlaCallModule is
// not quantized.

// CHECK: func.func private @not_quantized_without_stats_fn(%[[ARG_0:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
// CHECK: %[[CONST_0:.+]] = stablehlo.constant dense<3.000000e-01> : tensor<2x3xf32>
// CHECK: %[[CALL:.+]] = call @composite_dot_general_fn(%[[ARG_0]], %[[CONST_0]]) : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
// CHECK: return %[[CALL]]

  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK: func.func private @composite_dot_general_fn(%[[ARG_1:.+]]: tensor<1x2xf32>, %[[ARG_2:.+]]: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module}
// Check that the composite_dot_general_fn is untouched.
// CHECK: %[[DOT_GENERAL_0:.+]] = stablehlo.dot_general %[[ARG_1]], %[[ARG_2]]
// CHECK: return %[[DOT_GENERAL_0]]
}

// -----

// Tests that basic `stablehlo.gather` is properly quantized.

module attributes {tf_saved_model.semantics} {
// CHECK: func.func private @quantize_gather_fn(%[[ARG:.+]]: tensor<3x4x2xf32>) -> tensor<2x3x2x2xf32> attributes {tf._original_func_name = "main_0"}
  func.func private @quantize_gather_fn(%arg: tensor<3x4x2xf32>) -> tensor<2x3x2x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<1> : tensor<2x3x2xi32>} : () -> tensor<2x3x2xi32>
    %0 = "quantfork.stats"(%arg) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<2x3x2x2>], _entry_function = @composite_gather_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_gather_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<3x4x2xf32>, tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %2 : tensor<2x3x2x2xf32>
  }
// Checks that the quantized XlaCallModule has been replaced by a CallOp, which
// calls the quantized entry function.
// CHECK: %[[CONST:.+]] = stablehlo.constant dense<{{.*}}> : tensor<2x3x2xi32>
// CHECK: %[[UNIFORM_QUANTIZE:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<3x4x2xf32>) -> tensor<3x4x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL:.+]] = call @quantized_gather_fn(%[[UNIFORM_QUANTIZE]], %[[CONST]]) {_quantization_method = "static_range_ptq { }"} : (tensor<3x4x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x2xi32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE:.+]] = stablehlo.uniform_dequantize %[[CALL]] : (tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>) -> tensor<2x3x2x2xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE]] : tensor<2x3x2x2xf32>

// CHECK: func.func private @quantized_gather_fn(%[[ARG_0:.+]]: tensor<3x4x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_1:.+]]: tensor<2x3x2xi32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_gather_fn(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32> attributes {_from_xla_call_module} {
    %0 = "stablehlo.gather"(%arg0, %arg1) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [1, 0],
        index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<3x4x2xf32>, tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32>
    return %0 : tensor<2x3x2x2xf32>
  }
// CHECK: %[[GATHER:.+]] = "stablehlo.gather"(%[[ARG_0]], %[[ARG_1]]) {{.*}} : (tensor<3x4x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x2xi32>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[GATHER]] : tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE_0]] : tensor<2x3x2x2x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that a basic `stablehlo.add` and a fused `stablehlo.dot_general`
// are properly quantized.

module attributes {tf_saved_model.semantics} {
// CHECK: func.func private @quantize_add_fn(%[[ARG:.+]]: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"}
  func.func private @quantize_add_fn(%arg: tensor<1x2xf32>) -> tensor<1x3xf32> attributes {tf._original_func_name = "main_0"} {
    %cst_0 = "tf.Const"() {value = dense<1.00000000e-1> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
    %cst_1 = "tf.Const"() {value = dense<1.00000000e-1> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %0 = "quantfork.stats"(%arg) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst_0) {Sout = [#tf_type.shape<1x2>], _entry_function = @composite_add_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_add_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %3 = "quantfork.stats"(%2) {layerStats = dense<[5.00000000e-6, 6.00000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = "tf.XlaCallModule"(%3, %cst_1) {Sout = [#tf_type.shape<1x3>], _entry_function = @composite_dot_general_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_dot_general_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %5 = "quantfork.stats"(%4) {layerStats = dense<[5.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %5 : tensor<1x3xf32>
  }
// CHECK: %[[CONST:.+]] = stablehlo.constant() <{value = dense<127> : tensor<1x2xi8>}> : () -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CONST_0:.+]] = stablehlo.constant() <{value = dense<127> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE:.+]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL:.+]] = call @quantized_add_fn(%[[UNIFORM_QUANTIZE]], %[[CONST]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE:.+]] = stablehlo.uniform_dequantize %[[CALL]] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2xf32>
// CHECK: %[[UNIFORM_QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[UNIFORM_DEQUANTIZE]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CALL_0:.+]] = call @quantized_dot_general_fn(%[[UNIFORM_QUANTIZE_0]], %[[CONST_0]]) {_quantization_method = "static_range_ptq { }"} : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[UNIFORM_DEQUANTIZE_0:.+]] = stablehlo.uniform_dequantize %[[CALL_0]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK: return %[[UNIFORM_DEQUANTIZE_0]] : tensor<1x3xf32>

// CHECK: func.func private @quantized_add_fn(%[[ARG_0:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_1:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_add_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
// CHECK: %[[ADD:.+]] = stablehlo.add %arg0, %arg1 : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[ADD]] : tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>

// CHECK: func.func private @quantized_dot_general_fn(%[[ARG_0:.+]]: tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, %[[ARG_1:.+]]: tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>> attributes {_from_xla_call_module}
  func.func private @composite_dot_general_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1,{{.*}}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>
// CHECK: %[[UNIFORM_QUANTIZE:.+]] = stablehlo.uniform_quantize %[[DOT_GENERAL]] : (tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[UNIFORM_QUANTIZE]] : tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
}

// -----

// Tests that `stablehlo.add` is not quantized and emits error when the function
// does not include two ops.

module attributes {tf_saved_model.semantics} {
  func.func private @not_quantize_fn_when_not_singular(%arg: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<1.00000000e-1> : tensor<1x2xf32>} : () -> tensor<1x2xf32>
    %0 = "quantfork.stats"(%arg) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<1x2>], _entry_function = @composite_add_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_add_fn", _quantization_method = "static_range_ptq { }", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    // expected-error@+1 {{'stablehlo.uniform_dequantize' op operand #0 must be ranked tensor of per-tensor integer quantized or per-axis integer quantized values, but got 'tensor<1x2xf32>'}}
    %2 = "quantfork.stats"(%1) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    return %2 : tensor<1x2xf32>
  }

  func.func private @composite_add_fn(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<1x2xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<1x2xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<1x2xf32>
    return %1 : tensor<1x2xf32>
  }
}

// -----

// Tests that `stablehlo.gather` without `static_range_ptq` is not quantized.

module attributes {tf_saved_model.semantics} {
  func.func private @not_quantize_singular_op_without_static_range_ptq(%arg: tensor<3x4x2xf32>) -> tensor<2x3x2x2xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() {value = dense<1> : tensor<2x3x2xi32>} : () -> tensor<2x3x2xi32>
    %0 = "quantfork.stats"(%arg) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<3x4x2xf32>) -> tensor<3x4x2xf32>
    %1 = "tf.XlaCallModule"(%0, %cst) {Sout = [#tf_type.shape<2x3x2x2>], _entry_function = @composite_gather_fn, _stablehlo_version = "1.0.0", _original_entry_function = "composite_gather_fn", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = "", dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64} : (tensor<3x4x2xf32>, tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32>
    // expected-error@+1 {{'stablehlo.uniform_dequantize' op operand #0 must be ranked tensor of per-tensor integer quantized or per-axis integer quantized values, but got 'tensor<2x3x2x2xf32>'}}
    %2 = "quantfork.stats"(%1) {layerStats = dense<[4.00000000e-6, 9.80000000e-1]> : tensor<2xf32>} : (tensor<2x3x2x2xf32>) -> tensor<2x3x2x2xf32>
    return %2 : tensor<2x3x2x2xf32>
  }

  func.func private @composite_gather_fn(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32> attributes {_from_xla_call_module} {
    %0 = "stablehlo.gather"(%arg0, %arg1) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [1, 0],
        index_vector_dim = 2>,
      slice_sizes = array<i64: 1, 2, 2>,
      indices_are_sorted = false
    } : (tensor<3x4x2xf32>, tensor<2x3x2xi32>) -> tensor<2x3x2x2xf32>
    return %0 : tensor<2x3x2x2xf32>
  }
}
