// RUN: stablehlo-quant-opt %s -stablehlo-test-post-calibration-component \
// RUN:   -split-input-file | FileCheck %s
// RUN: stablehlo-quant-opt %s -stablehlo-test-post-calibration-component='unpack-quantized-types=false' \
// RUN:   -split-input-file | FileCheck %s --check-prefix=CHECK-NO-UNPACK

// Tests that a simple dot_general (lifted as a function) with CustomAggregators
// around it is quantized. The resulting graph has quantized types unpacked into
// int ops.
func.func @main(%arg0: tensor<1x1024xf32>) -> tensor<1x3xf32> {
  %0 = "tf.Const"() <{value = dense<0.5> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
  %1:4 = "tf.CustomAggregator"(%arg0) <{id = "1", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> {min = 7.547870e-07 : f32, max = 0.999992311 : f32} : (tensor<1x1024xf32>) -> (tensor<1x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %2 = "tf.XlaCallModule"(%1#0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  %3:4 = "tf.CustomAggregator"(%2) <{id = "2", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> {min = -17.5216827 : f32, max = 18.3033524 : f32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  return %3#0 : tensor<1x3xf32>
}
func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x1024xf32>) -> tensor<1x3xf32>

// Tests that the dot_general accepts i8 tensors and outputs an i32 tensor.
// Note: Argument quantization sequence omitted.
// CHECK: stablehlo.dot_general %{{.+}}, %{{.+}}, contracting_dims = [1] x [0] : (tensor<1x1024xi8>, tensor<1024x3xi8>) -> tensor<1x3xi32>

// Note: Result dequantization sequence omitted.
// CHECK: return %{{.+}} : tensor<1x3xf32>

// -----

// Tests that a simple dot_general (lifted as a function) with CustomAggregators
// around it is quantized, when the 'unpack-quantized-types' option is set to
// false. This test case inputs the same graph as the test above. Note that now
// the uniform quantized types are directly expressed within the graph.

func.func @main_no_unpack(%arg0: tensor<1x1024xf32>) -> tensor<1x3xf32> {
  %0 = "tf.Const"() <{value = dense<0.5> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
  %1:4 = "tf.CustomAggregator"(%arg0) <{id = "1", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> {device = "", max = 0.999992311 : f32, min = 7.547870e-07 : f32} : (tensor<1x1024xf32>) -> (tensor<1x1024xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  %2 = "tf.XlaCallModule"(%1#0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  %3:4 = "tf.CustomAggregator"(%2) <{id = "2", calibration_method = 1 : i32, num_bins = 0 : i32, max_percentile = 0.000000e+00 : f32, min_percentile = 0.000000e+00 : f32}> {device = "", max = 18.3033524 : f32, min = -17.5216827 : f32} : (tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<f32>, tensor<f32>, tensor<*xi64>)
  return %3#0 : tensor<1x3xf32>
}
func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK-NO-UNPACK-LABEL: func.func @main_no_unpack
// CHECK-NO-UNPACK-SAME: (%[[ARG_0:.+]]: tensor<1x1024xf32>) -> tensor<1x3xf32>
// CHECK-NO-UNPACK-DAG: %[[CONST:.+]] = stablehlo.constant() <{value = dense<{{.*}}> : tensor<1024x3xi8>}> : () -> tensor<1024x3x!quant.uniform<i8<-127:127>:f32:1, {{.*}}>>
// CHECK-NO-UNPACK: %[[QUANTIZE_0:.+]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<1x1024xf32>) -> tensor<1x1024x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-NO-UNPACK: %[[DOT:.+]] = stablehlo.dot_general %[[QUANTIZE_0]], %[[CONST]]
// CHECK-NO-UNPACK: %[[QUANTIZE_1:.+]] = stablehlo.uniform_quantize %[[DOT]] : (tensor<1x3x!quant.uniform<i32:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK-NO-UNPACK: %[[DEQUANTIZE:.+]] = stablehlo.uniform_dequantize %[[QUANTIZE_1]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NO-UNPACK: return %[[DEQUANTIZE]] : tensor<1x3xf32>

// -----

// Tests that a simple dot_general without CustomAggregators is not quantized.

func.func @main(%arg0: tensor<1x1024xf32>) -> tensor<1x3xf32> {
  %0 = "tf.Const"() <{value = dense<0.5> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
  %2 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _quantization_method = "static_range_ptq {}", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}
func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK: func.func @main
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x1024xf32>) -> tensor<1x3xf32>
// CHECK-DAG: %[[CONST_0:.+]] = stablehlo.constant dense<{{.*}}> : tensor<1024x3xf32>
// CHECK: stablehlo.dot_general %[[ARG_0]], %[[CONST_0]]
// CHECK-NOT: tf.XlaCallModule
