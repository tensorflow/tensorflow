// RUN: odml-to-stablehlo-opt --uniform-quantized-stablehlo-to-tfl \
// RUN:     --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: uniform_quantize_op
func.func @uniform_quantize_op(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
}
// CHECK: %[[QUANT:.*]] = "tfl.quantize"({{.*}}) {qtype = tensor<2x2x!quant.uniform<i8:f32, 3.000000e+00:127>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+00:127>>
// CHECK: return %[[QUANT]]

// -----

// Tests that the pattern doesn't match when the input tensor's type is a
// quantized type.

// CHECK-LABEL: uniform_quantize_op_quantized_input
func.func @uniform_quantize_op_quantized_input(%arg: tensor<2x2x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
}
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

// Tests that the pattern doesn't match when the output tensor's sotrage type
// is ui16. ui16 storage type for quantized type is not compatible with
// `tfl.quantize`.

// CHECK-LABEL: uniform_quantize_op_uint16_output
func.func @uniform_quantize_op_uint16_output(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>>
}
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

// Tests that the pattern doesn't match when the output tensor's sotrage type
// is i32. i32 storage type for quantized type is not compatible with
// `tfl.quantize`.

// CHECK-LABEL: uniform_quantize_op_i32_output
func.func @uniform_quantize_op_i32_output(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>>
}
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

// CHECK-LABEL: uniform_dequantize_op
func.func @uniform_dequantize_op(%arg: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: %[[DEQUANT:.*]] = "tfl.dequantize"({{.*}}) : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:8>>) -> tensor<2x2xf32>
// CHECK: return %[[DEQUANT]]

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is ui16. ui16 storage type is not compatible with
// `tfl.dequantize`.

// CHECK-LABEL: uniform_dequantize_op_ui16_storage_input
func.func @uniform_dequantize_op_ui16_storage_input(%arg: tensor<2x2x!quant.uniform<ui16:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<ui16:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is i32. i32 storage type is not compatible with
// `tfl.dequantize`.

// CHECK-LABEL: uniform_dequantize_op_i32_storage_input
func.func @uniform_dequantize_op_i32_storage_input(%arg: tensor<2x2x!quant.uniform<i32:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i32:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is i32. i32 storage type is not compatible with
// `tfl.dequantize`.

// CHECK-LABEL: uniform_dequantize_op_return_f64
func.func @uniform_dequantize_op_return_f64(%arg: tensor<2x2x!quant.uniform<i8:f64, 1.000000e+0:8>>) -> tensor<2x2xf64> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i8:f64, 1.000000e+0:8>>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

// CHECK-LABEL: convolution_op
func.func @convolution_op(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+0:-100>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>
// CHECK-DAG: %[[CONST_0:.*]] = arith.constant dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>
// Note that the quantized dimension is 0, and the shape has been transposed
// to (2, 3, 3, 4).
// CHECK-DAG: %[[QCONST_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>} : () -> tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>, value = dense<0> : tensor<2xi32>} : () -> tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>
// Explicit tfl.pad op to reflect explicit padding attribute.
// CHECK: %[[PAD:.*]] = "tfl.pad"(%[[ARG]], %[[CONST_0]]) : (tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>, tensor<4x2xi32>) -> tensor<1x5x5x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>
// CHECK: %[[CONV2D:.*]] = "tfl.conv_2d"(%[[PAD]], %[[QCONST_0]], %[[QCONST_1]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x5x5x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// CHECK-LABEL: convolution_op_non_const_filter
func.func @convolution_op_non_const_filter(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, %arg1: tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %0 : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>

// Confirm that the `stablehlo.convolution` is not converted to `tfl.conv_2d`.
// CHECK: stablehlo.convolution
// CHECK-NOT: tfl.conv_2d

// -----

// Test that if the window padding contains values of 0, tfl.pad op is not
// created and the `padding` attribute is set as "VALID".

// CHECK-LABEL: convolution_op_valid_padding
func.func @convolution_op_valid_padding(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>} : () -> tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[QCONST_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>} : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-NOT: tfl.pad
// CHECK: %[[CONV2D:.*]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test that if the window padding value is missing, tfl.pad op is not
// created and the `padding` attribute is set as "VALID".

// CHECK-LABEL: convolution_op_valid_padding
func.func @convolution_op_valid_padding(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  // The `window` attribute is empty.
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>} : () -> tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[QCONST_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>} : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[CONV2D:.*]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test that if the window stride value is explicitly set, the attribute
// value is transferred to tfl.conv_2d's stridw_h and stride_w values.

// CHECK-LABEL: convolution_strides
func.func @convolution_strides(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  // The stride value is explicitly set to [1, 2].
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK-DAG: %[[CONST:.*]] = arith.constant dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>
// CHECK-DAG: %[[QCONST_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>} : () -> tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>} : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[PAD:.*]] = "tfl.pad"(%arg0, %cst) : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<4x2xi32>) -> tensor<1x5x5x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// Tests that the stride_w is set to 2.
// CHECK: %[[CONV2D:.*]] = "tfl.conv_2d"(%[[PAD]], %[[QCONST_0]], %[[QCONST_1]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 2 : i32} : (tensor<1x5x5x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test full integer quantized dot_general with asymmetric quantized input.

// CHECK-LABEL: dot_general_full_integer_asym_input
func.func @dot_general_full_integer_asym_input(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.*]] =  "tfl.pseudo_qconst"() {qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) {adj_x = false, adj_y = false} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test full integer quantized dot_general with symmetric quantized input.

// CHECK-LABEL: dot_general_full_integer_sym_input
func.func @dot_general_full_integer_sym_input(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}

// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.*]] =  "tfl.pseudo_qconst"()
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) {adj_x = false, adj_y = false}

// -----

// Test full integer quantized dot_general with activation as RHS

// CHECK-LABEL: dot_general_full_integer_activation_rhs
func.func @dot_general_full_integer_activation_rhs(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, %arg1: tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %0 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK: "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test full integer quantized dot_general with adj_x

// CHECK-LABEL: dot_general_full_integer_adj_x
func.func @dot_general_full_integer_adj_x(%arg0: tensor<1x2x4x3x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      // implicit transpose of lhs
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x4x3x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}

// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x4x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.*]] =  "tfl.pseudo_qconst"() {qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) {adj_x = true, adj_y = false}

// -----

// Test full integer quantized dot_general with adj_y

// CHECK-LABEL: dot_general_full_integer_adj_y
func.func @dot_general_full_integer_adj_y(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x5x4xi8>} : () -> tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      // implicit transpose of rhs
      rhs_contracting_dimensions = [3]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}

// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.*]] =  "tfl.pseudo_qconst"() {qtype = tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x5x4xi8>} : () -> tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) {adj_x = false, adj_y = true}

// -----

// Test full integer quantized dot_general with wrong batch dims

// CHECK-LABEL: dot_general_full_integer_too_many_batches
func.func @dot_general_full_integer_too_many_batches(%arg0: tensor<1x1x1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x1x1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x1x1x2x4x5xi8>} : () -> tensor<1x1x1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1, 2, 3],
      rhs_batching_dimensions = [0, 1, 2, 3],
      lhs_contracting_dimensions = [5],
      rhs_contracting_dimensions = [4]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x1x1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x1x1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x1x1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x1x1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// Only support size(batching_dimensions) <= 3
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Test full integer quantized dot_general with too many contracting dimension

// CHECK-LABEL: dot_general_full_integer_too_many_contractions
func.func @dot_general_full_integer_too_many_contractions(%arg0: tensor<1x2x3x4x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x4x5xi8>} : () -> tensor<1x2x4x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3, 4],
      rhs_contracting_dimensions = [2, 3]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x4x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// Only support size(contracting_dimensions) == 1
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Test full integer quantized dot_general with unsupported contracting dim

// CHECK-LABEL: dot_general_full_integer_wrong_contracting
func.func @dot_general_full_integer_wrong_contracting(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x4x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 3],
      rhs_batching_dimensions = [0, 2],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x4x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x4x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}

// Contracting dimension must be the last two dimension
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Test full integer quantized dot_general with float operands

// CHECK-LABEL: dot_general_full_integer_float_operands
func.func @dot_general_full_integer_float_operands(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
  return %0 : tensor<1x2x3x5xf32>
}
// Do nothing for float operands
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Test full integer quantized dot_general with asymmetric weight (rhs).

// CHECK-LABEL: dot_general_full_integer_asym_weight
func.func @dot_general_full_integer_asym_weight(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0:5>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0:5>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.*]] =  "tfl.pseudo_qconst"() {qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00:5>>, value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00:5>>
// CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) {adj_x = false, adj_y = false} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00:5>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Test that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized, it is converted to `tfl.fully_connected` op.

// CHECK-LABEL: dot_general_per_axis_quantized_filter
func.func @dot_general_per_axis_quantized_filter(%arg0: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<3x2xi8>} : () -> tensor<3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// CHECK-SAME: %[[ARG_0:.*]]: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>
// Weight tensor is transposed, as tfl.fully_connected accepts a [o, i] matrix.
// CHECK-DAG: %[[QCONST_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x3x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<1> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x!quant.uniform<i32<-128:127>:f32:0, {1.000000e+08,1.500000e+09}>>, value = dense<0> : tensor<2xi32>} : () -> tensor<2x!quant.uniform<i32<-128:127>:f32:0, {1.000000e+08,1.500000e+09}>>
// Bias tensor's scale is input scale * filter scale.
// CHECK: %[[FC:.*]] = "tfl.fully_connected"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<2x3x!quant.uniform<i8:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32<-128:127>:f32:0, {1.000000e+08,1.500000e+09}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
// CHECK-NEXT: return %[[FC]] : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>

// -----

// Test that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has a batch dimension, it is not converted.

// CHECK-LABEL: dot_general_per_axis_quantized_filter_with_batch_dim
func.func @dot_general_per_axis_quantized_filter_with_batch_dim(%arg0: tensor<1x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x3x2xi8>} : () -> tensor<1x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<1x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>) -> tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul

// -----

// Test that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has a batch dim > 1, it is not converted.

// CHECK-LABEL: dot_general_per_axis_quantized_filter_multibatch
func.func @dot_general_per_axis_quantized_filter_multibatch(%arg0: tensor<3x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<3x3x2xi8>} : () -> tensor<3x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<3x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>) -> tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul

// -----

// Test that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has more than one contracting dimension, it is not converted.

// CHECK-LABEL: dot_general_per_axis_quantized_filter_with_multiple_contracting_dims
func.func @dot_general_per_axis_quantized_filter_with_multiple_contracting_dims(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x3x2xi8>} : () -> tensor<1x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1, 2] x [2, 1] : (tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<1x3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>) -> tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul
