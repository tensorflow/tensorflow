// RUN: odml-to-stablehlo-opt --uniform-quantized-stablehlo-to-tfl \
// RUN:     --split-input-file --verify-diagnostics %s | FileCheck %s

// ============================================================================
// The following functions tests example quantization patterns outputted from
// JAX Quantizer. JAX Quantizer should output integer types, which are
// composed into `UniformQuantized{|PerAxis}Type` via
// `compose_uniform_quantized_type_pass.cc`.
// ============================================================================

func.func @uniform_quantize_op(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
}
// CHECK-LABEL: uniform_quantize_op
// CHECK: %[[QUANT:.+]] = "tfl.quantize"({{.*}}) <{qtype = tensor<2x2x!quant.uniform<i8:f32, 3.000000e+00:127>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+00:127>>
// CHECK: return %[[QUANT]]

// -----

// Tests that the pattern doesn't match when the input tensor's type is a
// quantized type.

func.func @uniform_quantize_op_quantized_input(%arg: tensor<2x2x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 3.000000e+0:127>>
}
// CHECK-LABEL: uniform_quantize_op_quantized_input
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

// Tests that the pattern doesn't match when the output tensor's storage type
// is ui16. ui16 storage type for quantized type is not compatible with
// `tfl.quantize`.

func.func @uniform_quantize_op_uint16_output(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<ui16:f32, 3.000000e+0:127>>
}
// CHECK-LABEL: uniform_quantize_op_uint16_output
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

// Tests that the pattern doesn't match when the output tensor's storage type
// is i32. i32 storage type for quantized type is not compatible with
// `tfl.quantize`.

func.func @uniform_quantize_op_i32_output(%arg: tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>> {
  %0 = stablehlo.uniform_quantize %arg : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>>
  return %0 : tensor<2x2x!quant.uniform<i32:f32, 3.000000e+0:127>>
}
// CHECK-LABEL: uniform_quantize_op_i32_output
// CHECK: stablehlo.uniform_quantize
// CHECK-NOT: tfl.quantize

// -----

func.func @uniform_dequantize_op(%arg: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: uniform_dequantize_op
// CHECK: %[[DEQUANT:.+]] = "tfl.dequantize"({{.*}}) : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:8>>) -> tensor<2x2xf32>
// CHECK: return %[[DEQUANT]]

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is ui16. ui16 storage type is not compatible with
// `tfl.dequantize`.

func.func @uniform_dequantize_op_ui16_storage_input(%arg: tensor<2x2x!quant.uniform<ui16:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<ui16:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: uniform_dequantize_op_ui16_storage_input
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is i32. i32 storage type is not compatible with
// `tfl.dequantize`.

func.func @uniform_dequantize_op_i32_storage_input(%arg: tensor<2x2x!quant.uniform<i32:f32, 1.000000e+0:8>>) -> tensor<2x2xf32> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i32:f32, 1.000000e+0:8>>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: uniform_dequantize_op_i32_storage_input
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

// Tests that the pattern doesn't match when the input quantized tensor's
// storage type is i32. i32 storage type is not compatible with
// `tfl.dequantize`.

func.func @uniform_dequantize_op_return_f64(%arg: tensor<2x2x!quant.uniform<i8:f64, 1.000000e+0:8>>) -> tensor<2x2xf64> {
  %0 = stablehlo.uniform_dequantize %arg : (tensor<2x2x!quant.uniform<i8:f64, 1.000000e+0:8>>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
// CHECK-LABEL: uniform_dequantize_op_return_f64
// CHECK: stablehlo.uniform_dequantize
// CHECK-NOT: tfl.dequantize

// -----

func.func @convolution_upstream_same_padding_srq(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+0:-100>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// Note that the quantized dimension is 0, and the shape has been transposed
// to (2, 3, 3, 4).
// CHECK-LABEL: convolution_upstream_same_padding_srq
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>}> : () -> tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>
// CHECK: %[[CONV2D:.+]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x3x3x4x!quant.uniform<i8:f32, 3.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {6.000000e+02,9.000000e+03}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

func.func @convolution_upstream_srq_non_const_filter(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, %arg1: tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %0 : tensor<1x3x3x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// Confirm that the `stablehlo.convolution` is not converted to `tfl.conv_2d`.
// CHECK-LABEL: convolution_upstream_srq_non_const_filter
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: stablehlo.convolution
// CHECK-NOT: tfl.conv_2d

// -----

// Tests that if the window padding contains values of 0, tfl.pad op is not
// created and the `padding` attribute is set as "VALID".

func.func @convolution_upstream_srq_valid_padding(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[0, 0], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-LABEL: convolution_upstream_srq_valid_padding
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>}> : () -> tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-NOT: tfl.pad
// CHECK: %[[CONV2D:.+]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests that if the window padding value is missing, tfl.pad op is not
// created and the `padding` attribute is set as "VALID".

func.func @convolution_upstream_srq_valid_padding(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  // The `window` attribute is empty.
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-LABEL: convolution_upstream_srq_valid_padding
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>}> : () -> tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK: %[[CONV2D:.+]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x1x1x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests that if the window stride value is explicitly set, the attribute
// value is transferred to tfl.conv_2d's stridw_h and stride_w values.

func.func @convolution_upstream_srq_strides(%arg0: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  // The stride value is explicitly set to [1, 2].
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-LABEL: convolution_upstream_srq_strides
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>}> : () -> tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>
// Tests that the stride_w is set to 2.
// CHECK: %[[CONV2D:.+]] = "tfl.conv_2d"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 2 : i32}> : (tensor<1x3x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+00>>
// CHECK: return %[[CONV2D]] : tensor<1x3x2x2x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests static range quantized dot_general with asymmetric quantized input.

func.func @dot_general_upstream_srq_asym_input(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_asym_input
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>}> : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[BMM:.+]] = "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) <{adj_x = false, adj_y = false}> : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests static range quantized dot_general with symmetric quantized input.

func.func @dot_general_upstream_srq_sym_input(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_sym_input
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"()
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) <{adj_x = false, adj_y = false}>

// -----

// Tests static range quantized dot_general with activation as RHS

func.func @dot_general_upstream_srq_activation_rhs(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>, %arg1: tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_activation_rhs
// CHECK: "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false}> : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests static range quantized dot_general with adj_x

// CHECK-LABEL: dot_general_upstream_srq_adj_x
func.func @dot_general_upstream_srq_adj_x(%arg0: tensor<1x2x4x3x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x4x3x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>}> : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) <{adj_x = true, adj_y = false}>

// -----

// Tests static range quantized dot_general with adj_y

func.func @dot_general_upstream_srq_adj_y(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_adj_y
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x5x4xi8>}> : () -> tensor<1x2x5x4x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) <{adj_x = false, adj_y = true}>

// -----

// Tests static range quantized dot_general with wrong batch dims

func.func @dot_general_upstream_srq_too_many_batches(%arg0: tensor<1x1x1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x1x1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_too_many_batches
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Tests static range quantized dot_general with too many contracting dimension

func.func @dot_general_upstream_srq_too_many_contractions(%arg0: tensor<1x2x3x4x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_too_many_contractions
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Tests static range quantized dot_general with unsupported contracting dim

func.func @dot_general_upstream_srq_wrong_contracting(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x4x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
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
// CHECK-LABEL: dot_general_upstream_srq_wrong_contracting
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.batch_matmul

// -----

// Tests static range quantized dot_general with float operands

// CHECK-LABEL: dot_general_upstream_srq_float_operands
func.func @dot_general_upstream_srq_float_operands(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32> {
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

// Tests static range quantized dot_general with asymmetric weight (rhs).

// CHECK-LABEL: dot_general_upstream_srq_asym_weight
func.func @dot_general_upstream_srq_asym_weight(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0:0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]>, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+0:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0:0>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+0>>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>}> : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: %[[BMM:.+]] = "tfl.batch_matmul"(%[[ARG]], %[[QCONST_0]]) <{adj_x = false, adj_y = false}> : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:-100>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 4.000000e+00>>

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized, it is converted to `tfl.fully_connected` op.

// CHECK-LABEL: dot_general_upstream_srq_per_axis_quantized_filter
func.func @dot_general_upstream_srq_per_axis_quantized_filter(%arg0: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<3x2xi8>} : () -> tensor<3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<3x2x!quant.uniform<i8:f32:1,{2.000000e+02, 3.000000e+03}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// CHECK-SAME: %[[ARG_0:.+]]: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>
// Weight tensor is transposed, as tfl.fully_connected accepts a [o, i] matrix.
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<1> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {1.000000e+08,1.500000e+09}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {1.000000e+08,1.500000e+09}>>
// Bias tensor's scale is input scale * filter scale.
// CHECK: %[[FC:.+]] = "tfl.fully_connected"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, tensor<2x!quant.uniform<i32:f32:0, {1.000000e+08,1.500000e+09}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
// CHECK-NEXT: return %[[FC]] : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has a batch dimension, it is not converted.

// CHECK-LABEL: dot_general_upstream_srq_per_axis_quantized_filter_with_batch_dim
func.func @dot_general_upstream_srq_per_axis_quantized_filter_with_batch_dim(%arg0: tensor<1x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x3x2xi8>} : () -> tensor<1x3x2x!quant.uniform<i8:f32:0,{2.000000e+02}>>
  %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<1x3x2x!quant.uniform<i8:f32:0,{2.000000e+02}>>) -> tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has a batch dim > 1, it is not converted.

// CHECK-LABEL: dot_general_upstream_srq_per_axis_quantized_filter_multibatch
func.func @dot_general_upstream_srq_per_axis_quantized_filter_multibatch(%arg0: tensor<3x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<3x3x2xi8>} : () -> tensor<3x3x2x!quant.uniform<i8:f32:2,{2.000000e+02, 3.000000e+03}>>
  %1 = stablehlo.dot_general %arg0, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<3x1x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<3x3x2x!quant.uniform<i8:f32:2,{2.000000e+02, 3.000000e+03}>>) -> tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<3x1x2x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` is per-axis
// quantized but has more than one contracting dimension, it is not converted.

// CHECK-LABEL: dot_general_upstream_srq_per_axis_quantized_filter_with_multiple_contracting_dims
func.func @dot_general_upstream_srq_per_axis_quantized_filter_with_multiple_contracting_dims(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>) -> tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x3x2xi8>} : () -> tensor<1x3x2x!quant.uniform<i8:f32:0,{2.000000e+02}>>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1, 2] x [2, 1] : (tensor<1x2x3x!quant.uniform<i8:f32, 5.000000e+05:-100>>, tensor<1x3x2x!quant.uniform<i8:f32:0,{2.000000e+02}>>) -> tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>>
  return %1 : tensor<1x1x!quant.uniform<i8:f32, 4.000000e+04:127>>
}
// Nothing changes.
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul

// -----

// ============================================================================
// The following functions tests example quantization patterns outputted from
// StableHLO Quantizer. These patterns should be legalized early directly
// to fused tflite ops.
// ============================================================================

// Tests that a simple per-channel quantized `stablehlo.dot_general` is properly
// lowered to fused `tfl.fully_connected`.
// This case covers for the following quantization patterns because
// activation clipping ranges take affect in scale and zp of the final
// `stablehlo.uniform_quantize`. See more details in b/319168201.
// * dot_general_fn
// * dot_general_with_relu_fn
// * dot_general_with_relu6_fn

func.func @dot_general_srq(%arg0: tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+0:0>>) -> (tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>) {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1024x3xi8>} : () -> tensor<1024x3x!quant.uniform<i8<-127:127>:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+0:0>>, tensor<1024x3x!quant.uniform<i8<-127:127>:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
  return %2 : tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
}
// CHECK-LABEL: dot_general_srq
// CHECK-SAME: (%[[ARG_1:.+]]: tensor<1x1024x!quant.uniform<i8:f32, {{.*}}>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
// CHECK-NOT: stablehlo.dot_general
// CHECK: %[[QCONST_0:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, value = dense<1> : tensor<3x1024xi8>}> : () -> tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>
// CHECK: %[[QCONST_1:.+]] =  "tfl.pseudo_qconst"() <{qtype = tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, value = dense<0> : tensor<3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>
// CHECK: %[[FULLY_CONNECTED:.+]] =  "tfl.fully_connected"(%[[ARG_1]], %[[QCONST_0]], %[[QCONST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
// CHECK-NOT: tfl.batch_matmul
// CHECK: return %[[FULLY_CONNECTED]]

// -----

// Tests that a fused per-channel quantized `stablehlo.dot_general` is properly
// lowered to fused `tfl.fully_connected`.
// TODO: b/309896242 - Add more support for dynamic bias fusion cases.

func.func @dot_general_with_bias_same_shape_srq(%arg0: tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+0:0>>) -> (tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>) {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1024x3xi8>} : () -> tensor<1024x3x!quant.uniform<i8<-127:127>:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %1 = stablehlo.constant() {value = dense<2> : tensor<1x3xi32>} : () -> tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+0:0>>, tensor<1024x3x!quant.uniform<i8<-127:127>:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %3 = stablehlo.add %2, %1 : tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  %4 = stablehlo.uniform_quantize %3 : (tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
  return %4 : tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
}
// CHECK-LABEL: dot_general_with_bias_same_shape
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, value = dense<1> : tensor<3x1024xi8>}> : () -> tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, value = dense<2> : tensor<1x3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>
// CHECK: %[[FULLY_CONNECTED:.+]] = "tfl.fully_connected"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x1024x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<3x1024x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>, tensor<3x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,2.000000e+00}>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+00:-127>>
// CHECK: return %[[FULLY_CONNECTED]]

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` has a
// `stablehlo.constant` -> `stablehlo.transpose` pattern, the
// `stablehlo.constant` is directly transformed to `tfl.pseudo_qconst`, which
// becomes the rhs of `tfl.fully_connected`. This is because
// `tfl.fully_connected` accepts a [o, i] format for rhs, which
// `stablehlo.constant` op already has before the transpose.

// CHECK-LABEL: dot_general_srq_constant_transpose_rhs
func.func @dot_general_srq_constant_transpose_rhs(%arg0: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8:f32:0, {3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3x!quant.uniform<i8:f32:0, {3.000000e+00, 3.000000e+00}>>) -> tensor<3x2x!quant.uniform<i8:f32:1, {3.000000e+00, 3.000000e+00}>>
  %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>, tensor<3x2x!quant.uniform<i8:f32:1, {3.000000e+00, 3.000000e+00}>>) -> tensor<1x2x!quant.uniform<i32:f32:1, {2.000000e+00, 2.000000e+00}>>
  %3 = stablehlo.uniform_quantize %2 : (tensor<1x2x!quant.uniform<i32:f32:1, {2.000000e+00, 2.000000e+00}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>
  return %3 : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>
}
// CHECK-SAME: %[[ARG:.+]]: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>

// Checks that the `tfl.pseudo_qconst` corresponding to the `stablehlo.constant`
// has the same shape.
// CHECK: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x!quant.uniform<i8:f32:0, {3.000000e+00,3.000000e+00}>>, value = dense<1> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8:f32:0, {3.000000e+00,3.000000e+00}>>
// CHECK: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {1.500000e+01,1.500000e+01}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {1.500000e+01,1.500000e+01}>>
// CHECK: %[[FULLY_CONNECTED:.+]] = "tfl.fully_connected"(%[[ARG]], %[[QCONST_0]], %[[QCONST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>, tensor<2x3x!quant.uniform<i8:f32:0, {3.000000e+00,3.000000e+00}>>, tensor<2x!quant.uniform<i32:f32:0, {1.500000e+01,1.500000e+01}>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>

// Also checks that the i32 -> i8 uniform quantize is absorbed into
// `tfl.fully_connected`.
// CHECK: return %[[FULLY_CONNECTED]] : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>

// -----

// Tests that when the weight tensor for `stablehlo.dot_general` is coming from
// `stablehlo.transpose` and its operand is not a `stablehlo.constant`
// (e.g. argument), the conversion to `tfl.fully_connected` doesn't happen.

// CHECK-LABEL: dot_general_srq_arg_transpose_rhs
func.func @dot_general_srq_arg_transpose_rhs(%arg0: tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>, %arg1: tensor<2x3x!quant.uniform<i8:f32, 3.000000e+00>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>> {
  %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x3x!quant.uniform<i8:f32, 3.000000e+00>>) -> tensor<3x2x!quant.uniform<i8:f32, 3.000000e+00>>
  %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<1x3x!quant.uniform<i8:f32, 5.000000e+00:-128>>, tensor<3x2x!quant.uniform<i8:f32, 3.000000e+00>>) -> tensor<1x2x!quant.uniform<i32:f32, 2.000000e+00>>
  %3 = stablehlo.uniform_quantize %2 : (tensor<1x2x!quant.uniform<i32:f32, 2.000000e+00>>) -> tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>
  return %3 : tensor<1x2x!quant.uniform<i8:f32, 4.000000e+00:7>>
}
// Checks that the `stablehlo.dot_general` is not converted to
// `tfl.fully_connected`. Also notice that the `stablehlo.transpose` and
// `stablehlo.uniform_quantize` are converted separately.

// CHECK: tfl.transpose
// CHECK: stablehlo.dot_general
// CHECK-NOT: tfl.fully_connected
// CHECK: tfl.quantize

// -----

// Tests static range quantized dot_general with qi32 -> qi8 requantization is
// properly lowered to `tfl.batch_matmul`.

func.func @dot_general_srq_to_batch_matmul(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, %arg1: tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>
  %1 = stablehlo.uniform_quantize %0 : (tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
  return %1 : tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
}

// CHECK-LABEL: dot_general_srq_to_batch_matmul
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, %[[ARG_1:.+]]: tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
// CHECK: %[[BMM:.+]] = "tfl.batch_matmul"(%[[ARG_0]], %[[ARG_1]]) <{adj_x = false, adj_y = false}> : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
// CHECK-NOT: stablehlo.dot_general
// CHECK-NOT: stablehlo.uniform_quantize
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.quantize
// CHECK: return %[[BMM]]

// -----

// Tests static range quantized dot_general with qi32 -> qi8 requantization is
// not converted to `tfl.batch_matmul` when there are multiple use of the
// intermediate result.

func.func @dot_general_srq_multiple_use_of_intermediate_result(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, %arg1: tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4x!quant.uniform<i8:f32, 1.000000e+00:3>>, tensor<1x2x4x5x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>
  %1 = stablehlo.uniform_quantize %0 : (tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
  %2 = stablehlo.uniform_quantize %0 : (tensor<1x2x3x5x!quant.uniform<i32:f32, 4.000000e+00:1>>) -> tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
  %3 = stablehlo.add %1, %2 : tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
  return %3 : tensor<1x2x3x5x!quant.uniform<i8:f32, 6.000000e+00:5>>
}

// CHECK-LABEL: dot_general_srq_multiple_use_of_intermediate_result
// CHECK-NOT: tfl.fully_connected
// CHECK-NOT: tfl.batch_matmul
// CHECK: stablehlo.dot_general

// -----

// Tests that a simple per-channel quantized `stablehlo.convolution` is properly
// lowered to fused `tfl.conv_2d`.
// This case covers for the following quantization patterns because
// activation clipping ranges take affect in scale and zp of the final
// `stablehlo.uniform_quantize`. See more details in b/319168201.
// * conv_fn
// * conv_with_relu_fn
// * conv_with_relu6_fn

func.func @conv_srq(%arg0: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
  %0 = stablehlo.constant() {value = dense<3> : tensor<4x4x2x4xi8>} : () -> tensor<4x4x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<4x4x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  return %2 : tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_srq
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[CONST_0:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<4x4x4x2xi8>}> : () -> tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>, value = dense<0> : tensor<4xi32>}> : () -> tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>
// CHECK: %[[PAD:.+]] = "tfl.pad"(%[[ARG_0]], %[[CONST_0]]) : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x2xi32>) -> tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK: %[[CONV_2D:.+]] = "tfl.conv_2d"(%[[PAD]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[CONV_2D]]

func.func @conv_same_padding_srq(%arg0: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x3x2xi8>} : () -> tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  return %2 : tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_same_padding_srq
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<2x3x3x3xi8>}> : () -> tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>
// CHECK: %[[CONV_2D:.+]] = "tfl.conv_2d"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[CONV_2D]] : tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>

// -----

func.func @conv_same_padding_srq_non_unit_strides(%arg0: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x3x2xi8>} : () -> tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, window_strides = array<i64: 2, 2>} : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>) -> tensor<1x16x16x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x16x16x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>) -> tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  return %2 : tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_same_padding_srq_non_unit_strides
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<2x3x3x3xi8>}> : () -> tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>, value = dense<0> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>
// CHECK: %[[CONV_2D:.+]] = "tfl.conv_2d"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>) -> tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[CONV_2D]] : tensor<1x16x16x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>

// -----

func.func @conv_srq_transpose_conv(%arg0: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x14x14x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
  %0 = stablehlo.constant() {value = dense<3> : tensor<2x2x2x4xi8>} : () -> tensor<2x2x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 3, 3>} : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<2x2x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>) -> tensor<1x14x14x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>
  %2 = stablehlo.uniform_quantize %1 : (tensor<1x14x14x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>) -> tensor<1x14x14x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  return %2 : tensor<1x14x14x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_srq_transpose_conv
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x14x14x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[CONST_0:.+]] = arith.constant dense<[1, 14, 14, 4]> : tensor<4xi32>
// CHECK-DAG: %[[CONST_1:.*]] = "tfl.pseudo_const"() <{value = dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x2x2x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<4x2x2x2xi8>}> : () -> tensor<4x2x2x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>, value = dense<0> : tensor<4xi32>}> : () -> tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>
// CHECK: %[[PAD:.+]] = "tfl.pad"(%[[ARG_0]], %[[CONST_1]]) : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x2xi32>) -> tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK: %[[TRANSPOSE_CONV_2D:.+]] = "tfl.transpose_conv"(%[[CONST_0]], %[[QCONST_0]], %[[PAD]], %[[QCONST_1]]) <{fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 4 : i32}> : (tensor<4xi32>, tensor<4x2x2x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>) -> tensor<1x14x14x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[TRANSPOSE_CONV_2D]]

// -----

// Tests that a fused per-channel quantized `stablehlo.convolution` is properly
// lowered to fused `tfl.conv_2d`.
// This case covers for the following quantization patterns because
// activation clipping ranges take affect in scale and zp of the final
// `stablehlo.uniform_quantize`. See more details in b/319168201.
// * conv_with_bias_fn
// * conv_with_bias_and_relu_fn
// * conv_with_bias_and_relu6_fn

func.func @conv_with_bias_and_relu_srq(%arg0: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
    %0 = stablehlo.constant() {value = dense<5> : tensor<1x1x1x4xi32>} : () -> tensor<1x1x1x4x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00}>>
    %1 = stablehlo.constant() {value = dense<3> : tensor<4x4x2x4xi8>} : () -> tensor<4x4x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x1x1x4x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>
    %3 = stablehlo.convolution(%arg0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<4x4x2x4x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>
    %4 = stablehlo.add %3, %2 : tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>
    %5 = stablehlo.uniform_quantize %4 : (tensor<1x4x4x4x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
    return %5 : tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  }
// CHECK-LABEL: func.func @conv_with_bias_and_relu_srq
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[CONST_0:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<4x4x4x2xi8>}> : () -> tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>, value = dense<5> : tensor<1x1x1x4xi32>}> : () -> tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>
// CHECK: %[[PAD:.+]] = "tfl.pad"(%[[ARG_0]], %[[CONST_0]]) : (tensor<1x5x5x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x2xi32>) -> tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK: %[[CONV_2D:.+]] = "tfl.conv_2d"(%[[PAD]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x7x7x2x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x4x4x2x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00,3.000000e+00,3.000000e+00}>>, tensor<4x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00,6.000000e+00}>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[CONV_2D]]

func.func @conv_with_bias_same_padding_srq(%arg0: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
    %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x3x2xi8>} : () -> tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>
    %1 = stablehlo.constant() {value = dense<5> : tensor<1x1x1x2xi32>} : () -> tensor<1x1x1x2x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00}>>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x2x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>
    %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<3x3x3x2x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>
    %4 = stablehlo.add %3, %2 : tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>
    %5 = stablehlo.uniform_quantize %4 : (tensor<1x32x32x2x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
    return %5 : tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_with_bias_same_padding_srq
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<2x3x3x3xi8>}> : () -> tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>, value = dense<5> : tensor<1x1x1x2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>
// CHECK: %[[CONV_2D:.+]] = "tfl.conv_2d"(%[[ARG_0]], %[[QCONST_0]], %[[QCONST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x3x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<2x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {3.000000e+00,3.000000e+00}>>, tensor<2x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00}>>) -> tensor<1x32x32x2x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[CONV_2D]]

func.func @conv_with_bias_same_padding_srq_depthwise(%arg0: tensor<1x4x5x3x!quant.uniform<i8:f32, 2.000000e+00:0>>) -> (tensor<1x5x6x3x!quant.uniform<i8:f32, 8.000000e+00:-128>>) {
  %0 = stablehlo.constant() {value = dense<3> : tensor<2x2x1x3xi8>} : () -> tensor<2x2x1x3x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00}>>
  %1 = stablehlo.constant() {value = dense<5> : tensor<1x1x1x3xi32>} : () -> tensor<1x1x1x3x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00, 2.000000e+00}>>
  %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x3x!quant.uniform<i32:f32:3, {2.000000e+00, 2.000000e+00, 2.000000e+00}>>) -> tensor<1x5x6x3x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00}>>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 3 : i64} : (tensor<1x4x5x3x!quant.uniform<i8:f32, 2.000000e+00:0>>, tensor<2x2x1x3x!quant.uniform<i8:f32:3, {3.000000e+00, 3.000000e+00, 3.000000e+00}>>) -> tensor<1x5x6x3x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00}>>
  %4 = stablehlo.add %3, %2 : tensor<1x5x6x3x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00}>>
  %5 = stablehlo.uniform_quantize %4 : (tensor<1x5x6x3x!quant.uniform<i32:f32:3, {6.000000e+00, 6.000000e+00, 6.000000e+00}>>) -> tensor<1x5x6x3x!quant.uniform<i8:f32, 8.000000e+00:-128>>
  return %5 : tensor<1x5x6x3x!quant.uniform<i8:f32, 8.000000e+00:-128>>
}
// CHECK-LABEL: func.func @conv_with_bias_same_padding_srq_depthwise
// CHECK-SAME: (%[[ARG_0:.+]]: tensor<1x4x5x3x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x5x6x3x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK-DAG: %[[CONST_0:.+]] = "tfl.pseudo_const"() <{value = dense<{{\[\[0, 0\], \[1, 1\], \[1, 1\], \[0, 0\]\]}}> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
// CHECK-DAG: %[[QCONST_0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8<-127:127>:f32:3, {3.000000e+00,3.000000e+00,3.000000e+00}>>, value = dense<3> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8<-127:127>:f32:3, {3.000000e+00,3.000000e+00,3.000000e+00}>>
// CHECK-DAG: %[[QCONST_1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<3x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00}>>, value = dense<5> : tensor<1x1x1x3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00}>>
// CHECK: %[[PAD:.+]] = "tfl.pad"(%[[ARG_0]], %[[CONST_0]]) : (tensor<1x4x5x3x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<4x2xi32>) -> tensor<1x6x7x3x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK: %[[DEPTHWISE_CONV_2D:.+]] = "tfl.depthwise_conv_2d"(%[[PAD]], %[[QCONST_0]], %[[QCONST_1]]) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x7x3x!quant.uniform<i8:f32, 2.000000e+00>>, tensor<1x2x2x3x!quant.uniform<i8<-127:127>:f32:3, {3.000000e+00,3.000000e+00,3.000000e+00}>>, tensor<3x!quant.uniform<i32:f32:0, {6.000000e+00,6.000000e+00,6.000000e+00}>>) -> tensor<1x5x6x3x!quant.uniform<i8:f32, 8.000000e+00:-128>>
// CHECK: return %[[DEPTHWISE_CONV_2D]]

// -----

// Tests that a quantized `stablehlo.transpose` is converted to `tfl.transpose`.

func.func @transpose(
    %arg0: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<4x3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<4x3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<4x3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: transpose
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-NOT: stablehlo.transpose
// CHECK: %[[CST:.+]] = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
// CHECK: %[[TRANSPOSE:.+]] = "tfl.transpose"(%[[ARG0]], %[[CST]]) : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<3xi32>) -> tensor<4x3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[TRANSPOSE]]

// -----

// Tests that a float `stablehlo.transpose` is not converted to `tfl.transpose`.

func.func @transpose_float(%arg0: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
  return %0 : tensor<4x3x2xf32>
}
// CHECK-LABEL: transpose_float
// CHECK-NOT: tfl.transpose
// CHECK: stablehlo.transpose

// -----

// Tests that a quantized `stablehlo.reshape` is converted to `tfl.reshape`.

func.func @reshape(
    %arg0: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<6x4x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<6x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<6x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: reshape
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-NOT: stablehlo.reshape
// CHECK: %[[CST:.+]] = arith.constant dense<[6, 4]> : tensor<2xi32>
// CHECK: %[[RESHAPE:.+]] = "tfl.reshape"(%[[ARG0]], %[[CST]]) : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2xi32>) -> tensor<6x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[RESHAPE]]

// -----

// Tests that a float `stablehlo.reshape` is not converted to `tfl.reshape`.

func.func @reshape_float(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}
// CHECK-LABEL: reshape_float
// CHECK-NOT: tfl.reshape
// CHECK: stablehlo.reshape

// -----

// Tests that a quantized `stablehlo.select` is converted to `tfl.select_v2`.

func.func @select(
    %arg0: tensor<1x3xi1>,
    %arg1: tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    %arg2: tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (
    tensor<1x3xi1>,
    tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: select
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3xi1>, %[[ARG1:.+]]: tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>, %[[ARG2:.+]]: tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-NOT: stablehlo.select
// CHECK: %[[SELECT:.+]] = "tfl.select_v2"(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<1x3xi1>, tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<1x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[SELECT]]

// -----

// Tests that a float `stablehlo.select` is not converted to `tfl.select_v2`.

func.func @select_float(%arg0: tensor<1x3xi1>, %arg1: tensor<1x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<1x3xi1>, tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK-LABEL: select_float
// CHECK-NOT: tfl.select_v2
// CHECK: stablehlo.select

// -----

// Tests that a quantized `stablehlo.concatenate` is converted to
// `tfl.concatenation`.

func.func @concatenate(
    %arg0: tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    %arg1: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (
    tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<4x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: concatenate
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>, %[[ARG1:.+]]: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-NOT: stablehlo.concatenate
// CHECK: %[[CONCAT:.+]] = "tfl.concatenation"(%arg0, %arg1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[CONCAT]]

// -----

// Tests that a float `stablehlo.concatenate` is not converted to
// `tfl.concatenation`.

func.func @concatenate_float(%arg0: tensor<3x2xf32>, %arg1: tensor<1x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x2xf32>, tensor<1x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// CHECK-LABEL: concatenate_float
// CHECK-NOT: tfl.concatenation
// CHECK: stablehlo.concatenate

// -----

// Tests that a quantized `stablehlo.pad` without interior padding is
// converted to `tfl.padv2`.

func.func @pad_without_interior_padding(
    %arg0: tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<4x5x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [0, 0] : (
    tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<4x5x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<4x5x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: pad_without_interior_padding
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-SAME: %[[ARG1:.+]]: tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: %[[PADDING:.+]] = arith.constant
// CHECK{LITERAL}: dense<[[0, 2], [1, 1]]> : tensor<2x2xi32>
// CHECK: %[[PAD:.+]] = "tfl.padv2"(%[[ARG0]], %[[PADDING]], %[[ARG1]]) : (tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2x2xi32>, tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<4x5x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[PAD]]

// -----

// Tests that a quantized `stablehlo.pad` with interior padding is converted to
// `tfl.dilate` and `tfl.padv2`.

func.func @pad_with_interior_padding(
    %arg0: tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    %arg1: tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<5x9x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [1, 2] : (
    tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>,
    tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<5x9x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<5x9x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: pad_with_interior_padding
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-SAME: %[[ARG1:.+]]: tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: %[[PADDING:.+]] = arith.constant
// CHECK{LITERAL}: dense<[[0, 2], [1, 1]]> : tensor<2x2xi32>
// CHECK: %[[INTERIOR:.+]] = arith.constant
// CHECK{LITERAL}: dense<[1, 2]> : tensor<2xi32>
// CHECK: %[[DILATE:.+]] = "tfl.dilate"(%[[ARG0]], %[[INTERIOR]], %[[ARG1]]) : (tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2xi32>, tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<3x7x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: %[[PAD:.+]] = "tfl.padv2"(%[[DILATE]], %[[PADDING]], %[[ARG1]]) : (tensor<3x7x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2x2xi32>, tensor<!quant.uniform<i8:f32, 2.000000e+00:-1>>) -> tensor<5x9x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[PAD]]

// -----

// Tests that a float `stablehlo.pad` is not converted to `tfl.padv2`.

func.func @pad_float(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<4x5xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [0, 0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}
// CHECK-LABEL: pad_float
// CHECK-NOT: tfl.padv2
// CHECK: stablehlo.pad

// -----

// Tests that a quantized `stablehlo.slice` is converted to
// `tfl.slice` when stride is 1.

func.func @slice(
    %arg0: tensor<3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 2>,
    limit_indices = array<i64: 3, 4>,
    strides = array<i64: 1, 1>
  } : (
    tensor<3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: slice
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK-DAG: %[[START:.+]] = arith.constant dense<{{\[1, 2\]}}> : tensor<2xi32>
// CHECK-DAG: %[[SIZE:.+]] = arith.constant dense<2> : tensor<2xi32>
// CHECK: %[[SLICE:.+]] = "tfl.slice"(%[[ARG0]], %[[START]], %[[SIZE]]) : (tensor<3x4x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[SLICE]]

// -----

// Tests that a quantized `stablehlo.slice` is converted to `tfl.strided_slice`
// when stride is not 1.

func.func @strided_slice(
    %arg0: tensor<3x6x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0, 2>,
    limit_indices = array<i64: 3, 6>,
    strides = array<i64: 2, 3>
  } : (
    tensor<3x6x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  ) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
}
// CHECK-LABEL: strided_slice
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x6x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: %[[START:.+]] = arith.constant
// CHECK{LITERAL}: dense<[0, 2]> : tensor<2xi32>
// CHECK: %[[SIZE:.+]] = arith.constant
// CHECK{LITERAL}: dense<[3, 4]> : tensor<2xi32>
// CHECK: %[[STRIDE:.+]] = arith.constant
// CHECK{LITERAL}: dense<[2, 3]> : tensor<2xi32>
// CHECK: %[[SLICE:.+]] = "tfl.strided_slice"(%[[ARG0]], %[[START]], %[[SIZE]], %[[STRIDE]]) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<3x6x!quant.uniform<i8:f32, 2.000000e+00:-1>>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2x!quant.uniform<i8:f32, 2.000000e+00:-1>>
// CHECK: return %[[SLICE]]

// -----

// Tests that a float `stablehlo.slice` is not converted to `tfl.slice`.

func.func @slice_float(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 1, 2>,
    limit_indices = array<i64: 3, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<3x4xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: slice_float
// CHECK-NOT: tfl.slice
// CHECK-NOT: tfl.strided_slice
// CHECK: stablehlo.slice

// -----

// Tests that a quantized `stablehlo.broadcast_in_dim` is converted to
// `tfl.broadcast_to`.

func.func @broadcast_in_dim(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1>
  } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
}
// CHECK-LABEL: broadcast_in_dim
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[SHAPE:.+]] = arith.constant
// CHECK{LITERAL}: dense<[3, 2]> : tensor<2xi32>
// CHECK: %[[BROADCAST:.+]] = "tfl.broadcast_to"(%[[ARG0]], %[[SHAPE]]) : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<2xi32>) -> tensor<3x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: return %[[BROADCAST]]

// -----

// Tests that a quantized `stablehlo.broadcast_in_dim` is converted to
// `tfl.transpose` and `tfl.broadcast_to` when `broadcast_dimensions` is not in
// ascending order.

func.func @broadcast_in_dim_with_transpose(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1, 0>
  } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:3>>
}
// CHECK-LABEL: broadcast_in_dim_with_transpose
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[BROADCAST_DIM:.+]] = arith.constant
// CHECK{LITERAL}: dense<[2, 3]> : tensor<2xi32>
// CHECK: %[[PERM:.+]] = arith.constant
// CHECK{LITERAL}: dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[TRANSPOSE:.+]] = "tfl.transpose"(%[[ARG0]], %[[PERM]]) : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<2xi32>) -> tensor<2x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[BROADCAST:.+]] = "tfl.broadcast_to"(%[[TRANSPOSE]], %[[BROADCAST_DIM]]) : (tensor<2x1x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<2xi32>) -> tensor<2x3x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: return %[[BROADCAST]]

// -----

// Tests that a quantized `stablehlo.broadcast_in_dim` is converted to
// tfl.expand_dims and `tfl.broadcast_to` when input rank is smaller than output
// rank.

func.func @broadcast_in_dim_with_expand(
    %arg0: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<3x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1>
  } : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<3x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<3x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
}
// CHECK-LABEL: broadcast_in_dim_with_expand
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK-DAG: %[[BROADCAST_DIM:.+]] = arith.constant dense<{{\[3, 2, 1, 1\]}}> : tensor<4xi32>
// CHECK-DAG: %[[EXPAND_DIM1:.+]] = arith.constant dense<3> : tensor<1xi32>
// CHECK-DAG: %[[EXPAND_DIM0:.+]] = arith.constant dense<2> : tensor<1xi32>
// CHECK: %[[EXPAND0:.+]] = "tfl.expand_dims"(%[[ARG0]], %[[EXPAND_DIM0]]) : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<1xi32>) -> tensor<1x2x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[EXPAND1:.+]] = "tfl.expand_dims"(%[[EXPAND0]], %[[EXPAND_DIM1]]) : (tensor<1x2x1x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<1xi32>) -> tensor<1x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[BROADCAST:.+]] = "tfl.broadcast_to"(%[[EXPAND1]], %[[BROADCAST_DIM]]) : (tensor<1x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<4xi32>) -> tensor<3x2x1x1x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: return %[[BROADCAST]]

// -----

// Tests that a quantized `stablehlo.broadcast_in_dim` is converted to
// `tfl.transpose`, `tfl.expand_dims` and `tfl.broadcast_to` when
// `broadcast_dimensions` is not in ascending order and input rank is smaller
// than output rank.

func.func @broadcast_in_dim_with_transpose_and_expand(
    %arg0: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
  ) -> tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1, 0, 4>
  } : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:3>>) -> tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
  return %0 : tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
}
// CHECK-LABEL: broadcast_in_dim_with_transpose_and_expand
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK-DAG: %[[BROADCAST_DIM:.+]] = arith.constant dense<{{\[3, 2, 1, 1, 4\]}}> : tensor<5xi32>
// CHECK-DAG: %[[EXPAND_DIM1:.+]] = arith.constant dense<3> : tensor<1xi32>
// CHECK-DAG: %[[EXPAND_DIM0:.+]] = arith.constant dense<2> : tensor<1xi32>
// CHECK-DAG: %[[PERM:.+]] = arith.constant dense<{{\[1, 0, 2\]}}> : tensor<3xi32>
// CHECK: %[[TRANSPOSE:.+]] = "tfl.transpose"(%[[ARG0]], %[[PERM]]) : (tensor<2x3x4x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<3xi32>) -> tensor<3x2x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[EXPAND0:.+]] = "tfl.expand_dims"(%[[TRANSPOSE]], %[[EXPAND_DIM0]]) : (tensor<3x2x4x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<1xi32>) -> tensor<3x2x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[EXPAND1:.+]] = "tfl.expand_dims"(%[[EXPAND0]], %[[EXPAND_DIM1]]) : (tensor<3x2x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<1xi32>) -> tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: %[[BROADCAST:.+]] = "tfl.broadcast_to"(%[[EXPAND1]], %[[BROADCAST_DIM]]) : (tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>, tensor<5xi32>) -> tensor<3x2x1x1x4x!quant.uniform<i8:f32, 2.000000e+00:3>>
// CHECK: return %[[BROADCAST]]

// -----

// Tests that a float `stablehlo.broadcast_in_dim` is not converted to
// `tfl.broadcast_to`.

func.func @broadcast_in_dim_float(%arg0: tensor<1x2xf32>) -> tensor<3x2xf32> {
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 0, 1>
  } : (tensor<1x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// CHECK-LABEL: broadcast_in_dim_float
// CHECK-NOT: tfl.broadcast_to
// CHECK-NOT: tfl.transpose
// CHECK-NOT: tfl.expand_dims
// CHECK: stablehlo.broadcast_in_dim

// -----

// Tests that a quantized `stablehlo.reduce_window` with max is converted to
// `tfl.max_pool_2d`.

func.func @reduce_window_with_max(
  %arg0: tensor<2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
  %arg1: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
) -> tensor<2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, %arg3: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
    stablehlo.return %1 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
  }) {window_dimensions = array<i64: 1, 3, 4, 1>, window_strides = array<i64: 1, 2, 3, 1>} : (tensor<2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> tensor<2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: reduce_window_with_max
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK-SAME: %[[ARG1:.*]]: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK: %[[MAX_POOL:.*]] = "tfl.max_pool_2d"(%[[ARG0]])
// CHECK-SAME: {filter_height = 3 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 3 : i32}
// CHECK-SAME: (tensor<2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> tensor<2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK: return %[[MAX_POOL]]

// -----

// Tests that a quantized `stablehlo.reduce_window `with max whose rank is not 4
// is not converted to `tfl.max_pool_2d`.

func.func @reduce_window_not_4d(
  %arg0: tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
  %arg1: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
) -> tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, %arg3: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
    stablehlo.return %1 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
  }) {window_dimensions = array<i64: 1, 1, 3, 4, 1>, window_strides = array<i64: 1, 1, 2, 3, 1>} : (tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: reduce_window_not_4d
// CHECK: stablehlo.reduce_window
// CHECK-NOT: tfl.max_pool_2d

// -----

// Tests that a quantized `stablehlo.reduce_window` with max that takes multiple
// inputs is not converted to `tfl.max_pool_2d`.

func.func @reduce_window_not_binary(
  %arg0: tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
  %arg1: tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
  %arg2: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>,
  %arg3: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
) -> tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0, %1 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, %arg5: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, %arg6: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, %arg7: tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>):
    %2 = stablehlo.maximum %arg4, %arg5 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
    %3 = stablehlo.maximum %arg6, %arg7 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
    stablehlo.return %2, %3 : tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>
  }) {window_dimensions = array<i64: 1, 1, 3, 4, 1>, window_strides = array<i64: 1, 1, 2, 3, 1>} : (tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<3x2x9x10x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> (tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>)
  return %0 : tensor<3x2x4x3x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: reduce_window_not_binary
// CHECK: stablehlo.reduce_window
// CHECK-NOT: tfl.max_pool_2d

// -----

// Tests that a float `stablehlo.reduce_window` with max is not converted to
// `tfl.max_pool_2d`.

func.func @reduce_window_with_max_float(
  %arg0: tensor<2x9x10x3xf32>,
  %arg1: tensor<f32>
) -> tensor<2x4x3x3xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    stablehlo.return %1 : tensor<f32>
  }) {window_dimensions = array<i64: 1, 3, 4, 1>, window_strides = array<i64: 1, 2, 3, 1>} : (tensor<2x9x10x3xf32>, tensor<f32>) -> tensor<2x4x3x3xf32>
  return %0 : tensor<2x4x3x3xf32>
}

// CHECK-LABEL: reduce_window_with_max_float
// CHECK: stablehlo.reduce_window
// CHECK-NOT: tfl.max_pool_2d

// -----

// Tests that a quantized `stablehlo.dynamic_reshape` is converted to
// `tfl.reshape`.

func.func @dynamic_reshape(
    %arg0: tensor<?x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2xi32>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (
    tensor<?x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<2xi32>
  ) -> tensor<?x?x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<?x?x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @dynamic_reshape
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK-SAME: %[[ARG1:.+]]: tensor<2xi32>
// CHECK: %[[RESHAPE:.+]] = "tfl.reshape"(%[[ARG0]], %[[ARG1]]) : (tensor<?x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<2xi32>) -> tensor<?x?x!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK: return %[[RESHAPE]]
// CHECK-NOT: stablehlo.dynamic_reshape

// -----

// Tests that a float `stablehlo.dynamic_reshape` is not converted to
// `tfl.reshape`.

func.func @dynamic_reshape_float(%arg0: tensor<?x3xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_reshape_float
// CHECK: stablehlo.dynamic_reshape
// CHECK-NOT: tfl.reshape

// -----

// Tests that a quantized `stablehlo.gather` is converted to tfl.gather_nd.

func.func @gather(
    %arg0: tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @gather
// CHECK-SAME: %[[ARG_0:.+]]: tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>, %[[ARG_1:.+]]: tensor<2x3x2xi64>
// CHECK: %[[GATHER:.+]] = "tfl.gather_nd"(%[[ARG_0]], %[[ARG_1]])
// CHECK-SAME: (tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<2x3x2xi64>) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
// CHECK: return %[[GATHER]]

// -----

// Tests that a quantized `stablehlo.gather` with unsorted start_index_map is
// not converted to `tfl.gather_nd` (condition 1 is not satisfied).

func.func @gather_start_index_map_not_sorted(
    %arg0: tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [1, 0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @gather_start_index_map_not_sorted
// CHECK: stablehlo.gather
// CHECK-NOT: tfl.gather_nd
// CHECK-NOT: tfl.gather

// -----

// Tests that a quantized `stablehlo.gather` is not converted to tfl.gather_nd
// when index_vector_dim is not the last dimension of start_indices (condition 2
// is not satisfied).

func.func @gather_start_index_vector_dim_not_at_last(
    %arg0: tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2x3x2xi64>
  ) -> tensor<3x2x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 0>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    tensor<2x3x2xi64>
  ) -> tensor<3x2x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<3x2x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @gather_start_index_vector_dim_not_at_last
// CHECK: stablehlo.gather
// CHECK-NOT: tfl.gather_nd
// CHECK-NOT: tfl.gather

// -----

// Tests that a quantized `stablehlo.gather` is not converted to tfl.gather_nd
// when offset_dims are not the last dimensions of the output (condition 3 is
// not satisfied).

func.func @gather_offset_dims_not_at_last(
    %arg0: tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2x3x2xi64>
  ) -> tensor<2x2x2x3x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [0, 1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x2x2x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x2x2x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @gather_offset_dims_not_at_last
// CHECK: stablehlo.gather
// CHECK-NOT: tfl.gather_nd
// CHECK-NOT: tfl.gather

// -----

// Tests that a quantized `stablehlo.gather` is not converted to tfl.gather_nd
// when shape of slice is not same with shape of offset (condition 4 is not
// satisfied).

func.func @gather_different_slice_and_offset(
    %arg0: tensor<3x4x2x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (
    tensor<3x4x2x3x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    tensor<2x3x2xi64>
  ) -> tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x3x2x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}

// CHECK-LABEL: func @gather_different_slice_and_offset
// CHECK: stablehlo.gather
// CHECK-NOT: tfl.gather_nd
// CHECK-NOT: tfl.gather

// -----

// Tests that a float `stablehlo.gather` is not converted to `tfl.gather_nd`.

func.func @gather_float(%arg0: tensor<3x4x2x2xf32>, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2, 3],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<3x4x2x2xf32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xf32>
  return %0 : tensor<2x3x2x2xf32>
}

// CHECK-LABEL: func @gather_float
// CHECK: stablehlo.gather
// CHECK-NOT: tfl.gather_nd
// CHECK-NOT: tfl.gather

// -----

// Tests that a quantized `stablehlo.dynamic_slice` is converted to `tfl.slice`.

// CHECK-LABEL: func @dynamic_slice
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>, %[[ARG1:.+]]: tensor<i64>, %[[ARG2:.+]]: tensor<i64>
func.func @dynamic_slice(
    %arg0: tensor<4x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>,
    %arg1: tensor<i64>,
    %arg2: tensor<i64>
  ) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-01:-5>> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = array<i64: 2, 1>
  } : (
    tensor<4x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<i64>,
    tensor<i64>
  ) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  return %0 : tensor<2x1x!quant.uniform<i8:f32, 3.000000e-01:-5>>
}


// CHECK-DAG: %[[SLICE_SIZE:.+]] = arith.constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant dense<0> : tensor<1xi64>
// CHECK-DAG: %[[MAX1:.+]] = arith.constant dense<2> : tensor<1xi64>
// CHECK-DAG: %[[MAX2:.+]] = arith.constant dense<3> : tensor<1xi64>
// CHECK: %[[BITCAST1:.+]] = "tfl.bitcast"(%[[ARG1]]) : (tensor<i64>) -> tensor<1xi64>
// CHECK: %[[MIN1:.+]] = "tfl.minimum"(%[[BITCAST1]], %[[MAX1]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK: %[[BITCAST2:.+]] = "tfl.bitcast"(%[[ARG2]]) : (tensor<i64>) -> tensor<1xi64>
// CHECK: %[[MIN2:.+]] = "tfl.minimum"(%[[BITCAST2]], %[[MAX2]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK: %[[CONCAT:.+]] = "tfl.concatenation"(%[[MIN1]], %[[MIN2]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK: %[[MAX:.+]] = "tfl.maximum"(%[[CONCAT]], %[[ZERO]]) : (tensor<2xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK: %[[SLICE:.+]] = "tfl.slice"(%[[ARG0]], %[[MAX]], %[[SLICE_SIZE]])
// CHECK-SAME: (tensor<4x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-01:-5>>

// -----

// Tests that a float `stablehlo.dynamic_slice` is not converted to `tfl.slice`.

func.func @dynamic_slice_float(%arg0: tensor<4x4xf32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<2x1xf32> {
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1, %arg2) {
    slice_sizes = array<i64: 2, 1>
  } : (tensor<4x4xf32>, tensor<i64>, tensor<i64>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func @dynamic_slice_float
// CHECK: stablehlo.dynamic_slice
// CHECK-NOT: tfl.bitcast
// CHECK-NOT: tfl.minimum
// CHECK-NOT: tfl.maximum
// CHECK-NOT: tfl.slice

// -----

// Tests that `stablehlo.add` with both operands int8 UniformQuantizedType is
// properly converted into `tfl.add`.

func.func @add(%arg0: tensor<1x3x!quant.uniform<i8:f32, 1.000000e+0:8>>, %arg1: tensor<1x3x!quant.uniform<i8:f32, 2.000000e+0:8>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+0:8>> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e+0:8>>, tensor<1x3x!quant.uniform<i8:f32, 2.000000e+0:8>>) -> tensor<1x3x!quant.uniform<i8:f32, 3.000000e+0:8>>
  return %0 : tensor<1x3x!quant.uniform<i8:f32, 3.000000e+0:8>>
}

// CHECK-LABEL: func @add
// CHECK: %[[ADD:.+]] = tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>, tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: return %[[ADD]]

// -----

// Tests that `stablehlo.add` with int32 UniformQuantizedPerAxisTypes is
// not converted.

func.func @add_i32(%arg0: tensor<1x3x!quant.uniform<i32:f32:1, {1.000000e+0, 1.000000e+0, 1.000000e+0}>>, %arg1: tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<1x3x!quant.uniform<i32:f32:1, {1.000000e+0, 1.000000e+0, 1.000000e+0}>>, tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
  return %0 : tensor<1x3x!quant.uniform<i32:f32:1, {2.000000e+0, 2.000000e+0, 2.000000e+0}>>
}

// CHECK-LABEL: func @add_i32
// CHECK: stablehlo.add
// CHECK-NOT: tfl.add

// -----

// Tests that a quantized `stablehlo.constant` is converted into `tfl.qconst`.

// CHECK-LABEL: func @quantized_constant
func.func @quantized_constant() -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  return %0 : tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
}

// CHECK: %[[QCONST:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>}>
// CHECK-SAME: () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>
// CHECK: return %[[QCONST]]

// -----

// Tests that a float `stablehlo.constant` is not converted into `tfl.qconst`.

// CHECK-LABEL: func @float_constant
func.func @float_constant() -> tensor<1x2x4x5xf32> {
  %0 = stablehlo.constant() {value = dense<1.0> : tensor<1x2x4x5xf32>} : () -> tensor<1x2x4x5xf32>
  return %0 : tensor<1x2x4x5xf32>
}

// CHECK: stablehlo.constant
// CHECK-NOT: tfl.pseudo_qconst
// CHECK-NOT: tfl.pseudo_const
// CHECK-NOT: arith.constant

// -----

// Tests that a hybrid quantized dot_general is splitted into dequantize and float
// dot_general.

// CHECK-LABEL: func @dot_general_hybrid
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2x3x4xf32>
func.func @dot_general_hybrid(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x5xf32> {
  %0 = stablehlo.constant() {value = dense<1> : tensor<1x2x4x5xi8>} : () -> tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>
  %1 = "stablehlo.dot_general"(%arg0, %0) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3x4xf32>, tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+0>>) -> tensor<1x2x3x5xf32>
  return %1 : tensor<1x2x3x5xf32>
}

// CHECK: %[[WEIGHT:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>, value = dense<1> : tensor<1x2x4x5xi8>}>
// CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[WEIGHT]]) : (tensor<1x2x4x5x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x4x5xf32>
// CHECK: %[[DOT:.+]] = stablehlo.dot_general %[[ARG0]], %[[DQ]], batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1x2x3x4xf32>, tensor<1x2x4x5xf32>) -> tensor<1x2x3x5xf32>
// CHECK: return %[[DOT]]

// -----

// Tests that a hybrid per-channel quantized convolution for tfl.conv_2d is
// splitted into dequantize and float stablehlo.convolution.

// CHECK-LABEL: func @convolution_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x3x4xf32>
func.func @convolution_hybrid_per_channel(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x2xf32> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x2x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x2xf32>
  return %1 : tensor<1x3x3x2xf32>
}

// CHECK: %[[WEIGHT:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<2x3x3x4xi8>}>
// CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[WEIGHT]]) : (tensor<2x3x3x4x!quant.uniform<i8<-127:127>:f32:0, {2.000000e+02,3.000000e+03}>>) -> tensor<2x3x3x4xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG0]], %[[DQ]])
// CHECK{LITERAL}: dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
// CHECK-SAME: (tensor<1x3x3x4xf32>, tensor<2x3x3x4xf32>) -> tensor<1x3x3x2xf32>
// CHECK: return %[[CONV]]

// -----

// Tests that a hybrid per-tensor quantized convolution for tfl.conv_2d is
// splitted into dequantize and float stablehlo.convolution.

// CHECK-LABEL: func @convolution_hybrid_per_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x3x4xf32>
func.func @convolution_hybrid_per_tensor(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x2xf32> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x4x2xi8>} : () -> tensor<3x3x4x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x2x!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> tensor<1x3x3x2xf32>
  return %1 : tensor<1x3x3x2xf32>
}

// CHECK: %[[WEIGHT:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x3x3x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>, value = dense<3> : tensor<2x3x3x4xi8>}>
// CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[WEIGHT]]) : (tensor<2x3x3x4x!quant.uniform<i8:f32, 3.000000e-01:-5>>) -> tensor<2x3x3x4xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG0]], %[[DQ]])
// CHECK{LITERAL}: dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
// CHECK-SAME: (tensor<1x3x3x4xf32>, tensor<2x3x3x4xf32>) -> tensor<1x3x3x2xf32>
// CHECK: return %[[CONV]]

// -----

// Tests that a hybrid per-channel quantized convolution for tfl.depthwise_conv
// is splitted into dequantize and float stablehlo.convolution.

// CHECK-LABEL: func @depthwise_convolution_hybrid_per_channel
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x3x3x4xf32>
func.func @depthwise_convolution_hybrid_per_channel(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant() {value = dense<3> : tensor<3x3x1x4xi8>} : () -> tensor<3x3x1x4x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3, 2.000000e+2, 3.000000e+3}>>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x1x4x!quant.uniform<i8:f32:3, {2.000000e+2, 3.000000e+3, 2.000000e+2, 3.000000e+3}>>) -> tensor<1x3x3x4xf32>
  return %1 : tensor<1x3x3x4xf32>
}

// CHECK: %[[WEIGHT:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<1x3x3x4x!quant.uniform<i8<-127:127>:f32:3, {2.000000e+02,3.000000e+03,2.000000e+02,3.000000e+03}>>, value = dense<3> : tensor<1x3x3x4xi8>}>
// CHECK: %[[DQ:.+]] = "tfl.dequantize"(%[[WEIGHT]]) : (tensor<1x3x3x4x!quant.uniform<i8<-127:127>:f32:3, {2.000000e+02,3.000000e+03,2.000000e+02,3.000000e+03}>>) -> tensor<1x3x3x4xf32>
// CHECK: %[[CONV:.+]] = stablehlo.convolution(%[[ARG0]], %[[DQ]])
// CHECK{LITERAL}: dim_numbers = [b, 0, 1, f]x[i, 0, 1, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 4 : i64}
// CHECK-SAME: (tensor<1x3x3x4xf32>, tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
// CHECK: return %[[CONV]]
