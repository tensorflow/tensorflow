// RUN: odml-to-stablehlo-opt --compose-uniform-quantized-type \
// RUN:     --split-input-file --verify-diagnostics %s | FileCheck %s

module {
// CHECK-LABEL: quantized_conv_op
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4xf32>
  func.func @quantized_conv_op(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1x1x1x1xf32>  // Input inverse scale.
    %2 = stablehlo.constant dense<-128> : tensor<1x1x1x1xi8>  // Input zero point.
    %3 = stablehlo.constant dense<1> : tensor<3x3x4x4xi8>  // Quantized filter tensor.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x1x1x4xf32>
    %5 = stablehlo.constant dense<4.000000e+03> : tensor<1x1x1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<127> : tensor<1x1x1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %1, %2) : (tensor<1x3x3x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8>
    %8 = stablehlo.convert %7 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    %9 = stablehlo.convert %3 : (tensor<3x3x4x4xi8>) -> tensor<3x3x4x4xf32>
    %10 = stablehlo.convolution(%8, %9) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %11 = stablehlo.reshape %2 : (tensor<1x1x1x1xi8>) -> tensor<1xi8>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi8>) -> tensor<1x3x3x4xi8>
    %13 = stablehlo.convert %12 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    %14 = stablehlo.convert %3 : (tensor<3x3x4x4xi8>) -> tensor<3x3x4x4xf32>
    %15 = stablehlo.convolution(%13, %14) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %16 = stablehlo.subtract %10, %15 : tensor<1x3x3x4xf32>
    %17 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x4xf32>) -> tensor<1x3x3x4xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x3x3x4xf32>
    %19 = call @uniform_quantize_1(%18, %5, %6) : (tensor<1x3x3x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8>
    %20 = call @uniform_dequantize_0(%19, %5, %6) : (tensor<1x3x3x4xi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32>
    return %20 : tensor<1x3x3x4xf32>
  }
// CHECK: %[[FILTER:.*]] = stablehlo.constant() <{value = dense<1> : tensor<3x3x4x4xi8>}> : () -> tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{{.*}}}>>
// CHECK: %[[QUANT_ARG:.*]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[QUANT_ARG]], %[[FILTER]]) {{.*}} : (tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>, tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{.*}}>>) -> tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]] : (tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3x3x4xf32>
// CHECK: return %[[DEQUANT]] : tensor<1x3x3x4xf32>

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xi8>
    return %0 : tensor<1x3x3x4xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xi8>
    return %0 : tensor<1x3x3x4xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3x3x4xi8>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    return %0 : tensor<1x3x3x4xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests a variant where there is no stablehlo.convert op in between the
// filter constant and the convolution op.
//
// `filter (f32) -> convolution`
//
// instead of:
//
// `filter (i8) -> convert (i8 -> f32) -> convolution`

module {
// CHECK-LABEL: quantized_conv_op_with_no_filter_convert
// CHECK-SAME: %[[ARG:.*]]: tensor<1x3x3x4xf32>
  func.func @quantized_conv_op_with_no_filter_convert(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1x1x1x1xf32>  // Input inverse scale.
    %2 = stablehlo.constant dense<-128> : tensor<1x1x1x1xi8>  // Input zero point.
    %3 = stablehlo.constant dense<2.000000e+01> : tensor<3x3x4x4xf32>  // Quantized filter tensor.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x1x1x4xf32>
    %5 = stablehlo.constant dense<4.000000e+03> : tensor<1x1x1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<127> : tensor<1x1x1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %1, %2) : (tensor<1x3x3x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8>
    %8 = stablehlo.convert %7 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    %9 = stablehlo.convolution(%8, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %10 = stablehlo.reshape %2 : (tensor<1x1x1x1xi8>) -> tensor<1xi8>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<1xi8>) -> tensor<1x3x3x4xi8>
    %12 = stablehlo.convert %11 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    %13 = stablehlo.convolution(%12, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %14 = stablehlo.subtract %9, %13 : tensor<1x3x3x4xf32>
    %15 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x4xf32>) -> tensor<1x3x3x4xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<1x3x3x4xf32>
    %17 = call @uniform_quantize_1(%16, %5, %6) : (tensor<1x3x3x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8>
    %18 = call @uniform_dequantize_0(%17, %5, %6) : (tensor<1x3x3x4xi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32>
    return %18 : tensor<1x3x3x4xf32>
  }
// CHECK: %[[FILTER:.*]] = stablehlo.constant() <{value = dense<20> : tensor<3x3x4x4xi8>}> : () -> tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{{.*}}}>>
// CHECK: %[[QUANT_ARG:.*]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%[[QUANT_ARG]], %[[FILTER]]) {{.*}} : (tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>, tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{.*}}>>) -> tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]] : (tensor<1x3x3x4x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3x3x4xf32>
// CHECK: return %[[DEQUANT]] : tensor<1x3x3x4xf32>

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xi8>
    return %0 : tensor<1x3x3x4xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xi8>
    return %0 : tensor<1x3x3x4xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3x3x4xi8>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    return %0 : tensor<1x3x3x4xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// The pattern should not match when there are no `uniform_quantize` call
// for the input.

module {
// CHECK-LABEL: conv_no_input_uniform_quantize_call
  func.func @conv_no_input_uniform_quantize_call(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %1 = stablehlo.constant dense<1.000000e+03> : tensor<1x1x1x1xf32>  // Input inverse scale.
    %2 = stablehlo.constant dense<-128> : tensor<1x1x1x1xi8>  // Input zero point.
    %3 = stablehlo.constant dense<2.000000e+01> : tensor<3x3x4x4xf32>  // Quantized filter tensor.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x1x1x4xf32>
    %5 = stablehlo.constant dense<4.000000e+03> : tensor<1x1x1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<127> : tensor<1x1x1x1xi8>  // Output zero point.
    %10 = stablehlo.convolution(%arg0, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %11 = stablehlo.reshape %2 : (tensor<1x1x1x1xi8>) -> tensor<1xi8>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<1xi8>) -> tensor<1x3x3x4xi8>
    %13 = stablehlo.convert %12 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    %15 = stablehlo.convolution(%13, %3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %16 = stablehlo.subtract %10, %15 : tensor<1x3x3x4xf32>
    %17 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x4xf32>) -> tensor<1x3x3x4xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<1x3x3x4xf32>
    %19 = call @uniform_quantize_1(%18, %5, %6) : (tensor<1x3x3x4xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8>
    %20 = call @uniform_dequantize_0(%19, %5, %6) : (tensor<1x3x3x4xi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32>
    return %20 : tensor<1x3x3x4xf32>
  }
// CHECK-NOT: stablehlo.uniform_quantize
// CHECK-NOT: stablehlo.uniform_dequantize

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_1(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xi8>
    return %0 : tensor<1x3x3x4xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3x3x4xi8>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x1x1x1xi8>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3x3x4xi8>) -> tensor<1x3x3x4xf32>
    return %0 : tensor<1x3x3x4xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

module {
// CHECK-LABEL: quantized_dot_general
// CHECK-SAME: %[[ARG:.*]]: tensor<1x4x2xf32>
  func.func @quantized_dot_general(%arg0: tensor<1x4x2xf32>) -> tensor<1x4x3xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<1x1x1xf32>  // Input inverse scale.
    %1 = stablehlo.constant dense<1> : tensor<1x1x1xi8>  // Input zero point.
    %2 = stablehlo.constant dense<5> : tensor<2x3xi8>  // Quantized filter.
    %3 = stablehlo.constant dense<4> : tensor<1x1x3xi32>  // Precalculated q2 * z1.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x1x3xf32>  // Merged scale: s1 * s2.
    %5 = stablehlo.constant dense<2.000000e+02> : tensor<1x1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<2> : tensor<1x1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %0, %1) : (tensor<1x4x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x2xi8>
    %8 = stablehlo.convert %7 : (tensor<1x4x2xi8>) -> tensor<1x4x2xf32>
    %9 = stablehlo.convert %2 : (tensor<2x3xi8>) -> tensor<2x3xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [2] x [0] : (tensor<1x4x2xf32>, tensor<2x3xf32>) -> tensor<1x4x3xf32>
    %11 = stablehlo.convert %3 : (tensor<1x1x3xi32>) -> tensor<1x1x3xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x1x3xf32>) -> tensor<1x4x3xf32>  // Optional
    %13 = stablehlo.subtract %10, %12 : tensor<1x4x3xf32>  // Precalculated zp_neg.
    %14 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<1x1x3xf32>) -> tensor<1x4x3xf32>  // Optional
    %15 = stablehlo.multiply %13, %14 : tensor<1x4x3xf32>  // s1 * s2
    %16 = call @uniform_quantize_1(%15, %5, %6) : (tensor<1x4x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x3xi8>
    %17 = call @uniform_dequantize_0(%16, %5, %6) : (tensor<1x4x3xi8>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x3xf32>
    return %17 : tensor<1x4x3xf32>
  }
// Quantization dimension == 1 because it is the output feature dimension.
// CHECK: %[[FILTER:.*]] = stablehlo.constant() <{value = dense<5> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8:f32:1, {{{.*}}}>>
// CHECK: %[[QUANT_ARG:.*]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x4x2xf32>) -> tensor<1x4x2x!quant.uniform<i8:f32, {{.*}}:1>>
// CHECK: %[[CONV:.*]] = stablehlo.dot_general %[[QUANT_ARG]], %[[FILTER]], contracting_dims = [2] x [0] : (tensor<1x4x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8:f32:1, {{.*}}>>) -> tensor<1x4x3x!quant.uniform<i8:f32, {{.*}}:2>>
// CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]] : (tensor<1x4x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x4x3xf32>
// CHECK: return %[[DEQUANT]] : tensor<1x4x3xf32>

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x4x2xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x2xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x2xf32>) -> tensor<1x4x2xi8>
    return %0 : tensor<1x4x2xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x4x3xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x3xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x3xf32>) -> tensor<1x4x3xi8>
    return %0 : tensor<1x4x3xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x4x3xi8>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x3xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x3xi8>) -> tensor<1x4x3xf32>
    return %0 : tensor<1x4x3xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests that when dot_general's filter comes from an f32 constant
// it is cast to i8 after the conversion.

module {
// CHECK-LABEL: quantized_dot_general_float_filter
// CHECK-SAME: %[[ARG:.*]]: tensor<1x4x2xf32>
  func.func @quantized_dot_general_float_filter(%arg0: tensor<1x4x2xf32>) -> tensor<1x4x3xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<1x1x1xf32>  // Input inverse scale.
    %1 = stablehlo.constant dense<1> : tensor<1x1x1xi8>  // Input zero point.
    // Filter, disguised as f32 but the values are actually i8.
    %2 = stablehlo.constant dense<5.000000e+00> : tensor<2x3xf32>
    %3 = stablehlo.constant dense<4> : tensor<1x1x3xi32>  // Precalculated q2 * z1.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x1x3xf32>  // Merged scale: s1 * s2.
    %5 = stablehlo.constant dense<2.000000e+02> : tensor<1x1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<2> : tensor<1x1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %0, %1) : (tensor<1x4x2xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x2xi8>
    %8 = stablehlo.convert %7 : (tensor<1x4x2xi8>) -> tensor<1x4x2xf32>
    %9 = stablehlo.dot_general %8, %2, contracting_dims = [2] x [0] : (tensor<1x4x2xf32>, tensor<2x3xf32>) -> tensor<1x4x3xf32>
    %10 = stablehlo.convert %3 : (tensor<1x1x3xi32>) -> tensor<1x1x3xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x1x3xf32>) -> tensor<1x4x3xf32>  // Optional
    %12 = stablehlo.subtract %9, %11 : tensor<1x4x3xf32>  // Precalculated zp_neg.
    %13 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2] : (tensor<1x1x3xf32>) -> tensor<1x4x3xf32>  // Optional
    %14 = stablehlo.multiply %12, %13 : tensor<1x4x3xf32>  // s1 * s2
    %15 = call @uniform_quantize_1(%14, %5, %6) : (tensor<1x4x3xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x3xi8>
    %16 = call @uniform_dequantize_0(%15, %5, %6) : (tensor<1x4x3xi8>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<1x4x3xf32>
    return %16 : tensor<1x4x3xf32>
  }
// Quantization dimension == 1 because it is the output feature dimension.
// Quantized filter values (from f32 constant) are cast to i8.
// CHECK: %[[FILTER:.*]] = stablehlo.constant() <{value = dense<5> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8:f32:1, {{{.*}}}>>
// CHECK: %[[QUANT_ARG:.*]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x4x2xf32>) -> tensor<1x4x2x!quant.uniform<i8:f32, {{.*}}:1>>
// CHECK: %[[CONV:.*]] = stablehlo.dot_general %[[QUANT_ARG]], %[[FILTER]], contracting_dims = [2] x [0] : (tensor<1x4x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8:f32:1, {{.*}}>>) -> tensor<1x4x3x!quant.uniform<i8:f32, {{.*}}:2>>
// CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]] : (tensor<1x4x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x4x3xf32>
// CHECK: return %[[DEQUANT]] : tensor<1x4x3xf32>

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x4x2xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x2xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x2xf32>) -> tensor<1x4x2xi8>
    return %0 : tensor<1x4x2xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x4x3xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x3xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x3xf32>) -> tensor<1x4x3xi8>
    return %0 : tensor<1x4x3xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x4x3xi8>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<1x4x3xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x4x3xi8>) -> tensor<1x4x3xf32>
    return %0 : tensor<1x4x3xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests that the conversion is successful even when there are no
// broadcast_in_dim ops for the second arguments of the subtract op and
// multiply op.

module {
// CHECK-LABEL: quantized_dot_general_no_broadcast
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2xf32>
  func.func @quantized_dot_general_no_broadcast(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<1x1xf32>  // Input inverse scale.
    %1 = stablehlo.constant dense<1> : tensor<1x1xi8>  // Input zero point.
    %2 = stablehlo.constant dense<5> : tensor<2x3xi8>  // Quantized filter.
    %3 = stablehlo.constant dense<4> : tensor<1x3xi32>  // Precalculated z1 * q2.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x3xf32>  // Merged scale: s1 * s2.
    %5 = stablehlo.constant dense<2.000000e+02> : tensor<1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<2> : tensor<1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %0, %1) : (tensor<1x2xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x2xi8>
    %8 = stablehlo.convert %7 : (tensor<1x2xi8>) -> tensor<1x2xf32>
    %9 = stablehlo.convert %2 : (tensor<2x3xi8>) -> tensor<2x3xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %11 = stablehlo.convert %3 : (tensor<1x3xi32>) -> tensor<1x3xf32>
    %12 = stablehlo.subtract %10, %11 : tensor<1x3xf32>  // q1 * q2 - z1 * q2
    %13 = stablehlo.multiply %12, %4 : tensor<1x3xf32>  // s1 * s2
    %14 = call @uniform_quantize_1(%13, %5, %6) : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xi8>
    %15 = call @uniform_dequantize_0(%14, %5, %6) : (tensor<1x3xi8>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xf32>
    return %15 : tensor<1x3xf32>
  }
// Quantization dimension == 1 because it is the output feature dimension.
// CHECK: %[[FILTER:.*]] = stablehlo.constant() <{value = dense<5> : tensor<2x3xi8>}> : () -> tensor<2x3x!quant.uniform<i8:f32:1, {{{.*}}}>>
// CHECK: %[[QUANT_ARG:.*]] = stablehlo.uniform_quantize %[[ARG]] : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, {{.*}}:1>>
// CHECK: %[[CONV:.*]] = stablehlo.dot_general %[[QUANT_ARG]], %[[FILTER]], contracting_dims = [1] x [0] : (tensor<1x2x!quant.uniform<i8:f32, {{.*}}>>, tensor<2x3x!quant.uniform<i8:f32:1, {{.*}}>>) -> tensor<1x3x!quant.uniform<i8:f32, {{.*}}:2>>
// CHECK: %[[DEQUANT:.*]] = stablehlo.uniform_dequantize %[[CONV]] : (tensor<1x3x!quant.uniform<i8:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK: return %[[DEQUANT]] : tensor<1x3xf32>

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x2xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x2xf32>) -> tensor<1x2xi8>
    return %0 : tensor<1x2xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xf32>) -> tensor<1x3xi8>
    return %0 : tensor<1x3xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3xi8>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xi8>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests that the conversion doesn't happen when the uniform_quantize function
// outputs a i32 storage type.

module {
// CHECK-LABEL: quantized_dot_general_uniform_quantize_to_i32
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2xf32>
  func.func @quantized_dot_general_uniform_quantize_to_i32(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<1x1xf32>  // Input inverse scale.
    %1 = stablehlo.constant dense<1> : tensor<1x1xi8>  // Input zero point.
    %2 = stablehlo.constant dense<5> : tensor<2x3xi8>  // Quantized filter.
    %3 = stablehlo.constant dense<4> : tensor<1x3xi32>  // Precalculated z1 * q2.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x3xf32>  // Merged scale: s1 * s2.
    %5 = stablehlo.constant dense<2.000000e+02> : tensor<1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<2> : tensor<1x1xi8>  // Output zero point.
    // This uniform_quantize function is expected to output i8 instead of i32.
    %7 = call @uniform_quantize_0(%arg0, %0, %1) : (tensor<1x2xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x2xi32>
    %8 = stablehlo.convert %7 : (tensor<1x2xi32>) -> tensor<1x2xf32>
    %9 = stablehlo.convert %2 : (tensor<2x3xi8>) -> tensor<2x3xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %11 = stablehlo.convert %3 : (tensor<1x3xi32>) -> tensor<1x3xf32>
    %12 = stablehlo.subtract %10, %11 : tensor<1x3xf32>  // q1 * q2 - z1 * q2
    %13 = stablehlo.multiply %12, %4 : tensor<1x3xf32>  // s1 * s2
    %14 = call @uniform_quantize_1(%13, %5, %6) : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xi8>
    %15 = call @uniform_dequantize_0(%14, %5, %6) : (tensor<1x3xi8>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xf32>
    return %15 : tensor<1x3xf32>
  }
// CHECK-NOT: stablehlo.uniform_quantize
// CHECK-NOT: !quant.uniform
// CHECK: stablehlo.dot_general

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x2xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x2xf32>) -> tensor<1x2xi32>
    return %0 : tensor<1x2xi32>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xf32>) -> tensor<1x3xi8>
    return %0 : tensor<1x3xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3xi8>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xi8>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests that the conversion doesn't happen when the filter tensor is i32.

module {
// CHECK-LABEL: quantized_dot_general_filter_i32
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2xf32>
  func.func @quantized_dot_general_filter_i32(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant dense<3.000000e+00> : tensor<1x1xf32>  // Input inverse scale.
    %1 = stablehlo.constant dense<1> : tensor<1x1xi8>  // Input zero point.
    %2 = stablehlo.constant dense<5> : tensor<2x3xi32>  // Quantized filter - the pattern expects i8 but i32 is given.
    %3 = stablehlo.constant dense<4> : tensor<1x3xi32>  // Precalculated z1 * q2.
    %4 = stablehlo.constant dense<3.000000e+03> : tensor<1x3xf32>  // Merged scale: s1 * s2.
    %5 = stablehlo.constant dense<2.000000e+02> : tensor<1x1xf32>  // Output inverse scale.
    %6 = stablehlo.constant dense<2> : tensor<1x1xi8>  // Output zero point.
    %7 = call @uniform_quantize_0(%arg0, %0, %1) : (tensor<1x2xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x2xi8>
    %8 = stablehlo.convert %7 : (tensor<1x2xi8>) -> tensor<1x2xf32>
    %9 = stablehlo.convert %2 : (tensor<2x3xi32>) -> tensor<2x3xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
    %11 = stablehlo.convert %3 : (tensor<1x3xi32>) -> tensor<1x3xf32>
    %12 = stablehlo.subtract %10, %11 : tensor<1x3xf32>  // q1 * q2 - z1 * q2
    %13 = stablehlo.multiply %12, %4 : tensor<1x3xf32>  // s1 * s2
    %14 = call @uniform_quantize_1(%13, %5, %6) : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xi8>
    %15 = call @uniform_dequantize_0(%14, %5, %6) : (tensor<1x3xi8>, tensor<1x1xf32>, tensor<1x1xi8>) -> tensor<1x3xf32>
    return %15 : tensor<1x3xf32>
  }
// CHECK-NOT: stablehlo.uniform_quantize
// CHECK-NOT: !quant.uniform
// CHECK: stablehlo.dot_general

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize_0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x2xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x2xf32>) -> tensor<1x2xi8>
    return %0 : tensor<1x2xi8>
  }
// CHECK: @uniform_quantize_0
  func.func private @uniform_quantize_1(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xf32>) -> tensor<1x3xi8>
    return %0 : tensor<1x3xi8>
  }
// CHECK: @uniform_quantize_1
  func.func private @uniform_dequantize_0(%arg0: tensor<1x3xi8>, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xi8>) -> tensor<1x3xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x3xi8>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
// CHECK: @uniform_dequantize_0
}

// -----

// Tests that a quantized dot_general op is composed when both operands are
// actiavations.

// CHECK-LABEL: dot_general_with_two_activations
// CHECK-SAME: %[[ARG_0:.*]]: tensor<8x16x16xf32>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<8x16x4xf32>
module {
  func.func @dot_general_with_two_activations(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
    %1 = stablehlo.constant dense<2.000000e-01> : tensor<1x1x1xf32>  // Input 1 inverse scale (1 / s1).
    %2 = stablehlo.constant dense<-128> : tensor<1x1x1xi8>  // Input 1 zero point (z1).
    %3 = stablehlo.constant dense<-128> : tensor<1x1x1xi32>  // Input 1 zero point (z1) (upcast & folded into i32).
    %4 = stablehlo.constant dense<4.000000e-01> : tensor<1x1x1xf32>  // Input 2 inverse scale (1 / s2).
    %5 = stablehlo.constant dense<0> : tensor<1x1x1xi8>  // Input 2 zero point (z2).
    %6 = stablehlo.constant dense<0> : tensor<1x1x1xi32>  // Input 2 zero point (z2) (upcast & folded into i32).
    %7 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x1xf32>  // Output inverse scale (1 / s3).
    %8 = stablehlo.constant dense<-5> : tensor<1x1x1xi8>  // Output zero point (z3).
    %9 = stablehlo.constant dense<1.250000e+01> : tensor<1x1x1xf32>  // Merged scale (s1 * s2).
    %10 = call @uniform_quantize(%arg0, %1, %2) : (tensor<8x16x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x16xi8>  // q1
    %11 = call @uniform_quantize_0(%arg1, %4, %5) : (tensor<8x16x4xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xi8>  // q2
    %12 = stablehlo.convert %10 : (tensor<8x16x16xi8>) -> tensor<8x16x16xi32>
    %13 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x16x16xi32>
    %14 = stablehlo.subtract %12, %13 : tensor<8x16x16xi32>  // q1 - z1
    %15 = stablehlo.convert %11 : (tensor<8x16x4xi8>) -> tensor<8x16x4xi32>
    %16 = stablehlo.broadcast_in_dim %6, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x16x4xi32>
    %17 = stablehlo.subtract %15, %16 : tensor<8x16x4xi32>  // q2 - z2
    // Corresponds to einsum expression: b i j, b j d -> b i d
    %18 = stablehlo.dot_general %14, %17, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x16x16xi32>, tensor<8x16x4xi32>) -> tensor<8x16x4xi32>
    %19 = stablehlo.convert %18 : (tensor<8x16x4xi32>) -> tensor<8x16x4xf32>
    %20 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<8x16x4xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<8x16x4xf32>  // * s1 s2
    %22 = call @uniform_quantize_1(%21, %7, %8) : (tensor<8x16x4xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xi8>
    %23 = call @uniform_dequantize(%22, %7, %8) : (tensor<8x16x4xi8>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xf32>
    return %23 : tensor<8x16x4xf32>
  }
// CHECK: %[[UQ_0:.*]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<8x16x16xf32>) -> tensor<8x16x16x!quant.uniform<i8:f32, 5.000000e+00:-128>>
// CHECK: %[[UQ_1:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<8x16x4xf32>) -> tensor<8x16x4x!quant.uniform<i8:f32, 2.500000e+00>>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[UQ_0]], %[[UQ_1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x16x16x!quant.uniform<i8:f32, 5.000000e+00:-128>>, tensor<8x16x4x!quant.uniform<i8:f32, 2.500000e+00>>) -> tensor<8x16x4x!quant.uniform<i8:f32, 2.000000e+00:-5>>
// CHECK: %[[DQ_0:.*]] = stablehlo.uniform_dequantize %[[DOT_GENERAL]] : (tensor<8x16x4x!quant.uniform<i8:f32, 2.000000e+00:-5>>) -> tensor<8x16x4xf32>
// CHECK: return %[[DQ_0]]

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize(%arg0: tensor<8x16x16xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x16xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x16xf32>) -> tensor<8x16x16xi8>
    return %0 : tensor<8x16x16xi8>
  }
  func.func private @uniform_quantize_0(%arg0: tensor<8x16x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xf32>) -> tensor<8x16x4xi8>
    return %0 : tensor<8x16x4xi8>
  }
  func.func private @uniform_quantize_1(%arg0: tensor<8x16x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xf32>) -> tensor<8x16x4xi8>
    return %0 : tensor<8x16x4xi8>
  }
  func.func private @uniform_dequantize(%arg0: tensor<8x16x4xi8>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xi8>) -> tensor<8x16x4xf32>
    return %0 : tensor<8x16x4xf32>
  }
}

// -----

// Tests that a quantized dot_general op is composed when both operands are
// activations, where input zero points are not folded into i32 constants.

// CHECK-LABEL: dot_general_with_two_activations
// CHECK-SAME: %[[ARG_0:.*]]: tensor<8x16x16xf32>
// CHECK-SAME: %[[ARG_1:.*]]: tensor<8x16x4xf32>
module {
  func.func @dot_general_with_two_activations(%arg0: tensor<8x16x16xf32>, %arg1: tensor<8x16x4xf32>) -> tensor<8x16x4xf32> {
    %1 = stablehlo.constant dense<2.000000e-01> : tensor<1x1x1xf32>  // Input 1 inverse scale (1 / s1).
    %2 = stablehlo.constant dense<-128> : tensor<1x1x1xi8>  // Input 1 zero point (z1).
    %3 = stablehlo.constant dense<4.000000e-01> : tensor<1x1x1xf32>  // Input 2 inverse scale (1 / s2).
    %4 = stablehlo.constant dense<0> : tensor<1x1x1xi8>  // Input 2 zero point (z2).
    %5 = stablehlo.constant dense<5.000000e-01> : tensor<1x1x1xf32>  // Output inverse scale (1 / s3).
    %6 = stablehlo.constant dense<-5> : tensor<1x1x1xi8>  // Output zero point (z3).
    %7 = stablehlo.constant dense<1.250000e+01> : tensor<1x1x1xf32>  // Merged scale (s1 * s2).
    %8 = call @uniform_quantize(%arg0, %1, %2) : (tensor<8x16x16xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x16xi8>  // q1
    %9 = call @uniform_quantize_0(%arg1, %3, %4) : (tensor<8x16x4xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xi8>  // q2
    %10 = stablehlo.convert %8 : (tensor<8x16x16xi8>) -> tensor<8x16x16xi32>
    %11 = stablehlo.convert %2 : (tensor<1x1x1xi8>) -> tensor<1x1x1xi32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x16x16xi32>
    %13 = stablehlo.subtract %10, %12 : tensor<8x16x16xi32>  // q1 - z1
    %14 = stablehlo.convert %9 : (tensor<8x16x4xi8>) -> tensor<8x16x4xi32>
    %15 = stablehlo.convert %4 : (tensor<1x1x1xi8>) -> tensor<1x1x1xi32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<8x16x4xi32>
    %17 = stablehlo.subtract %14, %16 : tensor<8x16x4xi32>  // q2 - z2
    // Corresponds to einsum expression: b i j, b j d -> b i d
    %18 = stablehlo.dot_general %13, %17, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x16x16xi32>, tensor<8x16x4xi32>) -> tensor<8x16x4xi32>
    %19 = stablehlo.convert %18 : (tensor<8x16x4xi32>) -> tensor<8x16x4xf32>
    %20 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<8x16x4xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<8x16x4xf32>  // * s1 s2
    %22 = call @uniform_quantize_1(%21, %5, %6) : (tensor<8x16x4xf32>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xi8>
    %23 = call @uniform_dequantize(%22, %5, %6) : (tensor<8x16x4xi8>, tensor<1x1x1xf32>, tensor<1x1x1xi8>) -> tensor<8x16x4xf32>
    return %23 : tensor<8x16x4xf32>
  }
// CHECK: %[[UQ_0:.*]] = stablehlo.uniform_quantize %[[ARG_0]] : (tensor<8x16x16xf32>) -> tensor<8x16x16x!quant.uniform<i8:f32, 5.000000e+00:-128>>
// CHECK: %[[UQ_1:.*]] = stablehlo.uniform_quantize %[[ARG_1]] : (tensor<8x16x4xf32>) -> tensor<8x16x4x!quant.uniform<i8:f32, 2.500000e+00>>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[UQ_0]], %[[UQ_1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x16x16x!quant.uniform<i8:f32, 5.000000e+00:-128>>, tensor<8x16x4x!quant.uniform<i8:f32, 2.500000e+00>>) -> tensor<8x16x4x!quant.uniform<i8:f32, 2.000000e+00:-5>>
// CHECK: %[[DQ_0:.*]] = stablehlo.uniform_dequantize %[[DOT_GENERAL]] : (tensor<8x16x4x!quant.uniform<i8:f32, 2.000000e+00:-5>>) -> tensor<8x16x4xf32>
// CHECK: return %[[DQ_0]]

  // The following uniform_quantize & uniform_dequantize functions do NOT have
  // the correct body. Only the type signatures matter for testing.
  func.func private @uniform_quantize(%arg0: tensor<8x16x16xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x16xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x16xf32>) -> tensor<8x16x16xi8>
    return %0 : tensor<8x16x16xi8>
  }
  func.func private @uniform_quantize_0(%arg0: tensor<8x16x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xf32>) -> tensor<8x16x4xi8>
    return %0 : tensor<8x16x4xi8>
  }
  func.func private @uniform_quantize_1(%arg0: tensor<8x16x4xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xi8> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xf32>) -> tensor<8x16x4xi8>
    return %0 : tensor<8x16x4xi8>
  }
  func.func private @uniform_dequantize(%arg0: tensor<8x16x4xi8>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x1xi8>) -> tensor<8x16x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<8x16x4xi8>) -> tensor<8x16x4xf32>
    return %0 : tensor<8x16x4xf32>
  }
}
