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
// CHECK: %[[FILTER:.*]] = stablehlo.constant() {value = dense<1> : tensor<3x3x4x4xi8>} : () -> tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{{.*}}}>>
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
// CHECK: %[[FILTER:.*]] = stablehlo.constant() {value = dense<20> : tensor<3x3x4x4xi8>} : () -> tensor<3x3x4x4x!quant.uniform<i8:f32:3, {{{.*}}}>>
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
