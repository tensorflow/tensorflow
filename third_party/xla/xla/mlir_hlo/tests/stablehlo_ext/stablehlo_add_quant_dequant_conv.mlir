// RUN: mlir-hlo-opt --stablehlo-ext-add-qdq-after-conv --split-input-file --verify-diagnostics %s | FileCheck %s

module {
// CHECK-LABEL: func.func @add_qdq_after_conv
// CHECK-SAME: %[[VAL_0:.*]]: tensor<64x3x7x7xf32>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
// CHECK: %[[VAL_2:.*]] = stablehlo.uniform_quantize %[[VAL_1]] : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
// CHECK: %[[VAL_3:.*]] = stablehlo.uniform_dequantize %[[VAL_2]] : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
// CHECK: %[[VAL_4:.*]] = stablehlo.uniform_quantize %[[VAL_0]] : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
// CHECK: %[[VAL_5:.*]] = stablehlo.uniform_dequantize %[[VAL_4]] : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
// CHECK: %[[VAL_6:.*]] = stablehlo.convolution(%[[VAL_3]], %[[VAL_5]])
// CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
// CHECK: %[[VAL_7:.*]] = stablehlo.uniform_quantize %[[VAL_6]] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i32:f32, 1.000000e+00>>
// CHECK: %[[VAL_8:.*]] = stablehlo.uniform_dequantize %[[VAL_7]] : (tensor<1x64x112x112x!quant.uniform<i32:f32, 1.000000e+00>>) -> tensor<1x64x112x112xf32>
// CHECK: return %[[VAL_8]] : tensor<1x64x112x112xf32>
  func.func @add_qdq_after_conv(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %0 = stablehlo.uniform_quantize %arg1 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
    %2 = stablehlo.uniform_quantize %arg0 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
    %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    return %4 : tensor<1x64x112x112xf32>
  }
}

// -----

module {
// CHECK-LABEL: func.func @pass_failed_to_match_convolution_user_is_uniform_quantize
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<64x3x7x7xf32>,
// CHECK-SAME:  %[[VAL_1:.*]]: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
// CHECK: %[[VAL_2:.*]] = stablehlo.uniform_quantize %[[VAL_1]] : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
// CHECK: %[[VAL_3:.*]] = stablehlo.uniform_dequantize %[[VAL_2]] : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
// CHECK: %[[VAL_4:.*]] = stablehlo.uniform_quantize %[[VAL_0]] : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
// CHECK: %[[VAL_5:.*]] = stablehlo.uniform_dequantize %[[VAL_4]] : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
// CHECK: %[[VAL_6:.*]] = stablehlo.convolution(%[[VAL_3]], %[[VAL_5]])
// CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
// CHECK: %[[VAL_7:.*]] = stablehlo.uniform_quantize %[[VAL_6]] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 1.500000e+00:2>>
// CHECK: %[[VAL_8:.*]] = stablehlo.uniform_dequantize %[[VAL_7]] : (tensor<1x64x112x112x!quant.uniform<i8:f32, 1.500000e+00:2>>) -> tensor<1x64x112x112xf32>
// CHECK: return %[[VAL_8]] : tensor<1x64x112x112xf32>
  func.func @pass_failed_to_match_convolution_user_is_uniform_quantize(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
    %0 = stablehlo.uniform_quantize %arg1 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
    %2 = stablehlo.uniform_quantize %arg0 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
    %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i8:f32, 1.5:2>>
    %6 = stablehlo.uniform_dequantize %5 : (tensor<1x64x112x112x!quant.uniform<i8:f32, 1.5:2>>) -> tensor<1x64x112x112xf32>
    return %6 : tensor<1x64x112x112xf32>
  }
}

// -----

module {
// CHECK-LABEL: func.func @pass_failed_to_match_more_than_one_convolution_user
// CHECK-SAME: %[[VAL_0:.*]]: tensor<64x3x7x7xf32>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<1x3x224x224xf32>) -> (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) {
// CHECK: %[[VAL_2:.*]] = stablehlo.uniform_quantize %[[VAL_1]] : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
// CHECK: %[[VAL_3:.*]] = stablehlo.uniform_dequantize %[[VAL_2]] : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
// CHECK: %[[VAL_4:.*]] = stablehlo.uniform_quantize %[[VAL_0]] : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
// CHECK: %[[VAL_5:.*]] = stablehlo.uniform_dequantize %[[VAL_4]] : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
// CHECK: %[[VAL_6:.*]] = stablehlo.convolution(%[[VAL_3]], %[[VAL_5]])
// CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
// CHECK: %[[VAL_7:.*]] = stablehlo.uniform_quantize %[[VAL_6]] : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i32:f32, 1.500000e+00:2>>
// CHECK: %[[VAL_8:.*]] = stablehlo.uniform_dequantize %[[VAL_7]] : (tensor<1x64x112x112x!quant.uniform<i32:f32, 1.500000e+00:2>>) -> tensor<1x64x112x112xf32>
// CHECK: return %[[VAL_6]], %[[VAL_8]] : tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>
  func.func @pass_failed_to_match_more_than_one_convolution_user(%arg0: tensor<64x3x7x7xf32>, %arg1: tensor<1x3x224x224xf32>) -> (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) {
    %0 = stablehlo.uniform_quantize %arg1 : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<1x3x224x224x!quant.uniform<i8:f32, 3.158280e-02:1>>) -> tensor<1x3x224x224xf32>
    %2 = stablehlo.uniform_quantize %arg0 : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<64x3x7x7x!quant.uniform<i8<-127:127>:f32, 3.101380e-03>>) -> tensor<64x3x7x7xf32>
    %4 = stablehlo.convolution(%1, %3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    %5 = stablehlo.uniform_quantize %4 : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112x!quant.uniform<i32:f32, 1.5:2>>
    %6 = stablehlo.uniform_dequantize %5 : (tensor<1x64x112x112x!quant.uniform<i32:f32, 1.5:2>>) -> tensor<1x64x112x112xf32>
    return %4, %6 : tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>
  }
}

