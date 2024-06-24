// RUN: tf-opt -tfl-prepare-tf %s | FileCheck %s
// RUN: tf-opt %s -tf-layout-optimization=force-data-format=NHWC -tfl-prepare-tf  | FileCheck --check-prefix=LAYOUT --dump-input=always %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

func.func @conv(tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<256x3x32x32xf32>) -> (tensor<256x8x7x16xf32>, tensor<256x16x32x32xf32>, tensor<256x8x6x16xf32>, tensor<256x32x32x16xf32>, tensor<256x32x32x16xf32>) {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>, %arg2: tensor<256x3x32x32xf32>) :
   // OK
   %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
   // Unsupported data format
   %1 = "tf.Conv2D"(%arg2, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x3x32x32xf32>, tensor<3x3x3x16xf32>) -> tensor<256x16x32x32xf32>
   // OK
   %2 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC",                           padding = "VALID", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x6x16xf32>
   // Unsupported padding
   %3 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "EXPLICIT", strides = [1, 1, 1, 1], explicit_paddings = [0, 0, 1, 1, 1, 1, 0, 0]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
   // Unsupported strides
   %4 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [2, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>

  func.return %0, %1, %2, %3, %4 : tensor<256x8x7x16xf32>, tensor<256x16x32x32xf32>, tensor<256x8x6x16xf32>, tensor<256x32x32x16xf32>, tensor<256x32x32x16xf32>

// CHECK-LABEL: conv
// CHECK-DAG:  %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG:  %[[CONSTANT0:.*]] = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK-DAG:  %[[CONSTANT1:.*]] = arith.constant dense<[{{\[}}0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
// CHECK:  %0 = "tf.Transpose"(%arg1, %[[CONSTANT0]]) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %1 = "tfl.conv_2d"(%arg0, %0, %[[CONSTANT]]) <{dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
// CHECK:  %2 = "tf.Conv2D"
// CHECK:  %3 = "tf.Transpose"(%arg1, %[[CONSTANT0]]) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %4 = "tfl.conv_2d"(%arg0, %3, %[[CONSTANT]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x8x6x16xf32>
// CHECK:  %5 = "tf.Pad"(%arg0, %[[CONSTANT1]]) : (tensor<256x32x32x3xf32>, tensor<4x2xi32>) -> tensor<*xf32>
// CHECK:  %6 = "tf.Transpose"(%arg1, %[[CONSTANT0]]) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %7 = "tfl.conv_2d"(%5, %6, %[[CONSTANT]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<*xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
// CHECK:  %8 = "tf.Conv2D"(%arg0, %arg1) <{data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [2, 1, 1, 1]}> {T = "tfdtype$DT_FLOAT"} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
}

func.func @depthwiseConv2D(tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>, tensor<256x3x32x32xf32>) -> (tensor<256x30x30x12xf32>, tensor<256x12x30x30xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>) {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x4xf32>, %arg2: tensor<256x3x32x32xf32>) :
   // OK
   %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
   // Unsupported data format
   %1 = "tf.DepthwiseConv2dNative"(%arg2, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x3x32x32xf32>, tensor<3x3x3x4xf32>) -> tensor<256x12x30x30xf32>
   // OK
   %2 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC",                           padding = "VALID", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
   // Unsupported strides
   %3 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [2, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>

  func.return %0, %1, %2, %3 : tensor<256x30x30x12xf32>, tensor<256x12x30x30xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>

// CHECK-LABEL: depthwiseConv2D
// CHECK-DAG:  %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<12xf32>
// CHECK-DAG:  %[[CONSTANT0:.*]] = arith.constant dense<[1, 3, 3, 12]> : tensor<4xi32>
// CHECK:  %0 = "tf.Reshape"(%arg1, %[[CONSTANT0]]) : (tensor<3x3x3x4xf32>, tensor<4xi32>) -> tensor<1x3x3x12xf32>
// CHECK:  %1 = "tfl.depthwise_conv_2d"(%arg0, %0, %[[CONSTANT]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<1x3x3x12xf32>, tensor<12xf32>) -> tensor<256x30x30x12xf32>
// CHECK:  %2 = "tf.DepthwiseConv2dNative"
// CHECK:  %3 = "tf.Reshape"(%arg1, %[[CONSTANT0]]) : (tensor<3x3x3x4xf32>, tensor<4xi32>) -> tensor<1x3x3x12xf32>
// CHECK:  %4 = "tfl.depthwise_conv_2d"(%arg0, %3, %[[CONSTANT]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<1x3x3x12xf32>, tensor<12xf32>) -> tensor<256x30x30x12xf32>
// CHECK:  %5 = "tf.DepthwiseConv2dNative"
}

func.func @Conv2dNCHW(%arg0: tensor<256x3x32x32xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x16x32x32xf32> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x3x32x32xf32>, tensor<3x3x3x16xf32>) -> tensor<256x16x32x32xf32>
  func.return %0 : tensor<256x16x32x32xf32>

  // LAYOUT-LABEL: Conv2dNCHW
  // LAYOUT: "tfl.conv_2d"
}

func.func @fusedBatchNormV3(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>) {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // OK
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", U = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // Training with non-broadcastable shape
  %cst = arith.constant dense<0.0> : tensor<4xf32>
  %1:6 = "tf.FusedBatchNormV3"( %0#0, %cst, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", U = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true}  : (tensor<8x8x8x8xf32>, tensor<4xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // Inference with non-broadcastable shape
  %2:6 = "tf.FusedBatchNormV3"( %1#0, %cst, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", U = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<4xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // Use other output
  %3:6 = "tf.FusedBatchNormV3"( %2#0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", U = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)

  func.return %3, %3#1 : tensor<8x8x8x8xf32>, tensor<8xf32>

// CHECK-LABEL: fusedBatchNormV3
// CHECK-DAG:  %[[CONSTANT:.*]] = arith.constant dense<1.000000e-03>
// CHECK-DAG:  %[[CONSTANT1:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
//              variance + epsilon
// CHECK:  %[[ADD1:.*]] = "tf.Add"(%[[ARG4:.*]], %[[CONSTANT]])
//              rsqrt(variance + epsilon)
// CHECK:  %[[RSQRT:.*]] = "tf.Rsqrt"(%[[ADD1]])
//              scale * rsqrt(variance + epsilon)
// CHECK:  %[[MUL1:.*]] = "tf.Mul"(%[[ARG1:.*]], %[[RSQRT]])
//              x * scale * rsqrt(variance + epsilon)
// CHECK:  %[[MUL2:.*]] = "tf.Mul"(%[[ARG0:.*]], %[[MUL1]])
//              mean * scale * rsqrt(variance + epsilon)
// CHECK:  %[[MUL3:.*]] = "tf.Mul"(%[[ARG3:.*]], %[[MUL1]])
//              offset - mean * scale * rsqrt(variance + epsilon)
// CHECK:  %[[SUB:.*]] = "tf.Sub"(%[[ARG2:.*]], %[[MUL3]])
//              x * scale * rsqrt(variance + epsilon) +
//              offset - mean * scale * rsqrt(variance + epsilon)
// CHECK:  %[[ADD2:.*]] = "tf.Add"(%[[MUL2]], %[[SUB]])
// CHECK:  %[[BATCHNORM1_a:[^,]+]], {{.*}} = "tf.FusedBatchNormV3"(%[[ADD2]], %[[CONSTANT1]], %[[ARG2]], %[[ARG3]], %[[ARG4]])
// CHECK:  %[[BATCHNORM1_b:[^,]+]], {{.*}} = "tf.FusedBatchNormV3"(%[[BATCHNORM1_a]], %[[CONSTANT1]], %[[ARG2]], %[[ARG3]], %[[ARG4]])
// CHECK:  "tf.FusedBatchNormV3"(%[[BATCHNORM1_b]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]])
}


func.func @batchNormWithGlobalNormalization(
    %t:tensor<1x10x10x3xf32>, %m:tensor<3xf32>, %v:tensor<3xf32>, %beta:tensor<3xf32>, %gamma:tensor<3xf32>) -> (tensor<1x10x10x3xf32>) {
  %0 = "tf.BatchNormWithGlobalNormalization"(%t, %m, %v, %beta, %gamma) {T = "tfdtype$DT_FLOAT", variance_epsilon = 0.001 : f32, scale_after_normalization = false} : (tensor<1x10x10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x10x10x3xf32>)
  func.return %0 : tensor<1x10x10x3xf32>
// CHECK-LABEL: batchNormWithGlobalNormalization
// CHECK:  %[[EPSILON:.*]] = arith.constant dense<1.000000e-03>
// CHECK:  %[[VARIANCE:.*]] = "tf.Add"(%[[ARG_V:.*]], %[[EPSILON]])
// CHECK:  %[[RSQRT:.*]] = "tf.Rsqrt"(%[[VARIANCE]])
// CHECK:  %[[MUL1:.*]] = "tf.Mul"(%[[ARG_T:.*]], %[[RSQRT]])
// CHECK:  %[[MUL2:.*]] = "tf.Mul"(%[[ARG_M:.*]], %[[RSQRT]])
// CHECK:  %[[SUB:.*]] = "tf.Sub"(%[[ARG_BETA:.*]], %[[MUL2]])
// CHECK:  %[[RESULT:.*]] = "tf.Add"(%[[MUL1]], %[[SUB]])
// CHECK:  return %[[RESULT]]
}

func.func @batchNormWithGlobalNormalizationWithScaleAfterNormalization(
    %t:tensor<1x10x10x3xf32>, %m:tensor<3xf32>, %v:tensor<3xf32>, %beta:tensor<3xf32>, %gamma:tensor<3xf32>) -> (tensor<1x10x10x3xf32>) {
  %0 = "tf.BatchNormWithGlobalNormalization"(%t, %m, %v, %beta, %gamma) {T = "tfdtype$DT_FLOAT", variance_epsilon = 0.001 : f32, scale_after_normalization = true} : (tensor<1x10x10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<1x10x10x3xf32>)
  func.return %0 : tensor<1x10x10x3xf32>
// CHECK-LABEL: batchNormWithGlobalNormalizationWithScaleAfterNormalization
// CHECK:  %[[EPSILON:.*]] = arith.constant dense<1.000000e-03>
// CHECK:  %[[VARIANCE:.*]] = "tf.Add"(%[[ARG_V:.*]], %[[EPSILON]])
// CHECK:  %[[RSQRT:.*]] = "tf.Rsqrt"(%[[VARIANCE]])
// CHECK:  %[[MUL0:.*]] = "tf.Mul"(%[[RSQRT]], %[[ARG_GAMMA:.*]])
// CHECK:  %[[MUL1:.*]] = "tf.Mul"(%[[ARG_T:.*]], %[[MUL0]])
// CHECK:  %[[MUL2:.*]] = "tf.Mul"(%[[ARG_M:.*]], %[[MUL0]])
// CHECK:  %[[SUB:.*]] = "tf.Sub"(%[[ARG_BETA:.*]], %[[MUL2]])
// CHECK:  %[[RESULT:.*]] = "tf.Add"(%[[MUL1]], %[[SUB]])
// CHECK:  return %[[RESULT]]
}

func.func @QDQsFollowedByTranspose(tensor<1x2xf32>) -> (tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>):
  %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0){qtype = tensor<1x2x!quant.uniform<u8:f32, 1.0>>}: (tensor<1x2xf32>) -> (tensor<1x2x!quant.uniform<u8:f32, 1.0>>)
  %1 = "tfl.dequantize"(%0): (tensor<1x2x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1x2xf32>)
  %2 = "tf.Transpose"(%1, %cst_0): (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  func.return %2 : tensor<2x1xf32>

// CHECK: %cst = arith.constant
// CHECK: %[[trans:.*]] = "tf.Transpose"
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[trans]]) <{qtype = tensor<2x1x!quant.uniform<u8:f32, 1.000000e+00>>}>
// CHECK-SAME: -> tensor<2x1x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK: return %[[dq]]
}

// CHECK-LABEL: QDQFollowedByReshape
func.func @QDQFollowedByReshape(tensor<1x2xf32>) -> (tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>):
  %cst_0 = arith.constant dense<[2, 1]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0){qtype = tensor<1x2x!quant.uniform<u8:f32, 1.0>>}: (tensor<1x2xf32>) -> (tensor<1x2x!quant.uniform<u8:f32, 1.0>>)
  %1 = "tfl.dequantize"(%0): (tensor<1x2x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1x2xf32>)
  %2 = "tf.Reshape"(%1, %cst_0): (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  func.return %2 : tensor<2x1xf32>

// CHECK: %cst = arith.constant
// CHECK: %[[rs:.*]] = "tf.Reshape"
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[rs]]) <{qtype = tensor<2x1x!quant.uniform<u8:f32, 1.000000e+00>>}>
// CHECK-SAME: -> tensor<2x1x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK: return %[[dq]]
}

// CHECK-LABEL: QDQFollowedByRank
func.func @QDQFollowedByRank(%arg0: tensor<1x2xf32>) -> (tensor<i32>) {
  %0 = "tfl.quantize"(%arg0){qtype = tensor<1x2x!quant.uniform<u8:f32, 1.0>>}: (tensor<1x2xf32>) -> (tensor<1x2x!quant.uniform<u8:f32, 1.0>>)
  %1 = "tfl.dequantize"(%0): (tensor<1x2x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1x2xf32>)
  %2 = "tf.Rank"(%1): (tensor<1x2xf32>) -> tensor<i32>
  func.return %2 : tensor<i32>

// CHECK: %[[R:.*]] = arith.constant dense<2>
// CHECK: return %cst : tensor<i32>
}

func.func @identity(%arg0: tensor<10xi32>, %arg1: tensor<20xi32>, %arg2: tensor<30xi32>) -> (tensor<10xi32>, tensor<20xi32>, tensor<30xi32>, tensor<*xi32>) {
  %0 = "tf.Identity"(%arg0) : (tensor<10xi32>) -> tensor<10xi32>
  %1:2 = "tf.IdentityN"(%arg1,%arg2) : (tensor<20xi32>, tensor<30xi32>) -> (tensor<20xi32>, tensor<30xi32>)
  %2 = "tf.Identity"(%arg0) : (tensor<10xi32>) -> tensor<*xi32>
  func.return %0, %1#0, %1#1, %2: tensor<10xi32>, tensor<20xi32>, tensor<30xi32>, tensor<*xi32>

// CHECK-LABEL: identity
// CHECK: %0 = "tf.Identity"(%arg0) : (tensor<10xi32>) -> tensor<*xi32>
// CHECK: return %arg0, %arg1, %arg2, %0
}

func.func @sharding(%arg0: tensor<10x10xi32>) -> (tensor<10x10xi32>) {
  %0 = "tf.MatMul"(%arg0, %arg0) {device = "", transpose_a = false, transpose_b = false} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  %1 = "tf.MatMul"(%arg0, %arg0) {device = "", transpose_a = false, transpose_b = false} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  %2 = "tf.XlaSharding"(%0) {_XlaSharding = "\08\03\1A\02\01\01\22\01\00", device = "", sharding = "\08\03\1A\02\01\01\22\01\00", unspecified_dims = []} : (tensor<10x10xi32>) -> tensor<10x10xi32>
  %3 = "tf.XlaSharding"(%1) {_XlaSharding = "\08\03\1A\02\01\01\22\01\00", device = "", sharding = "\08\03\1A\02\01\01\22\01\00", unspecified_dims = []} : (tensor<10x10xi32>) -> tensor<10x10xi32>
  %4 = "tf.AddV2"(%2, %3) {device = ""} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  func.return %4 : tensor<10x10xi32>

// CHECK-LABEL: sharding
// CHECK-NOT: %2 = "tf.XlaSharding"(%0) {_XlaSharding = "\08\03\1A\02\01\01\22\01\00", device = "", sharding = "\08\03\1A\02\01\01\22\01\00", unspecified_dims = []} : (tensor<10x10xi32>) -> tensor<10x10xi32>
// CHECK-NOT: %3 = "tf.XlaSharding"(%1) {_XlaSharding = "\08\03\1A\02\01\01\22\01\00", device = "", sharding = "\08\03\1A\02\01\01\22\01\00", unspecified_dims = []} : (tensor<10x10xi32>) -> tensor<10x10xi32>
}

func.func @preventGradient(%arg0: tensor<10x10xi32>) -> (tensor<10x10xi32>) {
  %0 = "tf.MatMul"(%arg0, %arg0) {device = "", transpose_a = false, transpose_b = false} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  %1 = "tf.MatMul"(%arg0, %arg0) {device = "", transpose_a = false, transpose_b = false} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  %2 = "tf.PreventGradient"(%0) : (tensor<10x10xi32>) -> tensor<10x10xi32>
  %3 = "tf.PreventGradient"(%1) : (tensor<10x10xi32>) -> tensor<10x10xi32>
  %4 = "tf.AddV2"(%2, %3) {device = ""} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xi32>
  func.return %4 : tensor<10x10xi32>

// CHECK-LABEL: preventGradient
// CHECK-NOT: %2 = "tf.PreventGradient"(%0) : (tensor<10x10xi32>) -> tensor<10x10xi32>
// CHECK-NOT: %3 = "tf.PreventGradient"(%1) : (tensor<10x10xi32>) -> tensor<10x10xi32>
}

func.func @matmulNoTransposeAOrB(%arg0: tensor<1x1280xf32>, %arg1: tensor<1280x1000xf32>) -> tensor<1x1000xf32> {
  %166 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", _output_shapes = ["tfshape$dim { size = 1} dim { size = 1000}"], device = "", name = "matmul", transpose_a = false, transpose_b = false} : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
  func.return %166 : tensor<1x1000xf32>

  // CHECK-LABEL: matmulNoTransposeAOrB
  // CHECK: %[[RES:.*]] = "tf.Const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<?xi32>
  // CHECK: %[[TRANS:.*]] = "tf.Transpose"(%arg1, %[[RES]]) : (tensor<1280x1000xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %[[MM:.*]] = "tf.MatMul"(%arg0, %[[TRANS]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = true}> : (tensor<1x1280xf32>, tensor<*xf32>) -> tensor<1x1000xf32>
  // CHECK: return %[[MM]] : tensor<1x1000xf32>
 }

func.func @matmulNoTransposeB(%arg0: tensor<1x1280xf32>, %arg1: tensor<1280x1000xf32>) -> tensor<1x1000xf32> {
  %166 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", _output_shapes = ["tfshape$dim { size = 1} dim { size = 1000}"], device = "", name = "matmul", transpose_a = true, transpose_b = false} : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
  func.return %166 : tensor<1x1000xf32>

  // CHECK-LABEL: matmulNoTransposeB
  // CHECK: %[[RES:.*]] = "tf.Const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<?xi32>
  // CHECK: %[[TRANS1:.*]] = "tf.Transpose"(%arg0, %[[RES]]) : (tensor<1x1280xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %[[TRANS2:.*]] = "tf.Transpose"(%arg1, %[[RES]]) : (tensor<1280x1000xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %[[MM:.*]] = "tf.MatMul"(%[[TRANS1]], %[[TRANS2]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = true}> : (tensor<*xf32>, tensor<*xf32>) -> tensor<1x1000xf32>
  // CHECK: return %[[MM]] : tensor<1x1000xf32>

}

func.func @snapshot(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "tf.Snapshot"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-LABEL: snapshot
  // CHECK:  return %arg0 : tensor<3xi32>
}

func.func @stop_gradient(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "tf.StopGradient"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-LABEL: stop_gradient
  // CHECK:  return %arg0 : tensor<3xi32>
}

func.func @CheckNumerics(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "tf.CheckNumerics"(%arg0) {message = ""}: (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-LABEL: CheckNumerics
  // CHECK:  return %arg0 : tensor<3xf32>
}

func.func @placeholder_with_default(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "tf.PlaceholderWithDefault"(%arg0): (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
  // Should be converted to Identity and then from Identity to value
  // CHECK-LABEL: placeholder_with_default
  // CHECK:  return %arg0 : tensor<3xf32>
}

// CHECK-LABEL: @StridedSliceEllipsisMaskBefore
func.func @StridedSliceEllipsisMaskBefore(%arg0: tensor<21x15x7xf32>) -> tensor<21x15x2xf32> {
  %cst = arith.constant dense<0> : tensor<2xi32>
  %cst_0 = arith.constant dense<1> : tensor<2xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 0 : i64, ellipsis_mask = 1 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<21x15x7xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<21x15x2xf32>
  func.return %0 : tensor<21x15x2xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<3xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1> : tensor<3xi32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST]], %[[CST_0]]) <{begin_mask = 3 : i64, ellipsis_mask = 0 : i64, end_mask = 3 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<21x15x7xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<21x15x2xf32>
}

// CHECK-LABEL: @StridedSliceEllipsisMaskBeforeWithBeginAndEndMask
func.func @StridedSliceEllipsisMaskBeforeWithBeginAndEndMask(%arg0: tensor<4x5x4xf32>) -> tensor<4x4x4xf32> {
  %cst = arith.constant dense<[0, 1, 0]> : tensor<3xi32>
  %cst_0 = arith.constant dense<0> : tensor<3xi32>
  %cst_1 = arith.constant dense<1> : tensor<3xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 6 : i64, ellipsis_mask = 1 : i64, end_mask = 4 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x5x4xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x4x4xf32>
  func.return %0 : tensor<4x4x4xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1, 0]> : tensor<3xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<0> : tensor<3xi32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<1> : tensor<3xi32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST_0]], %[[CST_1]]) <{begin_mask = 7 : i64, ellipsis_mask = 0 : i64, end_mask = 5 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<4x5x4xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x4x4xf32>
}

// CHECK-LABEL: @StridedSliceEllipsisMaskAfter
func.func @StridedSliceEllipsisMaskAfter(%arg0: tensor<21x15x7xf32>) -> tensor<5x15x7xf32> {
  %cst = arith.constant dense<0> : tensor<2xi32>
  %cst_0 = arith.constant dense<1> : tensor<2xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 0 : i64, ellipsis_mask = 2 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<21x15x7xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5x15x7xf32>
  func.return %0 : tensor<5x15x7xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<3xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1> : tensor<3xi32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST]], %[[CST_0]]) <{begin_mask = 6 : i64, ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<21x15x7xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<5x15x7xf32>
}

// CHECK-LABEL: @NoStridedSliceEllipsisMask
func.func @NoStridedSliceEllipsisMask(%arg0: tensor<*xf32>) -> tensor<21x15x2xf32> {
  %cst = arith.constant dense<0> : tensor<2xi32>
  %cst_0 = arith.constant dense<1> : tensor<2xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 0 : i64, ellipsis_mask = 1 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<*xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<21x15x2xf32>
  func.return %0 : tensor<21x15x2xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<2xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST]], %[[CST_0]]) <{begin_mask = 0 : i64, ellipsis_mask = 1 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<*xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<21x15x2xf32>
}

// CHECK-LABEL: @NoPadStridedSliceNonNewAxisMask
func.func @NoPadStridedSliceNonNewAxisMask(%arg0: tensor<1x2x3x1xf32>) -> tensor<1x2x3x1xf32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %cst_0 = arith.constant dense<1> : tensor<4xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 15 : i64, ellipsis_mask = 0 : i64, end_mask = 15 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1x2x3x1xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
  func.return %0 : tensor<1x2x3x1xf32>

  // CHECK-DAG: %cst = arith.constant dense<0> : tensor<4xi32>
  // CHECK-DAG: %cst_0 = arith.constant dense<1> : tensor<4xi32>
  // CHECK: %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) <{begin_mask = 15 : i64, ellipsis_mask = 0 : i64, end_mask = 15 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<1x2x3x1xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
}

// CHECK-LABEL: @PadStridedSliceNewAxisMask1
func.func @PadStridedSliceNewAxisMask1(%arg0: tensor<2x3xf32>) -> tensor<1x2x3x1xf32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %cst_0 = arith.constant dense<1> : tensor<4xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 6 : i64, ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 9 : i64, shrink_axis_mask = 0 : i64} : (tensor<2x3xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
  func.return %0 : tensor<1x2x3x1xf32>

  // CHECK-DAG: %[[CST0:.*]] = arith.constant dense<0> : tensor<4xi32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[1, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %0 = "tf.Reshape"(%arg0, %[[cst_1]]) : (tensor<2x3xf32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
  // CHECK: %1 = "tf.StridedSlice"(%0, %[[CST0]], %[[CST0]], %[[CST1]]) <{begin_mask = 15 : i64, ellipsis_mask = 0 : i64, end_mask = 15 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<1x2x3x1xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x2x3x1xf32>
}

// CHECK-LABEL: @PadStridedSliceNewAxisMask2
func.func @PadStridedSliceNewAxisMask2(%arg0: tensor<4x64x64x1xf32>) -> tensor<1x4x64x64xf32> {
  %cst = arith.constant dense<0> : tensor<3xi32>
  %cst_0 = arith.constant dense<1> : tensor<3xi32>
  %0 = "tf.Squeeze"(%arg0) {T = f32, _output_shapes = ["tfshape$dim { size: 4 } dim { size: 64 } dim { size: 64 }"], device = "", squeeze_dims = []} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %1 = "tf.StridedSlice"(%0, %cst, %cst, %cst_0) {Index = i32, T = f32, _output_shapes = ["tfshape$dim { size: 1 } dim { size: 4 } dim { size: 64 } dim { size: 64 }"], begin_mask = 6 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 1 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x64x64xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x4x64x64xf32>
  func.return %1 : tensor<1x4x64x64xf32>
}

// CHECK-LABEL: @AvoidPadStridedSliceNewAxisMaskOnUnknownShapes
func.func @AvoidPadStridedSliceNewAxisMaskOnUnknownShapes(%arg0: tensor<?x?xf32>) -> tensor<1x?x?x1xf32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %cst_0 = arith.constant dense<1> : tensor<4xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst, %cst_0) {begin_mask = 6 : i64, ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 9 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x?xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x?x?x1xf32>
  func.return %0 : tensor<1x?x?x1xf32>

  // CHECK-NOT: "tf.Reshape"
  // CHECK: "tf.StridedSlice"
}

// CHECK-LABEL: @StridedSliceRewriteMasks
func.func @StridedSliceRewriteMasks(%arg0: tensor<8x4x16x2xf32>) -> tensor<8x4x16x1xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[1, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %cst_0 = "tf.Const"() {device = "", value = dense<[1, 0, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
  %cst_1 = "tf.Const"() {device = "", value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1, 0, 0, 1]> : tensor<4xi32>
  // CHECK-DAG: %[[CST0:.*]] = arith.constant dense<[1, 0, 0, 0]> : tensor<4xi32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST0]], %[[CST1]])
  // CHECK-SAME: begin_mask = 7 : i64
  // CHECK-SAME: ellipsis_mask = 0 : i64
  // CHECK-SAME: end_mask = 14 : i64
  // CHECK-SAME: new_axis_mask = 0 : i64
  // CHECK-SAME: shrink_axis_mask = 0 : i64

  %0 = "tf.StridedSlice"(%arg0, %cst, %cst_0, %cst_1) {begin_mask = 1 : i64, device = "", ellipsis_mask = 2 : i64, end_mask = 4 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<8x4x16x2xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<8x4x16x1xf32>
  func.return %0 : tensor<8x4x16x1xf32>
}

func.func @strided_slice_with_constant_attributes(%arg0: tensor<10x10x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<10x10xf32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %cst_1 = arith.constant dense<0> : tensor<1xi32>
  %cst_2 = arith.constant dense<1> : tensor<1xi32>
  %0 = "tf.StridedSlice"(%arg0, %cst, %cst_1, %cst_2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<10x10x10xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
  // CHECK-LABEL: strided_slice_with_constant_attributes
  // CHECK-DAG: [[BEGIN:%cst.*]] = arith.constant dense<[-1, 0, 0]> : tensor<3xi32>
  // CHECK-DAG: [[END:%cst.*]] = arith.constant dense<[0, 10, 10]> : tensor<3xi32>
  // CHECK-DAG: [[STRIDES:%cst.*]] = arith.constant dense<1> : tensor<3xi32>
  // CHECK-NEXT: "tf.StridedSlice"(%arg0, [[BEGIN]], [[END]], [[STRIDES]]) <{begin_mask = 6 : i64, ellipsis_mask = 0 : i64, end_mask = 6 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64}> : (tensor<10x10x10xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<10x10xf32>
}

// CHECK-LABEL: @StridedSliceEllipsisAndNewAxisMaskBothSet
func.func @StridedSliceEllipsisAndNewAxisMaskBothSet(%arg0: tensor<6x7x8xf32>) -> tensor<2x1x7x8x1xf32> {
  %begin = arith.constant dense<0> : tensor<4xi32>
  %end = arith.constant dense<[2,3,4,5]> : tensor<4xi32>
  %step = arith.constant dense<1> : tensor<4xi32>
  %0 = "tf.StridedSlice"(%arg0, %begin, %end, %step) {
    begin_mask = 0 : i64, ellipsis_mask = 4 : i64, end_mask = 0 : i64, new_axis_mask = 10 : i64, shrink_axis_mask = 0 : i64
  } : (tensor<6x7x8xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<2x1x7x8x1xf32>
  func.return %0 : tensor<2x1x7x8x1xf32>

  // CHECK-DAG: %[[BEGIN:.*]] = arith.constant dense<0> : tensor<5xi32>
  // CHECK-DAG: %[[END:.*]] = arith.constant dense<[2, 3, 0, 0, 5]> : tensor<5xi32>
  // CHECK-DAG: %[[STEP:.*]] = arith.constant dense<1> : tensor<5xi32>
  // CHECK-DAG: %[[NEW_DIMS:.*]] = arith.constant dense<[6, 1, 7, 8, 1]> : tensor<5xi32>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[NEW_DIMS]]) : (tensor<6x7x8xf32>, tensor<5xi32>) -> tensor<6x1x7x8x1xf32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%[[RESHAPE]], %[[BEGIN]], %[[END]], %[[STEP]]) <{begin_mask = 30 : i64, ellipsis_mask = 0 : i64, end_mask = 30 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<6x1x7x8x1xf32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<2x1x7x8x1xf32>
}

// CHECK-LABEL: @StridedSliceShrinkAxisAndNewAxisMaskBothSet
func.func @StridedSliceShrinkAxisAndNewAxisMaskBothSet(%arg0: tensor<6x7x8xf32>) -> tensor<1x4x1x8xf32> {
  %begin = arith.constant dense<0> : tensor<4xi32>
  %end = arith.constant dense<[2,3,4,5]> : tensor<4xi32>
  %step = arith.constant dense<1> : tensor<4xi32>
  %0 = "tf.StridedSlice"(%arg0, %begin, %end, %step) {
    begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 10 : i64, shrink_axis_mask = 1 : i64
  } : (tensor<6x7x8xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x4x1x8xf32>
  func.return %0 : tensor<1x4x1x8xf32>

  // CHECK-DAG: %[[NEW_DIMS:.*]] = arith.constant dense<[6, 1, 7, 1, 8]> : tensor<5xi32>
  // CHECK-DAG: %[[BEGIN:.*]] = arith.constant dense<0> : tensor<5xi32>
  // CHECK-DAG: %[[END:.*]] = arith.constant dense<[2, 3, 4, 5, 8]> : tensor<5xi32>
  // CHECK-DAG: %[[STEP:.*]] = arith.constant dense<1> : tensor<5xi32>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[NEW_DIMS]]) : (tensor<6x7x8xf32>, tensor<5xi32>) -> tensor<6x1x7x1x8xf32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tf.StridedSlice"(%[[RESHAPE]], %[[BEGIN]], %[[END]], %[[STEP]]) <{begin_mask = 26 : i64, ellipsis_mask = 0 : i64, end_mask = 26 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64}> : (tensor<6x1x7x1x8xf32>, tensor<5xi32>, tensor<5xi32>, tensor<5xi32>) -> tensor<1x4x1x8xf32>
}

func.func @broadcast_to_i16_low_dim(%input: tensor<3xi16>, %shape: tensor<2xi32>) -> tensor<3x3xi16> {
  %0 = "tf.BroadcastTo"(%input, %shape) : (tensor<3xi16>, tensor<2xi32>) -> tensor<3x3xi16>
  func.return %0: tensor<3x3xi16>

// CHECK-LABEL: broadcast_to_i16_low_dim
// CHECK:    %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xi16>, tensor<2xi32>) -> tensor<3x3xi16>
// CHECK:    return %0 : tensor<3x3xi16>
}

func.func @broadcast_to_low_dim_with_unknown_shape(%arg0: tensor<3xf32>, %arg1: tensor<*xi32>) -> tensor<3x3xf32> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xf32>, tensor<*xi32>) -> tensor<3x3xf32>
  func.return %0: tensor<3x3xf32>

// CHECK-LABEL: broadcast_to_low_dim_with_unknown_shape
// CHECK: %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xf32>, tensor<*xi32>) -> tensor<3x3xf32>
// CHECK: return %0 : tensor<3x3xf32>
}

func.func @broadcast_to_i32_low_dim_with_unknown_output(%input: tensor<3xi32>, %shape: tensor<2xi32>) -> tensor<*xi32> {
  %0 = "tf.BroadcastTo"(%input, %shape) : (tensor<3xi32>, tensor<2xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>

// CHECK-LABEL: broadcast_to_i32_low_dim_with_unknown_output
// CHECK:  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32>) -> tensor<*xi32>
// CHECK:  return %0 : tensor<*xi32>
}

func.func @broadcast_to_high_dim_with_unknown_shape(%arg0: tensor<1x2x3x4x5x6xf32>, %arg1: tensor<*xi32>) -> tensor<7x8x1x2x3x4x5x6xf32> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<1x2x3x4x5x6xf32>, tensor<*xi32>) -> tensor<7x8x1x2x3x4x5x6xf32>
  func.return %0: tensor<7x8x1x2x3x4x5x6xf32>

// CHECK-LABEL: broadcast_to_high_dim_with_unknown_shape
// CHECK:  [[BCT:%.*]] = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<1x2x3x4x5x6xf32>, tensor<*xi32>) -> tensor<7x8x1x2x3x4x5x6xf32>
// CHECK:  return [[BCT]] : tensor<7x8x1x2x3x4x5x6xf32>
}

func.func @broadcast_to_high_dim_with_unknown_output(%arg0: tensor<1x2x3x4x5x6xf32>, %arg1: tensor<8xi32>) -> tensor<*xf32> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<1x2x3x4x5x6xf32>, tensor<8xi32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>

// CHECK-LABEL: broadcast_to_high_dim_with_unknown_output
// CHECK:  [[BCT:%.*]] = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<1x2x3x4x5x6xf32>, tensor<8xi32>) -> tensor<*xf32>
// CHECK:  return [[BCT]] : tensor<*xf32>
}

func.func @broadcast_to_with_unknown_shape_and_output(%arg0: tensor<1x2x3x4x5x6xf32>, %arg1: tensor<*xi32>) -> tensor<*xf32> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<1x2x3x4x5x6xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>

// CHECK-LABEL: broadcast_to_with_unknown_shape_and_output
// CHECK:  "tf.BroadcastTo"(%arg0, %arg1)
}

// CHECK-LABEL: xla_conv_v2
func.func @xla_conv_v2(%arg0: tensor<4x8x8x16xf32>) -> tensor<4x8x8x16xf32> {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<3x3x16x16xf32>} : () -> tensor<3x3x16x16xf32> loc("Const_1")
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("XlaConv/feature_group_count")
  %2 = "tf.Const"() {value = dense<1> : tensor<2x2xi32>} : () -> tensor<2x2xi32> loc("XlaConv/padding")
  %3 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32> loc("XlaConv/window_strides")
  %4 = "tf.XlaConvV2"(%arg0, %0, %3, %2, %3, %3, %1) {batch_group_count = 1 : i64, device = "", dimension_numbers = "\18\02 \032\02\00\01@\03P\03Z\02\01\02b\02\01\02", precision_config = ""} : (tensor<4x8x8x16xf32>, tensor<3x3x16x16xf32>, tensor<2xi32>, tensor<2x2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<i32>) -> tensor<4x8x8x16xf32>
  func.return %4 : tensor<4x8x8x16xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
  // CHECK-DAG: %[[CST0:.*]] = arith.constant dense<1.000000e+00> : tensor<16x3x3x16xf32>
  // CHECK: %[[RES:.*]] = "tfl.conv_2d"(%arg0, %[[CST0]], %[[CST]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<4x8x8x16xf32>, tensor<16x3x3x16xf32>, tensor<16xf32>) -> tensor<4x8x8x16xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: lower_rfft_to_rfft2d
func.func @lower_rfft_to_rfft2d(%input: tensor<10x20x30xf32>, %fft_len: tensor<1xi32>) -> tensor<10x20x30xcomplex<f64>> {
  %0 = "tf.RFFT"(%input, %fft_len) : (tensor<10x20x30xf32>, tensor<1xi32>) -> tensor<10x20x30xcomplex<f64>>
  func.return %0: tensor<10x20x30xcomplex<f64>>

// CHECK-DAG:  %[[CST:.*]] = arith.constant dense<-2> : tensor<i32>
// CHECK-DAG:  %[[CST0:.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK-DAG:  %[[CST1:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  %[[EXP:.*]] = "tf.ExpandDims"(%arg0, %[[CST]]) : (tensor<10x20x30xf32>, tensor<i32>) -> tensor<10x20x1x30xf32>
// CHECK:  %[[CON:.*]] = "tf.ConcatV2"(%[[CST0]], %arg1, %[[CST1]]) : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
// CHECK:  %[[RFF:.*]] = "tf.RFFT2D"(%[[EXP]], %[[CON]]) : (tensor<10x20x1x30xf32>, tensor<2xi32>) -> tensor<10x20x1x30xcomplex<f64>>
// CHECK:  %[[SQE:.*]] = "tf.Squeeze"(%[[RFF]]) <{squeeze_dims = [-2]}> : (tensor<10x20x1x30xcomplex<f64>>) -> tensor<10x20x30xcomplex<f64>>
}

// CHECK-LABEL: xla_gather_to_strided_slice
func.func @xla_gather_to_strided_slice(%arg0 : tensor<1x9x104x768xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<[1, 9, 23, 768]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.XlaGather"(%arg0, %0, %1) {device = "", dimension_numbers = "\0A\04\00\01\02\03\1A\01\02", indices_are_sorted = false} : (tensor<1x9x104x768xf32>, tensor<1xi32>, tensor<4xi32>) -> tensor<?x?x?x?xf32>
  func.return %2 : tensor<?x?x?x?xf32>

// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<4xi64>
// CHECK-DAG: %[[CST0:.*]] = arith.constant dense<[1, 9, 23, 768]> : tensor<4xi64>
// CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : tensor<4xi64>
// CHECK: %[[V0:.*]] = "tf.StridedSlice"(%arg0, %[[CST]], %[[CST0]], %[[CST1]]) <{begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}> : (tensor<1x9x104x768xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// CHECK: return %[[V0]] : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: DontMatchFusedBatchNormV3
func.func @DontMatchFusedBatchNormV3(%arg0 :tensor<?x576x1x1xf32>, %arg1 : tensor<576xf32>, %arg2 : tensor<576xf32>, %arg3 : tensor<576xf32>,%arg4 : tensor<576xf32>) -> (tensor<?x576x1x1xf32>) {
  %result:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {data_format = "NHWC", device = "", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = false} : (tensor<?x576x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<?x576x1x1xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<576xf32>, tensor<*xf32>)
  func.return %result : tensor<?x576x1x1xf32>
  // CHECK: "tf.FusedBatchNormV3"
}

// CHECK-LABEL: DoNotConvertConv2DWhenFilterTypeDimIsNotDecided
func.func @DoNotConvertConv2DWhenFilterTypeDimIsNotDecided(%arg0 : tensor<?x?x?x96xf32>, %arg1 : tensor<3x3x96x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x?x?x96xf32>, tensor<3x3x96x?xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
// CHECK: tf.Conv2D
}

// CHECK-LABEL: conv2d_f16
func.func @conv2d_f16(%arg0 : tensor<?x224x224x3xf16>, %arg1 : tensor<3x3x3x16xf16>) -> tensor<?x112x112x16xf16> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<?x224x224x3xf16>, tensor<3x3x3x16xf16>) -> tensor<?x112x112x16xf16>
  func.return %0 : tensor<?x112x112x16xf16>
  // CHECK: "tf.Conv2D"
}

// CHECK-LABEL: fused_batch_norm_v3_f16
func.func @fused_batch_norm_v3_f16(%arg0 : tensor<?x112x112x16xf16>, %arg1 : tensor<16xf32>, %arg2 : tensor<16xf32>, %arg3 : tensor<16xf32>, %arg4 : tensor<16xf32>) -> tensor<?x112x112x16xf16> {
  %0, %1, %2, %3, %4, %5 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {data_format = "NHWC", device = "", epsilon = 1.000000e-03 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<?x112x112x16xf16>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<?x112x112x16xf16>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<*xf32>)
  func.return %0 : tensor<?x112x112x16xf16>
  // CHECK: "tf.FusedBatchNormV3"
}

// CHECK-LABEL: depthwise_conv2d_native_f16
func.func @depthwise_conv2d_native_f16(%arg0 : tensor<?x112x112x16xf16>, %arg1 : tensor<3x3x16x1xf16>) -> tensor<?x112x112x16xf16> {
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<?x112x112x16xf16>, tensor<3x3x16x1xf16>) -> tensor<?x112x112x16xf16>
  func.return %0 : tensor<?x112x112x16xf16>
  // CHECK: "tf.DepthwiseConv2dNative"
}

// CHECK-LABEL: conv_2d_bf16
func.func @conv_2d_bf16(%arg0 : tensor<256x32x32x3xbf16>, %arg1 : tensor<3x3x3x16xbf16>) -> tensor<256x8x7x16xbf16> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xbf16>, tensor<3x3x3x16xbf16>) -> tensor<256x8x7x16xbf16>
  func.return %0 : tensor<256x8x7x16xbf16>
  // CHECK: "tf.Conv2D"
}

// CHECK-LABEL: fused_batch_norm_v3_bf16
func.func @fused_batch_norm_v3_bf16(%arg0 : tensor<?x112x112x16xbf16>, %arg1 : tensor<16xf32>, %arg2 : tensor<16xf32>, %arg3 : tensor<16xf32>, %arg4 : tensor<16xf32>) -> tensor<?x112x112x16xbf16> {
  %0, %1, %2, %3, %4, %5 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {data_format = "NHWC", device = "", epsilon = 1.000000e-03 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<?x112x112x16xbf16>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<?x112x112x16xbf16>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<*xf32>)
  func.return %0 : tensor<?x112x112x16xbf16>
  // CHECK: "tf.FusedBatchNormV3"
}

// CHECK-LABEL: depthwise_conv_2d_bf16
func.func @depthwise_conv_2d_bf16(%arg0 : tensor<256x32x32x3xbf16>, %arg1 : tensor<3x3x3x4xf32>, %arg2 : tensor<256x3x32x32xf32>) -> tensor<256x30x30x12xbf16> {
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xbf16>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xbf16>
  func.return %0 : tensor<256x30x30x12xbf16>
  // CHECK: "tf.DepthwiseConv2dNative"
}

// CHECK-LABEL: strided_slice_unranked_input
func.func @strided_slice_unranked_input(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %18 = "tf.Const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %57 = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %534 = "tf.StridedSlice"(%arg0, %57, %57, %18) {begin_mask = 11 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 11 : i64, new_axis_mask = 4 : i64, shrink_axis_mask = 0 : i64} : (tensor<*xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<*xf32>
  func.return %534 : tensor<*xf32>
  // CHECK: "tf.StridedSlice"
}

func.func @fused_batch_norm_v3_training(%arg0 : tensor<1x1x6x2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>) -> tensor<1x1x6x2xf32> {
  %0, %1, %2, %3, %4, %5 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {data_format = "NHWC", epsilon = 1.000000e-03 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = true} : (tensor<1x1x6x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x1x6x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<*xf32>)
  func.return %0 : tensor<1x1x6x2xf32>
  // CHECK-LABEL: fused_batch_norm_v3_training
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1.000000e-03> : tensor<f32>
  // CHECK:  %[[MEAN:.*]] = "tf.Mean"(%arg0, %[[CST]]) <{keep_dims = false}> : (tensor<1x1x6x2xf32>, tensor<3xi32>) -> tensor<2xf32>
  // CHECK:  %[[SQ:.*]] = "tf.SquaredDifference"(%arg0, %[[MEAN]]) : (tensor<1x1x6x2xf32>, tensor<2xf32>) -> tensor<1x1x6x2xf32>
  // CHECK:  %[[MEAN0:.*]] = "tf.Mean"(%[[SQ]], %[[CST]]) <{keep_dims = false}> : (tensor<1x1x6x2xf32>, tensor<3xi32>) -> tensor<2xf32>
  // CHECK:  %[[ADD:.*]] = "tf.Add"(%[[MEAN0]], %[[CST1]]) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK:  %[[RSQRT:.*]] = "tf.Rsqrt"(%[[ADD]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK:  %[[MUL1:.*]] = "tf.Mul"(%arg1, %[[RSQRT]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK:  %[[MUL2:.*]] = "tf.Mul"(%arg0, %[[MUL1]]) : (tensor<1x1x6x2xf32>, tensor<2xf32>) -> tensor<1x1x6x2xf32>
  // CHECK:  %[[MUL3:.*]] = "tf.Mul"(%[[MEAN]], %[[MUL1]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK:  %[[SUB:.*]] = "tf.Sub"(%arg2, %[[MUL3]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK:  %[[ADD0:.*]] = "tf.Add"(%[[MUL2]], %[[SUB]]) : (tensor<1x1x6x2xf32>, tensor<2xf32>) -> tensor<1x1x6x2xf32>
  // CHECK:  return %[[ADD0]] : tensor<1x1x6x2xf32>
}

func.func @scatter_nd_add(%arg0: tensor<7xi64>, %arg1: tensor<1x1xi32>, %arg2: tensor<1xi64>) -> tensor<7xi64> {
  %0 = "tf.TensorScatterAdd"(%arg0, %arg1, %arg2) : (tensor<7xi64>, tensor<1x1xi32>, tensor<1xi64>) -> tensor<7xi64>
  func.return %0 : tensor<7xi64>

  // CHECK-LABEL: scatter_nd_add
  // CHECK:  %[[GATHER:.*]] = "tf.GatherNd"(%arg0, %arg1) <{bad_indices_policy = ""}> : (tensor<7xi64>, tensor<1x1xi32>) -> tensor<1xi64>
  // CHECK:  %[[ADD:.*]] = "tf.Add"(%arg2, %[[GATHER]]) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  // CHECK:  %[[SCATTER:.*]] = "tf.TensorScatterUpdate"(%arg0, %arg1, %[[ADD]]) : (tensor<7xi64>, tensor<1x1xi32>, tensor<1xi64>) -> tensor<7xi64>
  // CHECK:  return %[[SCATTER]] : tensor<7xi64>
}

func.func @add_v2_uint32(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>

  // CHECK-LABEL: add_v2_uint32
  // CHECK:  %[[CAST:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<ui32>) -> tensor<i32>
  // CHECK:  %[[CAST1:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<ui32>) -> tensor<i32>
  // CHECK:  %[[ADD:.*]] = "tf.AddV2"(%[[CAST]], %[[CAST1]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK:  %[[CAST2:.*]] = "tf.Cast"(%[[ADD]]) <{Truncate = false}> : (tensor<i32>) -> tensor<ui32>
  // CHECK:  return %[[CAST2]] : tensor<ui32>
}

func.func @QuantDequantTranspose(%arg0: tensor<2x3xf32>) -> (tensor<2x4xf32>) {
  %cst = "tf.Const"() {value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> tensor<3x4xf32>
  %cst_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Abs"(%cst) {device = ""} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %1 = "tf.Max"(%0, %cst_0) {device = "", keep_dims = false} : (tensor<3x4xf32>, tensor<i32>) -> tensor<4xf32>
  %2 = "tf.Neg"(%1) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "tfl.custom_tf"(%cst, %2, %1) ({
  ^bb0(%arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %7 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg1, %arg2, %arg3) {device = "", narrow_range = false, num_bits = 8 : i64} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tfl.yield"(%7) : (tensor<*xf32>) -> ()
  }) {device = "", narrow_range = false, num_bits = 8 : i64} : (tensor<3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  %4 = "tf.MatMul"(%arg0, %3) {device = "", transpose_a = false, transpose_b = false} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %6 = "tf.Identity"(%5) {device = ""} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %6 : tensor<2x4xf32>

  // CHECK-LABEL: QuantDequantTranspose
  // CHECK-DAG: %[[CST:.*]] = "tf.Const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<?xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<1.00392163> : tensor<3x4xf32>
  // CHECK: %[[QUANT:.*]] = "tfl.quantize"(%[[CST_0]]) <{qtype = tensor<3x4x!quant.uniform<u8:f32:1, {0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128}>>}> : (tensor<3x4xf32>) -> tensor<3x4x!quant.uniform<u8:f32:1, {0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128}>>
  // CHECK: %[[DEQUANT:.*]] = "tfl.dequantize"(%[[QUANT]]) : (tensor<3x4x!quant.uniform<u8:f32:1, {0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128,0.0078431372549019607:128}>>) -> tensor<3x4xf32>
  // CHECK: %[[TRANSPOSE:.*]] = "tf.Transpose"(%[[DEQUANT]], %[[CST]]) : (tensor<3x4xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %[[TRANSPOSE]]) <{grad_a = false, grad_b = false, transpose_a = false, transpose_b = true}> : (tensor<2x3xf32>, tensor<*xf32>) -> tensor<2x4xf32>
  // CHECK: return %[[MATMUL]] : tensor<2x4xf32>
}

func.func @GroupConv(%arg0: tensor<?x1x26x14xf32>, %arg1: tensor<1x3x2x14xf32>) -> (tensor<?x1x6x14xf32>) {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 5, 1], use_cudnn_on_gpu = true} : (tensor<?x1x26x14xf32>, tensor<1x3x2x14xf32>) -> tensor<?x1x6x14xf32>
  func.return %0 : tensor<?x1x6x14xf32>
  // CHECK-LABEL: GroupConv
  // CHECK-DAG:  %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<14xf32>
  // CHECK-DAG:  %[[CONSTANT0:.*]] = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  // CHECK:  %0 = "tf.Transpose"(%arg1, %[[CONSTANT0]]) : (tensor<1x3x2x14xf32>, tensor<4xi32>) -> tensor<14x1x3x2xf32>
  // CHECK:  %1 = "tfl.conv_2d"(%arg0, %0, %[[CONSTANT]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 5 : i32}> : (tensor<?x1x26x14xf32>, tensor<14x1x3x2xf32>, tensor<14xf32>) -> tensor<?x1x6x14xf32>
}

func.func @UnsupportedGroupConv_UnrankedTensorType(%arg0: tensor<*xf32>, %arg1: tensor<1x3x2x14xf32>) -> (tensor<?x1x6x14xf32>) {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 5, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<1x3x2x14xf32>) -> tensor<?x1x6x14xf32>
  func.return %0 : tensor<?x1x6x14xf32>

  // CHECK-LABEL: UnsupportedGroupConv_UnrankedTensorType
  // CHECK-NOT: "tfl.conv_2d"
  // CHECK: "tf.Conv2D"
}

func.func @UnsupportedGroupConv_DynamicDimAtInputDimThree(%arg0: tensor<?x1x26x?xf32>, %arg1: tensor<1x3x2x14xf32>) -> (tensor<?x1x6x14xf32>) {
  %0 = "tf.Conv2D"(%arg0, %arg1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 5, 1], use_cudnn_on_gpu = true} : (tensor<?x1x26x?xf32>, tensor<1x3x2x14xf32>) -> tensor<?x1x6x14xf32>
  func.return %0 : tensor<?x1x6x14xf32>

  // CHECK-LABEL: UnsupportedGroupConv_DynamicDimAtInputDimThree
  // CHECK-NOT: "tfl.conv_2d"
  // CHECK: "tf.Conv2D"
}

func.func @RedundantShapeOp(%shape: tensor<?xi64>, %fill: tensor<f32>) -> (tensor<?xi64>) {
  %0 = "tf.Fill"(%shape, %fill) : (tensor<?xi64>, tensor<f32>) -> (tensor<*xf32>)
  %1 = "tf.Shape"(%0) : (tensor<*xf32>) -> (tensor<?xi64>)
  func.return %1 : tensor<?xi64>

  // CHECK-LABEL: RedundantShapeOp
  // CHECK-NOT: "tf.Shape"
}

// CHECK-LABEL: @MoveTransposeAcrossPerChannelQuant
func.func @MoveTransposeAcrossPerChannelQuant(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x112x112x6xf32> {
  %cst = "tf.Const"() <{value = dense<6.0> : tensor<6x3x7x7xf32>}> : () -> tensor<6x3x7x7xf32>
  %cst_14 = "tf.Const"() <{value = dense<[2, 3, 1, 0]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %126 = "tfl.quantize"(%cst) {qtype = tensor<6x3x7x7x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>} : (tensor<6x3x7x7xf32>) -> tensor<6x3x7x7x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>
  %127 = "tfl.dequantize"(%126) : (tensor<6x3x7x7x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>) -> tensor<6x3x7x7xf32>
  %129 = "tf.Transpose"(%127, %cst_14) : (tensor<6x3x7x7xf32>, tensor<4xi64>) -> tensor<7x7x3x6xf32>
  %130 = "tf.Conv2D"(%arg0, %129) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [0, 0, 3, 3, 3, 3, 0, 0], padding = "EXPLICIT", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true}> : (tensor<1x224x224x3xf32>, tensor<7x7x3x6xf32>) -> tensor<1x112x112x6xf32>
  return %130 : tensor<1x112x112x6xf32>
  // CHECK: %cst = arith.constant dense<6.000000e+00> : tensor<6x7x7x3xf32>
  // CHECK: %cst_0 = arith.constant dense<0.000000e+00> : tensor<6xf32>
  // CHECK: %cst_1 = arith.constant dense<{{\[\[}}0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>
  // CHECK: %0 = "tf.Pad"(%arg0, %cst_1) : (tensor<1x224x224x3xf32>, tensor<4x2xi32>) -> tensor<*xf32>
  // CHECK: %1 = "tfl.quantize"(%cst) <{qtype = tensor<6x7x7x3x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>}> : (tensor<6x7x7x3xf32>) -> tensor<6x7x7x3x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>
  // CHECK: %2 = "tfl.dequantize"(%1) : (tensor<6x7x7x3x!quant.uniform<i8<-127:127>:f32:0, {1.412750e-03,3.503970e-04,2.441410e-04,3.823330e-04,2.441410e-04,8.950800e-04}>>) -> tensor<6x7x7x3xf32>
  // CHECK: %3 = "tfl.conv_2d"(%0, %2, %cst_0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<*xf32>, tensor<6x7x7x3xf32>, tensor<6xf32>) -> tensor<1x112x112x6xf32>
  // CHECK: return %3 : tensor<1x112x112x6xf32>
}

// CHECK-LABEL: @FoldDoubleTranspose
func.func @FoldDoubleTranspose(%arg0: tensor<1x4x1440x256xf32>) -> tensor<1x1440x256x4xf32> {
    %cst_12 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %cst_18 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %2112 = "tf.Transpose"(%arg0, %cst_18) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x4x256xf32>
    %2114 = "tf.Transpose"(%2112, %cst_12) : (tensor<1x1440x4x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
    return %2114 : tensor<1x1440x256x4xf32>
  // CHECK-DAG: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %0 = "tf.Transpose"(%arg0, %cst) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
  // CHECK: return %0
}

// CHECK-LABEL: @FoldMultpleTranspose
func.func @FoldMultpleTranspose(%arg0: tensor<1x4x1440x256xf32>) -> tensor<1x256x4x1440xf32> {
    %cst_11 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %cst_12 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %cst_18 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %2112 = "tf.Transpose"(%arg0, %cst_11) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
    %2113 = "tf.Transpose"(%2112, %cst_18) : (tensor<1x1440x256x4xf32>, tensor<4xi32>) -> tensor<1x256x1440x4xf32>
    %2114 = "tf.Transpose"(%2113, %cst_12) : (tensor<1x256x1440x4xf32>, tensor<4xi32>) -> tensor<1x256x4x1440xf32>
    return %2114 : tensor<1x256x4x1440xf32>
  // CHECK-DAG: %cst = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  // CHECK: %0 = "tf.Transpose"(%arg0, %cst) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x256x4x1440xf32>
  // CHECK: return %0
}

// CHECK-LABEL @FoldTrivialReshapeIntoTranspose
func.func @FoldTrivialReshapeIntoTranspose(%arg: tensor<2x1x3x3xf32>) -> tensor<1x3x3x2xf32> {
  %cst = arith.constant dense<[1, 3, 3, 2]> : tensor<4xi32>
  %cst_2 = arith.constant dense<[2, 3, 0, 1]> : tensor<4xi32>
  %2 = "tf.Transpose"(%arg, %cst_2) : (tensor<2x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x2x1xf32>
  %3 = "tf.Reshape"(%2, %cst) : (tensor<3x3x2x1xf32>, tensor<4xi32>) -> tensor<1x3x3x2xf32>
  return %3: tensor<1x3x3x2xf32>
  // CHECK:  %cst = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi32>
  // CHECK:  %0 = "tf.Transpose"(%arg0, %cst) : (tensor<2x1x3x3xf32>, tensor<4xi32>) -> tensor<1x3x3x2xf32>
  // CHECK:  return %0 : tensor<1x3x3x2xf32>
}

// CHECK-LABEL: @MoveTransposeAcrossDepthwiseConvPerChannelQuant
func.func @MoveTransposeAcrossDepthwiseConvPerChannelQuant(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %cst_1 = arith.constant dense<6.000000e+00> : tensor<2x1x3x3xf32>
  %0 = "tfl.quantize"(%cst_1) {qtype = tensor<2x1x3x3x!quant.uniform<i8<-127:127>:f32:0, {6.587140e-03,1.888450e-02}>>} : (tensor<2x1x3x3xf32>) -> tensor<2x1x3x3x!quant.uniform<i8<-127:127>:f32:0, {6.587140e-03,1.888450e-02}>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x3x3x!quant.uniform<i8<-127:127>:f32:0, {6.587140e-03,1.888450e-02}>>) -> tensor<2x1x3x3xf32>
  %2 = "tf.Transpose"(%1, %cst) : (tensor<2x1x3x3xf32>, tensor<4xi32>) -> tensor<1x3x3x2xf32>
  %3 = "tfl.depthwise_conv_2d"(%arg0, %2, %cst_0) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  return %3 : tensor<1x112x112x2xf32>
  // CHECK: %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
  // CHECK: %cst_0 = arith.constant dense<6.000000e+00> : tensor<1x3x3x2xf32>
  // CHECK: %0 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {6.587140e-03,1.888450e-02}>>}> : (tensor<1x3x3x2xf32>) -> tensor<1x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {6.587140e-03,1.888450e-02}>>
  // CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {6.587140e-03,1.888450e-02}>>) -> tensor<1x3x3x2xf32>
  // CHECK: %2 = "tfl.depthwise_conv_2d"(%arg0, %1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %2 : tensor<1x112x112x2xf32>
}

}
