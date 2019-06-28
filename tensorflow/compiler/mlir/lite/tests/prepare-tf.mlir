// RUN: tf-opt -tfl-prepare-tf %s | FileCheck %s

func @conv(tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> (tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>) {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) :
   // OK
   %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
   // Unsupported data format
   %1 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
   // OK
   %2 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC",                           padding = "VALID", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
   // Unsupported padding
   %3 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "EXPLICIT", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
   // Unsupported strides
   %4 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [2, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>

  return %0, %1, %2, %3, %4 : tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>

// CHECK-LABEL: conv
// CHECK:  %cst = constant dense<0.000000e+00> : tensor<16xf32>
// CHECK:  %cst_0 = constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK:  %0 = "tf.Transpose"(%arg1, %cst_0) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %1 = "tfl.conv_2d"(%arg0, %0, %cst) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK:  %2 = "tf.Conv2D"
// CHECK:  %3 = "tf.Transpose"(%arg1, %cst_0) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %4 = "tfl.conv_2d"(%arg0, %3, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK:  %5 = "tf.Conv2D"
// CHECK:  %6 = "tf.Conv2D"
}

func @depthwiseConv2D(tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> (tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>) {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x4xf32>) :
   // OK
   %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
   // Unsupported data format
   %1 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
   // OK
   %2 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC",                           padding = "VALID", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
   // Unsupported strides
   %3 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [2, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>

  return %0, %1, %2, %3 : tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>, tensor<256x30x30x12xf32>

// CHECK-LABEL: depthwiseConv2D
// CHECK:  %cst = constant dense<0.000000e+00> : tensor<12xf32>
// CHECK:  %cst_0 = constant dense<[1, 3, 3, 12]> : tensor<4xi64>
// CHECK:  %0 = "tf.Reshape"(%arg1, %cst_0) : (tensor<3x3x3x4xf32>, tensor<4xi64>) -> tensor<1x3x3x12xf32>
// CHECK:  %1 = "tfl.depthwise_conv_2d"(%arg0, %0, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<1x3x3x12xf32>, tensor<12xf32>) -> tensor<256x30x30x12xf32>
// CHECK:  %2 = "tf.DepthwiseConv2dNative"
// CHECK:  %3 = "tf.Reshape"(%arg1, %cst_0) : (tensor<3x3x3x4xf32>, tensor<4xi64>) -> tensor<1x3x3x12xf32>
// CHECK:  %4 = "tfl.depthwise_conv_2d"(%arg0, %3, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<1x3x3x12xf32>, tensor<12xf32>) -> tensor<256x30x30x12xf32>
// CHECK:  %5 = "tf.DepthwiseConv2dNative"
}

func @fusedBatchNorm(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>) {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // OK
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // Unsupported training
  %1:5 = "tf.FusedBatchNorm"( %0#0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true}  : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  // Use other output
  %2:5 = "tf.FusedBatchNorm"( %1#0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)

  return %2, %2#1 : tensor<8x8x8x8xf32>, tensor<8xf32>

// CHECK-LABEL: fusedBatchNorm
// CHECK:%cst = constant dense<1.000000e-03> : tensor<f32>
//              variance + epsilon
// CHECK:  %0 = "tf.Add"(%arg4, %cst) : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
//              rsqrt(variance + epsilon)
// CHECK:  %1 = "tf.Rsqrt"(%0) : (tensor<8xf32>) -> tensor<8xf32>
//              scale * rsqrt(variance + epsilon)
// CHECK:  %2 = "tf.Mul"(%arg1, %1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
//              x * scale * rsqrt(variance + epsilon)
// CHECK:  %3 = "tf.Mul"(%arg0, %2) : (tensor<8x8x8x8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
//              mean * scale * rsqrt(variance + epsilon)
// CHECK:  %4 = "tf.Mul"(%arg3, %2) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
//              offset - mean * scale * rsqrt(variance + epsilon)
// CHECK:  %5 = "tf.Sub"(%arg2, %4) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
//              x * scale * rsqrt(variance + epsilon) +
//              offset - mean * scale * rsqrt(variance + epsilon)
// CHECK:  %6 = "tf.Add"(%3, %5) : (tensor<8x8x8x8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>

// CHECK:  %7:5 = "tf.FusedBatchNorm"(%6, %arg1, %arg2, %arg3, %arg4)
// CHECK:  %8:5 = "tf.FusedBatchNorm"(%7#0, %arg1, %arg2, %arg3, %arg4)
}

func @fakeQuantNotFollowedByQuant(tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>) {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  %arg1 = constant dense<-0.1> : tensor<f32>
  %arg2 = constant dense<0.2> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>

// CHECK-LABEL: fakeQuantNotFollowedByQuant
// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  %1 = "tfl.quantize"(%0) {qtype = tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>}
// CHECK:  %2 = "tfl.dequantize"(%1) : (tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>)
// CHECK:  return %2 : tensor<8x8x8x8xf32>
}

func @fakeQuantFollowedByQuant(tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>) {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  %arg1 = constant dense<-0.1> : tensor<f32>
  %arg2 = constant dense<0.2> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>
  %2 = "tfl.dequantize"(%1) : (tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>) -> tensor<8x8x8x8xf32>
  return %2 : tensor<8x8x8x8xf32>

// CHECK-LABEL: fakeQuantFollowedByQuant
// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  %1 = "tfl.quantize"(%0) {qtype = tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>}
// CHECK:  %2 = "tfl.dequantize"(%1) : (tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>)
// CHECK:  return %2 : tensor<8x8x8x8xf32>
}

func @fakeQuantVarsNotConst(tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8x8x8x8xf32>) {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
  %1 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg3, %arg4) {num_bits = 3, narrow_range = false} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %1 : tensor<8x8x8x8xf32>

// CHECK-LABEL: fakeQuantVarsNotConst
// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  return %0 : tensor<8x8x8x8xf32>
}

func @fakeQuantFollowedByTranspose(tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> (tensor<16x3x3x3xf32>) {
^bb0(%arg0: tensor<3x3x3x16xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x16xf32>
  %1 = "tf.Transpose"(%0, %cst_0): (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
  return %1 : tensor<16x3x3x3xf32>

// CHECK-LABEL: fakeQuantFollowedByTranspose
// CHECK:  %cst = constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK:  %0 = "tf.Transpose"(%arg0, %cst) : (tensor<3x3x3x16xf32>, tensor<4xi32>) -> tensor<16x3x3x3xf32>
// CHECK:  %1 = "tf.FakeQuantWithMinMaxVars"(%0, %arg1, %arg2) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  return %1 : tensor<16x3x3x3xf32>
}

func @fakeQuantFollowedByReshape(tensor<3x3x3x4xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x3x3x12xf32>) {
^bb0(%arg0: tensor<3x3x3x4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = constant dense<[1, 3, 3, 12]> : tensor<4xi64>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x4xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x4xf32>
  %1 = "tf.Reshape"(%0, %cst_0) : (tensor<3x3x3x4xf32>, tensor<4xi64>) -> tensor<1x3x3x12xf32>
  return %1 : tensor<1x3x3x12xf32>

// CHECK-LABEL: fakeQuantFollowedByReshape
// CHECK:  %cst = constant dense<[1, 3, 3, 12]> : tensor<4xi64>
// CHECK:  %0 = "tf.Reshape"(%arg0, %cst) : (tensor<3x3x3x4xf32>, tensor<4xi64>) -> tensor<1x3x3x12xf32>
// CHECK:  %1 = "tf.FakeQuantWithMinMaxVars"(%0, %arg1, %arg2) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  return %1 : tensor<1x3x3x12xf32>
}

func @identity(tensor<10xi32>) -> tensor<10xi32> {
^bb0(%arg0: tensor<10xi32>):
  %0 = "tf.Identity"(%arg0) : (tensor<10xi32>) -> tensor<10xi32>
  return %0: tensor<10xi32>

// CHECK-LABEL: identity
// CHECK: return %arg0
}


func @matmulNoTransposeAOrB(%arg0: tensor<1x1280xf32>, %arg1: tensor<1280x1000xf32>) -> tensor<1x1000xf32> {
  %166 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", _output_shapes = ["tfshape$dim { size = 1} dim { size = 1000}"], device = "", name = "matmul", transpose_a = false, transpose_b = false} : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
  return %166 : tensor<1x1000xf32>

  // CHECK-LABEL: matmulNoTransposeAOrB
  // CHECK: %cst = constant dense<0> : tensor<i32>
  // CHECK: %cst_0 = constant dense<-1> : tensor<i32>
  // CHECK: %cst_1 = constant dense<1> : tensor<i32>
  // CHECK: %0 = "tf.Rank"(%arg1) : (tensor<1280x1000xf32>) -> tensor<i32>
  // CHECK: %1 = "tf.Range"(%0, %cst, %cst_0) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %2 = "tf.Sub"(%1, %cst_1) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %3 = "tf.Transpose"(%arg1, %2) : (tensor<1280x1000xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %4 = "tf.MatMul"(%arg0, %3) {transpose_a = false, transpose_b = true} : (tensor<1x1280xf32>, tensor<*xf32>) -> tensor<1x1000xf32>
 }

func @matmulNoTransposeB(%arg0: tensor<1x1280xf32>, %arg1: tensor<1280x1000xf32>) -> tensor<1x1000xf32> {
  %166 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", _output_shapes = ["tfshape$dim { size = 1} dim { size = 1000}"], device = "", name = "matmul", transpose_a = true, transpose_b = false} : (tensor<1x1280xf32>, tensor<1280x1000xf32>) -> tensor<1x1000xf32>
  return %166 : tensor<1x1000xf32>

  // CHECK-LABEL: matmulNoTransposeB
  // CHECK: %cst = constant dense<0> : tensor<i32>
  // CHECK: %cst_0 = constant dense<-1> : tensor<i32>
  // CHECK: %cst_1 = constant dense<1> : tensor<i32>
  // CHECK: %0 = "tf.Rank"(%arg0) : (tensor<1x1280xf32>) -> tensor<i32>
  // CHECK: %1 = "tf.Range"(%0, %cst, %cst_0) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %2 = "tf.Sub"(%1, %cst_1) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %3 = "tf.Transpose"(%arg0, %2) : (tensor<1x1280xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %4 = "tf.Rank"(%arg1) : (tensor<1280x1000xf32>) -> tensor<i32>
  // CHECK: %5 = "tf.Range"(%4, %cst, %cst_0) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %6 = "tf.Sub"(%5, %cst_1) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  // CHECK: %7 = "tf.Transpose"(%arg1, %6) : (tensor<1280x1000xf32>, tensor<?xi32>) -> tensor<*xf32>
  // CHECK: %8 = "tf.MatMul"(%3, %7) {transpose_a = false, transpose_b = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<1x1000xf32>
}
