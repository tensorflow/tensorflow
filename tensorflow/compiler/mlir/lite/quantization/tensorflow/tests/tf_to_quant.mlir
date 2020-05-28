// RUN: tf-opt -tf-to-quant %s | FileCheck %s

// CHECK-LABEL: fakeQuantPerChannelForActivation
func @fakeQuantPerChannelForActivation(%arg0: tensor<8x3xf32>) -> (tensor<8x3xf32>) {
  %arg1 = constant dense<[0.0, -1.0, 1.0]> : tensor<3xf32>
  %arg2 = constant dense<[255.0, 254.0, 256.0]> : tensor<3xf32>
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<8x3xf32>
  return %0 : tensor<8x3xf32>

// CHECK:  %[[fq:.*]] = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %cst, %cst_0)
// CHECK:  %[[q:.*]] = "quant.qcast"(%[[fq]]) : (tensor<8x3xf32>) -> tensor<8x3x!quant.uniform<i8:f32:1, {1.000000e+00:-128,1.000000e+00:-127,1.000000e+00:-128}>>
// CHECK:  %[[dq:.*]] = "quant.dcast"(%[[q]])
// CHECK:  return %[[dq]]
}

// CHECK-LABEL: fakeQuantForActivation
func @fakeQuantForActivation(tensor<8xf32>) -> (tensor<8xf32>) {
^bb0(%arg0: tensor<8xf32>):
  %arg1 = constant dense<0.0> : tensor<f32>
  %arg2 = constant dense<255.0> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>

// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0)
// CHECK:  %1 = "quant.qcast"(%0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK:  %2 = "quant.dcast"(%1)
// CHECK:  return %2
}

// CHECK-LABEL: fakeQuantForActivationNoDuplication
func @fakeQuantForActivationNoDuplication(tensor<8xf32>) -> (tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>) {
^bb0(%arg0: tensor<8xf32>):
  %arg1 = constant dense<0.0> : tensor<f32>
  %arg2 = constant dense<255.0> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  %1 = "quant.qcast"(%0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>
  return %1 : tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>

// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0) {narrow_range = false, num_bits = 3 : i64}
// CHECK:  %1 = "quant.qcast"(%0) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK:  return %1
}

// CHECK-LABEL: fakeQuantFolded
func @fakeQuantFolded() -> (tensor<8xf32>) {
  %in = constant dense<0.0> : tensor<8xf32>
  %min = constant dense<0.0> : tensor<f32>
  %max = constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %rst = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 3, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  return %rst : tensor<8xf32>

// CHECK: %[[CONSTANT:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<8xf32>}
// CHECK: %[[QUANTIZE:.*]] = "quant.qcast"(%[[CONSTANT]]) : (tensor<8xf32>) -> tensor<8x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[DEQUANTIZE:.*]] = "quant.dcast"(%[[QUANTIZE]])
// CHECK: return %[[DEQUANTIZE]] : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantNotFolded
func @fakeQuantNotFolded(tensor<8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8xf32>) {
^bb0(%arg0: tensor<8xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
  %1 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg3, %arg4) {num_bits = 3, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>

// CHECK: %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2)
// CHECK: return %0 : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantWithConv2D
func @fakeQuantWithConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = constant dense<0.0> : tensor<f32>
  %max = constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %fq = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.Conv2D"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %rst : tensor<256x30x30x16xf32>

// CHECK: %[[CONSTANT0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3x3x3x16xf32>}
// CHECK: %[[QUANTIZE:.*]] = "quant.qcast"(%[[CONSTANT0]]) : (tensor<3x3x3x16xf32>) -> tensor<3x3x3x16x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[DEQUANTIZE:.*]] = "quant.dcast"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tf.Conv2D"(%arg0, %[[DEQUANTIZE]])
// CHECK: return %[[CONV]]
}

// CHECK-LABEL: perChannelFakeQuantWithConv2D
func @perChannelFakeQuantWithConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = constant dense<0.0> : tensor<16xf32>
  %max = constant dense<255.0> : tensor<16xf32>
  %mini = "tf.Identity"(%min) : (tensor<16xf32>) -> tensor<16xf32>
  %maxi = "tf.Identity"(%max) : (tensor<16xf32>) -> tensor<16xf32>
  %fq = "tf.FakeQuantWithMinMaxVarsPerChannel"(%in, %mini, %maxi) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.Conv2D"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %rst : tensor<256x30x30x16xf32>

// CHECK: %[[CONSTANT0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3x3x3x16xf32>}
// CHECK: %[[QUANTIZE:.*]] = "quant.qcast"(%[[CONSTANT0]]) : (tensor<3x3x3x16xf32>) -> tensor<3x3x3x16x!quant.uniform<i8:f32:3,
// CHECK-SAME: {1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,
// CHECK-SAME: 1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128}>>
// CHECK: %[[DEQUANTIZE:.*]] = "quant.dcast"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tf.Conv2D"(%arg0, %[[DEQUANTIZE]])
// CHECK: return %[[CONV]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: fakeQuantWithDepthwiseConv2D
func @fakeQuantWithDepthwiseConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = constant dense<0.0> : tensor<f32>
  %max = constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %fq = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.DepthwiseConv2dNative"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %rst : tensor<256x30x30x16xf32>

// CHECK: %[[CONSTANT0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3x3x3x16xf32>}
// CHECK: %[[QUANTIZE:.*]] = "quant.qcast"(%[[CONSTANT0]]) : (tensor<3x3x3x16xf32>) -> tensor<3x3x3x16x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[DEQUANTIZE:.*]] = "quant.dcast"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[DEQUANTIZE]])
// CHECK: return %[[CONV]]
}

// CHECK-LABEL: perChannelFakeQuantWithDepthwiseConv2D
func @perChannelFakeQuantWithDepthwiseConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = constant dense<0.0> : tensor<16xf32>
  %max = constant dense<255.0> : tensor<16xf32>
  %mini = "tf.Identity"(%min) : (tensor<16xf32>) -> tensor<16xf32>
  %maxi = "tf.Identity"(%max) : (tensor<16xf32>) -> tensor<16xf32>
  %fq = "tf.FakeQuantWithMinMaxVarsPerChannel"(%in, %mini, %maxi) {num_bits = 3, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.DepthwiseConv2dNative"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %rst : tensor<256x30x30x16xf32>

// CHECK: %[[CONSTANT0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<3x3x3x16xf32>}
// CHECK: %[[QUANTIZE:.*]] = "quant.qcast"(%[[CONSTANT0]]) : (tensor<3x3x3x16xf32>) -> tensor<3x3x3x16x!quant.uniform<i8:f32:3,
// CHECK-SAME: {1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,
// CHECK-SAME: 1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128,1.000000e+00:-128}>>
// CHECK: %[[DEQUANTIZE:.*]] = "quant.dcast"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[DEQUANTIZE]])
// CHECK: return %[[CONV]]
}
