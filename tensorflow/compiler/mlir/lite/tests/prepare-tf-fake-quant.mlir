// RUN: tf-opt %s -tfl-raise-custom-ops="test-raise-tf-targets=tf.FakeQuantWithMinMaxVarsPerChannel,tf.FakeQuantWithMinMaxVars" -tfl-prepare-tf | FileCheck --dump-input=always %s
// RUN: tf-opt %s -tfl-raise-custom-ops="test-raise-tf-targets=tf.FakeQuantWithMinMaxVarsPerChannel,tf.FakeQuantWithMinMaxVars" -tfl-prepare-tf=use-fake-quant-num-bits=true | FileCheck --check-prefix LOBIT --dump-input=always %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: fakeQuantPerChannelForActivation
func.func @fakeQuantPerChannelForActivation(%arg0: tensor<8x4xf32>) -> (tensor<8x4xf32>) {
  %arg1 = arith.constant dense<[0.0, -1.0, 1.0, 0.0]> : tensor<4xf32>
  %arg2 = arith.constant dense<[255.0, 254.0, 256.0, 1.0e-9]> : tensor<4xf32>
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<8x4xf32>
  func.return %0 : tensor<8x4xf32>

// CHECK:  %[[fq:.*]] = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %cst, %cst_0)
// CHECK:  %[[q:.*]] = "tfl.quantize"(%[[fq]]) {qtype = tensor<8x4x!quant.uniform<u8:f32:1, {1.000000e+00,1.000000e+00:1,1.000000e+00,3.9215686274509805E-9:127}>>}
// CHECK:  %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK:  return %[[dq]]
}

// CHECK-LABEL: fakeQuantForActivation
func.func @fakeQuantForActivation(tensor<8xf32>) -> (tensor<8xf32>) {
^bb0(%arg0: tensor<8xf32>):
  %arg1 = arith.constant dense<0.0> : tensor<f32>
  %arg2 = arith.constant dense<255.0> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>

// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0)
// CHECK:  %1 = "tfl.quantize"(%0) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK:  %2 = "tfl.dequantize"(%1)
// CHECK:  return %2
}

// CHECK-LABEL: fakeQuantForActivationNoDuplication
func.func @fakeQuantForActivationNoDuplication(tensor<8xf32>) -> (tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>) {
^bb0(%arg0: tensor<8xf32>):
  %arg1 = arith.constant dense<0.0> : tensor<f32>
  %arg2 = arith.constant dense<255.0> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>} : (tensor<8xf32>) -> tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>
  func.return %1 : tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>

// CHECK:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0) <{narrow_range = false, num_bits = 5 : i64}>
// CHECK:  %1 = "tfl.quantize"(%0) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK:  return %1
}

// CHECK-LABEL: WrappedFakeQuantFolded
func.func @WrappedFakeQuantFolded() -> tensor<8xf32> {
  %in = arith.constant dense<0.0> : tensor<8xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %rst = "tfl.custom_tf"(%in, %mini, %maxi) ({
  ^bb0(%arg1: tensor<8xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = "tf.FakeQuantWithMinMaxVars"(%arg1, %arg2, %arg3) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
    "tfl.yield"(%2) : (tensor<8xf32>) -> ()
  }) {num_bits = 5, narrow_range = false} :  (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %rst : tensor<8xf32>

// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT]]) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: return %[[DEQUANTIZE]] : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantFolded
func.func @fakeQuantFolded() -> (tensor<8xf32>) {
  %in = arith.constant dense<0.0> : tensor<8xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %rst = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %rst : tensor<8xf32>

// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT]]) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: return %[[DEQUANTIZE]] : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantFoldedWithoutIdentity
func.func @fakeQuantFoldedWithoutIdentity() -> (tensor<8xf32>) {
  %in = arith.constant dense<0.0> : tensor<8xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %rst = "tf.FakeQuantWithMinMaxVars"(%in, %min, %max) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %rst : tensor<8xf32>

// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT]]) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: return %[[DEQUANTIZE]] : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantFoldedWithCast
func.func @fakeQuantFoldedWithCast() -> (tensor<8xf32>) {
  %in = arith.constant dense<0.0> : tensor<8xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %minc = "tf.Cast"(%mini) : (tensor<f32>) -> tensor<f32>
  %maxc = "tf.Cast"(%maxi) : (tensor<f32>) -> tensor<f32>
  %rst = "tf.FakeQuantWithMinMaxVars"(%in, %minc, %maxc) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %rst : tensor<8xf32>

// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT]]) {qtype = tensor<8x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: return %[[DEQUANTIZE]] : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantNotFolded
func.func @fakeQuantNotFolded(tensor<8xf32>, tensor<f32>, tensor<f32>) -> (tensor<8xf32>) {
^bb0(%arg0: tensor<8xf32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
  %1 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg3, %arg4) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %1 : tensor<8xf32>

// CHECK: %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2)
// CHECK: return %0 : tensor<8xf32>
}

// CHECK-LABEL: fakeQuantFollowedByTranspose
func.func @fakeQuantFollowedByTranspose(tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2xf32>
  %1 = "tf.Transpose"(%0, %cst_0): (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  func.return %1 : tensor<2x1xf32>

// CHECK:  %cst = arith.constant
// CHECK:  %0 = "tf.Transpose"(%arg0, %cst)
// CHECK:  %1 = "tf.FakeQuantWithMinMaxVars"(%0, %arg1, %arg2)
// CHECK:  return %1
}

// CHECK-LABEL: fakeQuantFollowedByTransposes
func.func @fakeQuantFollowedByTransposes(tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x1xf32>, tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2xf32>
  %1 = "tf.Transpose"(%0, %cst_0): (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  %2 = "tf.Transpose"(%0, %cst_0): (tensor<1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
  func.return %1, %2 : tensor<2x1xf32>, tensor<2x1xf32>

// CHECK:  %cst = arith.constant
// CHECK:  %[[FQ:.*]] = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2)
// CHECK:  %[[T1:.*]] = "tf.Transpose"(%[[FQ]], %cst)
// CHECK:  %[[T2:.*]] = "tf.Transpose"(%[[FQ]], %cst)
}

// CHECK-LABEL: fakeQuantFollowedByReshape
func.func @fakeQuantFollowedByReshape(tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = arith.constant dense<[2, -1]> : tensor<2xi64>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2xf32>
  %1 = "tf.Reshape"(%0, %cst_0) : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>
  func.return %1 : tensor<2x1xf32>

// CHECK:  %cst = arith.constant
// CHECK:  %0 = "tf.Reshape"(%arg0, %cst)
// CHECK-SAME: tensor<2x1xf32>
// CHECK:  %1 = "tf.FakeQuantWithMinMaxVars"(%0, %arg1, %arg2)
// CHECK:  return %1
}

// CHECK-LABEL: fakeQuantFollowedByReshapes
func.func @fakeQuantFollowedByReshapes(tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2x1xf32>, tensor<2x1xf32>) {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %cst_0 = arith.constant dense<[2, -1]> : tensor<2xi64>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<1x2xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2xf32>
  %1 = "tf.Reshape"(%0, %cst_0) : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>
  %2 = "tf.Reshape"(%0, %cst_0) : (tensor<1x2xf32>, tensor<2xi64>) -> tensor<2x1xf32>
  func.return %1, %2 : tensor<2x1xf32>, tensor<2x1xf32>

// CHECK:  %cst = arith.constant
// CHECK:  %[[FQ:.*]] = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2)
// CHECK:  %[[R1:.*]] = "tf.Reshape"(%[[FQ]], %cst)
// CHECK-SAME: tensor<2x1xf32>
// CHECK:  %[[R2:.*]] = "tf.Reshape"(%[[FQ]], %cst)
// CHECK-SAME: tensor<2x1xf32>
}

// CHECK-LABEL: fakeQuantWithConv2D
func.func @fakeQuantWithConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x8x7x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = arith.constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %fq = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.Conv2D"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
  func.return %rst : tensor<256x8x7x16xf32>

// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[CONSTANT0:.*]] = arith.constant dense<0.000000e+00> : tensor<16x3x3x3xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT0]]) {qtype = tensor<16x3x3x3x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tfl.conv_2d"(%arg0, %[[DEQUANTIZE]], %[[CONSTANT]])
// CHECK: return %[[CONV]]
}

// CHECK-LABEL: perChannelFakeQuantWithConv2D
func.func @perChannelFakeQuantWithConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x8x7x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = arith.constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = arith.constant dense<0.0> : tensor<16xf32>
  %max = arith.constant dense<255.0> : tensor<16xf32>
  %mini = "tf.Identity"(%min) : (tensor<16xf32>) -> tensor<16xf32>
  %maxi = "tf.Identity"(%max) : (tensor<16xf32>) -> tensor<16xf32>
  %fq = "tf.FakeQuantWithMinMaxVarsPerChannel"(%in, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.Conv2D"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x8x7x16xf32>
  func.return %rst : tensor<256x8x7x16xf32>

// CHECK-DAG: %[[CONSTANT0:.*]] = arith.constant dense<0.000000e+00> : tensor<16x3x3x3xf32>
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT0]]) {qtype = tensor<16x3x3x3x!quant.uniform<u8:f32:0,
// CHECK-SAME: {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>>
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tfl.conv_2d"(%arg0, %[[DEQUANTIZE]], %[[CONSTANT]])
// CHECK: return %[[CONV]] : tensor<256x8x7x16xf32>
}

// CHECK-LABEL: fakeQuantWithDepthwiseConv2D
func.func @fakeQuantWithDepthwiseConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = arith.constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = arith.constant dense<0.0> : tensor<f32>
  %max = arith.constant dense<255.0> : tensor<f32>
  %mini = "tf.Identity"(%min) : (tensor<f32>) -> tensor<f32>
  %maxi = "tf.Identity"(%max) : (tensor<f32>) -> tensor<f32>
  %fq = "tf.FakeQuantWithMinMaxVars"(%in, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<f32>, tensor<f32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.DepthwiseConv2dNative"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %rst : tensor<256x30x30x16xf32>

// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<48xf32>
// CHECK-DAG: %[[CONSTANT0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x3x48xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT0]]) {qtype = tensor<1x3x3x48x!quant.uniform<u8:f32, 1.000000e+00>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[DEQUANTIZE]], %[[CONSTANT]])
// CHECK: return %[[CONV]]
}

// CHECK-LABEL: perChannelFakeQuantWithDepthwiseConv2D
func.func @perChannelFakeQuantWithDepthwiseConv2D(tensor<256x32x32x3xf32>) -> (tensor<256x30x30x16xf32>) {
^bb0(%arg: tensor<256x32x32x3xf32>) :
  %in = arith.constant dense<0.0> : tensor<3x3x3x16xf32>
  %min = arith.constant dense<0.0> : tensor<16xf32>
  %max = arith.constant dense<255.0> : tensor<16xf32>
  %mini = "tf.Identity"(%min) : (tensor<16xf32>) -> tensor<16xf32>
  %maxi = "tf.Identity"(%max) : (tensor<16xf32>) -> tensor<16xf32>
  %fq = "tf.FakeQuantWithMinMaxVarsPerChannel"(%in, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<3x3x3x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<3x3x3x16xf32>
  %rst = "tf.DepthwiseConv2dNative"(%arg, %fq) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %rst : tensor<256x30x30x16xf32>

// CHECK-DAG: %[[CONSTANT0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x3x48xf32>
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<48xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT0]]) {qtype = tensor<1x3x3x48x!quant.uniform<u8:f32:3,
// CHECK-SAME: {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-SAME:  1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-SAME:  1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[DEQUANTIZE]], %[[CONSTANT]])
// CHECK: return %[[CONV]]
}

// CHECK-LABEL: perChannelFakeQuantWithDepthwiseConv2DWithReshape
func.func @perChannelFakeQuantWithDepthwiseConv2DWithReshape(%arg: tensor<1x160x160x48xf32>) -> (tensor<1x160x160x48xf32>) {
  %in = arith.constant dense<0.0> : tensor<3x3x48x1xf32>
  %min = arith.constant dense<0.0> : tensor<48xf32>
  %max = arith.constant dense<255.0> : tensor<48xf32>
  %mini = "tf.Identity"(%min) : (tensor<48xf32>) -> tensor<48xf32>
  %maxi = "tf.Identity"(%max) : (tensor<48xf32>) -> tensor<48xf32>
  %s1 = arith.constant dense<[3, 3, 48]> : tensor<3xi32>
  %s2 = arith.constant dense<[3, 3, 48, 1]> : tensor<4xi32>
  %r1 = "tf.Reshape"(%in, %s1) {T = f32, Tshape = i32, device = ""} : (tensor<3x3x48x1xf32>, tensor<3xi32>) -> tensor<3x3x48xf32>
  %fq = "tf.FakeQuantWithMinMaxVarsPerChannel"(%r1, %mini, %maxi) {num_bits = 5, narrow_range = false} : (tensor<3x3x48xf32>, tensor<48xf32>, tensor<48xf32>) -> tensor<3x3x48xf32>
  %r2 = "tf.Reshape"(%fq, %s2) {T = f32, Tshape = i32, device = ""} : (tensor<3x3x48xf32>, tensor<4xi32>) -> tensor<3x3x48x1xf32>
  %rst = "tf.DepthwiseConv2dNative"(%arg, %r2) {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x160x160x48xf32>, tensor<3x3x48x1xf32>) -> tensor<1x160x160x48xf32>
  func.return %rst : tensor<1x160x160x48xf32>

// CHECK-DAG: %[[CONSTANT0:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x3x48xf32>
// CHECK-DAG: %[[CONSTANT:.*]] = arith.constant dense<0.000000e+00> : tensor<48xf32>
// CHECK: %[[QUANTIZE:.*]] = "tfl.quantize"(%[[CONSTANT0]]) {qtype = tensor<1x3x3x48x!quant.uniform<u8:f32:3,
// CHECK-SAME: {1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-SAME:  1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,
// CHECK-SAME:  1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00}>>}
// CHECK: %[[DEQUANTIZE:.*]] = "tfl.dequantize"(%[[QUANTIZE]])
// CHECK: %[[CONV:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[DEQUANTIZE]], %[[CONSTANT]])
// CHECK: return %[[CONV]]
}

// LOBIT-LABEL: fakeQuant3BitPerChannelForActivation
func.func @fakeQuant3BitPerChannelForActivation(%arg0: tensor<8x4xf32>) -> (tensor<8x4xf32>) {
  %arg1 = arith.constant dense<[0.0, -1.0, -31.0, -30.0]> : tensor<4xf32>
  %arg2 = arith.constant dense<[31.0, 30.0, 31.0, 32.0]> : tensor<4xf32>
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<8x4xf32>
  func.return %0 : tensor<8x4xf32>

// LOBIT:  %[[fq:.*]] = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %cst, %cst_0)
// LOBIT:  %[[q:.*]] = "tfl.quantize"(%[[fq]]) {qtype = tensor<8x4x!quant.uniform<u8<0:31>:f32:1, {1.000000e+00,1.000000e+00:1,2.000000e+00:16,2.000000e+00:15}>>}
// LOBIT:  %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// LOBIT:  return %[[dq]]
}

// LOBIT-LABEL: fakeQuant3BitForActivation
func.func @fakeQuant3BitForActivation(tensor<8xf32>) -> (tensor<8xf32>) {
^bb0(%arg0: tensor<8xf32>):
  %arg1 = arith.constant dense<-30.0> : tensor<f32>
  %arg2 = arith.constant dense<32.0> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 5, narrow_range = false} : (tensor<8xf32>, tensor<f32>, tensor<f32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>

// LOBIT:  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0)
// LOBIT:  %1 = "tfl.quantize"(%0) {qtype = tensor<8x!quant.uniform<u8<0:31>:f32, 2.000000e+00:15>>}
// LOBIT:  %2 = "tfl.dequantize"(%1)
// LOBIT:  return %2
}

}
