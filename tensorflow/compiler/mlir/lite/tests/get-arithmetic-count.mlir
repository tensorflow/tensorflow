// RUN: tf-opt -split-input-file -verify-diagnostics -tfl-get-arithmetic-count %s | FileCheck %s

func @testConv2D(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: _arithmetic_count = 230686720 : i64
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  return %0 : tensor<256x32x32x16xf32>
}

func @testConv2DDynamicShape(tensor<?x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32> {
^bb0(%arg0: tensor<?x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: _arithmetic_count = -1 : i64
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<?x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32>
  return %0 : tensor<?x32x32x16xf32>
}

func @testDepthwiseConv2D(tensor<1x112x112x3xf32>, tensor<1x3x3x32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32> {
^bb0(%arg0: tensor<1x112x112x3xf32>, %arg1: tensor<1x3x3x32xf32>, %arg2: tensor<32xf32>):
  // CHECK: _arithmetic_count = 7626752 : i64
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<1x112x112x3xf32>, tensor<1x3x3x32xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}

func @fully_connected(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>, %arg2: tensor<40xf32>) -> tensor<1x40xf32> {
  // CHECK: _arithmetic_count = 3000 : i64
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<1x40xf32>
  return %0 : tensor<1x40xf32>
}

func @testAdd(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testAddBroadcast(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testSub(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.sub"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testSubBroadcast(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.sub"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testMul(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testMulBroadcast(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testDiv(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.div"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testDivBroadcast(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> {
  // CHECK: _arithmetic_count = 1000 : i64
  %0 = "tfl.div"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

func @testLogistic(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 6400 : i64
  %0 = "tfl.logistic"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

func @testLogSoftmax(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 6400 : i64
  %0 = "tfl.log_softmax"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

func @testSoftmax(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 6400 : i64
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

func @testTanh(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 6400 : i64
  %0 = "tfl.tanh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

func @testAddN(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 200 : i64
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE"} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}

func @testMaxPool2D(tensor<1x10x10x3xf32>) -> tensor<1x10x10x3xf32> {
^bb0(%arg0: tensor<1x10x10x3xf32>):
  // CHECK: _arithmetic_count = 2700 : i64
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x10x10x3xf32>) -> tensor<1x10x10x3xf32>
  return %0 : tensor<1x10x10x3xf32>
}

func @testAveragePool2D(tensor<1x10x10x3xf32>) -> tensor<1x10x10x3xf32> {
^bb0(%arg0: tensor<1x10x10x3xf32>):
  // CHECK: _arithmetic_count = 2700 : i64
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x10x10x3xf32>) -> tensor<1x10x10x3xf32>
  return %0 : tensor<1x10x10x3xf32>
}

func @testTransposeConv(%arg0: tensor<4xi32>, %arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = constant unit
  // CHECK: _arithmetic_count = 176160768 : i64
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  return %0 : tensor<1x64x84x32xf32>
}

func @testL2Norm(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: _arithmetic_count = 300 : i64
  %0 = "tfl.l2_normalization"(%arg0) {fused_activation_function = "NONE"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  return %0 : tensor<10x10xf32>
}
