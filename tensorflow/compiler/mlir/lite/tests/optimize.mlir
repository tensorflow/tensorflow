// Run optimize pass only and check the results.
// RUN: tf-opt %s -tfl-optimize='enable-canonicalization=false' | FileCheck %s
// Run optimize pass and then canonicalize pass, and make sure some folding is applied.
// RUN: tf-opt %s -tfl-optimize | FileCheck --check-prefix=FOLD %s
// Run legalize pass and then optimize pass, and make sure some fusing is applied.
// RUN: tf-opt %s -tfl-legalize-tf -tfl-optimize='enable-canonicalization=false' | FileCheck --check-prefix=Fusing %s
// Run legalize pass and then optimize pass, and make sure some fusing is applied, but no mul->fc.
// RUN: tf-opt %s -tfl-legalize-tf -tfl-optimize='enable-canonicalization=false disable-fuse-mul-and-fc=true'  | FileCheck --check-prefix=NoFusing %s

// CHECK-LABEL: fusedConv2dRelu
func.func @fusedConv2dRelu(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x32x32x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<256x32x32x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %1 : tensor<256x32x32x16xf32>

  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fusedDepthwiseConv2dRelu6
func.func @fusedDepthwiseConv2dRelu6(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.relu6"(%0) : (tensor<256x30x30x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) <{depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fusedMaxPool2dRelu
func.func @fusedMaxPool2dRelu(%arg0: tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32> {
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32>
  %1 = "tfl.relu"(%0) : (tensor<1x73x73x16xf32>) -> tensor<1x73x73x16xf32>
  func.return %1 : tensor<1x73x73x16xf32>

  // CHECK: %0 = "tfl.max_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "RELU", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fusedAvgPool2dRelu1
func.func @fusedAvgPool2dRelu1(%arg0: tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32> {
  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<1x73x73x16xf32>) -> tensor<1x73x73x16xf32>
  func.return %1 : tensor<1x73x73x16xf32>

  // CHECK: %0 = "tfl.average_pool_2d"(%arg0) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "RELU_N1_TO_1", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x147x147x16xf32>) -> tensor<1x73x73x16xf32>
  // CHECK: return %0
}

// CHECK-LABEL: fuseAddIntoConv2d
func.func @fuseAddIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x32x32x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  func.return %1 : tensor<256x32x32x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseBroadcastedAddIntoConv2D
func.func @fuseBroadcastedAddIntoConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x32x32x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<1x1x16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x16xf32>, tensor<1x1x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %1 : tensor<256x32x32x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuse4DAddIntoConv2d
func.func @fuse4DAddIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<2x3x3x3xf32>) -> tensor<256x32x32x2xf32> {
  %cst = arith.constant dense<[[[[1.0, 2.0]]]]> : tensor<1x1x1x2xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {
    dilation_h_factor = 1 : i32,
    dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE",
    padding = "SAME",
    stride_h = 1 : i32,
    stride_w = 1 : i32
  } : (tensor<256x32x32x3xf32>, tensor<2x3x3x3xf32>, tensor<2xf32>) -> tensor<256x32x32x2xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x2xf32>, tensor<1x1x1x2xf32>) -> tensor<256x32x32x2xf32>
  func.return %1 : tensor<256x32x32x2xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseSubIntoConv2d
func.func @fuseSubIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x32x32x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x16xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  func.return %1 : tensor<256x32x32x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseAddIntoTransposeConv
func.func @fuseAddIntoTransposeConv(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = arith.constant dense<[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]> : tensor<32xf32>
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.000000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<[2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00, 2.500000e+00, 3.500000e+00]> : tensor<32xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseSubIntoTransposeConv
func.func @fuseSubIntoTransposeConv(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = arith.constant dense<[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]> : tensor<32xf32>
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.000000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<[-5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01, -5.000000e-01, 5.000000e-01]> : tensor<32xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseAddIntoTransposeConvNoBias
func.func @fuseAddIntoTransposeConvNoBias(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.000000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<1.500000e+00> : tensor<32xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseSubIntoTransposeConvNoBias
func.func @fuseSubIntoTransposeConvNoBias(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.000000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<-1.500000e+00> : tensor<32xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseMulIntoTransposeConv
func.func @fuseMulIntoTransposeConv(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = arith.constant dense<[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]> : tensor<32xf32>
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.500000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<[1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00, 1.500000e+00, 3.000000e+00]> : tensor<32xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseMulIntoTransposeConvNoBias
func.func @fuseMulIntoTransposeConvNoBias(%arg0: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = arith.constant dense<1.5> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_1 = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_2 = arith.constant dense<1.0> : tensor<32x4x4x128xf32>
  %cst_3 = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.transpose_conv"(%cst_1, %cst_2, %arg0, %cst_3) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<1x64x84x32xf32>, tensor<32xf32>) -> tensor<1x64x84x32xf32>
  func.return %1 : tensor<1x64x84x32xf32>

  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<1.500000e+00> : tensor<32x4x4x128xf32>
  // CHECK-DAG: %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[RESULT:.*]] = "tfl.transpose_conv"(%[[SHAPE]], %[[WEIGHTS]], %arg0, %[[BIAS]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseAddIntoFollowingConv2d
func.func @fuseAddIntoFollowingConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>
  %w = arith.constant dense<1.0> : tensor<16x3x3x3xf32>
  %bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.conv_2d"(%0, %w, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
// CHECK-DAG: %[[b:.*]] = "tfl.pseudo_const"(){{.*}}dense<[4.150000e+01, 4.250000e+01, 4.350000e+01, 4.450000e+01, 4.550000e+01, 4.650000e+01, 4.750000e+01, 4.850000e+01, 4.950000e+01, 5.050000e+01, 5.150000e+01, 5.250000e+01, 5.350000e+01, 5.450000e+01, 5.550000e+01, 5.650000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[c:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[c]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: fuseSubIntoFollowingConv2d
func.func @fuseSubIntoFollowingConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>
  %w = arith.constant dense<1.0> : tensor<16x3x3x3xf32>
  %bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.conv_2d"(%0, %w, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
// CHECK-DAG: %[[b:.*]] = "tfl.pseudo_const"(){{.*}}dense<[-3.950000e+01, -3.850000e+01, -3.750000e+01, -3.650000e+01, -3.550000e+01, -3.450000e+01, -3.350000e+01, -3.250000e+01, -3.150000e+01, -3.050000e+01, -2.950000e+01, -2.850000e+01, -2.750000e+01, -2.650000e+01, -2.550000e+01, -2.450000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[c:.*]] = "tfl.conv_2d"(%arg0, %[[w]], %[[b]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[c]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: @fuseAddIntoDepthwiseConv2d
func.func @fuseAddIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = arith.constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseSubIntoDepthwiseConv2d
func.func @fuseSubIntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<0.5> : tensor<16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[5.000000e-01, 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: dontFuseSubIntoDepthwiseConv2d
func.func @dontFuseSubIntoDepthwiseConv2d(%arg0: tensor<256x3x3x3xf32>, %arg1: tensor<3x3x3x5xf32>) -> tensor<256x2x2x4xf32> {
  %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]> : tensor<2x4xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {
    depth_multiplier = 4 : i32,
    dilation_h_factor = 2 : i32,
    dilation_w_factor = 3 : i32,
    fused_activation_function = "NONE",
    padding = "SAME",
    stride_h = 4 : i32,
    stride_w = 5 : i32
    } : (tensor<256x3x3x3xf32>, tensor<3x3x3x5xf32>, tensor<4xf32>) -> tensor<256x2x2x4xf32>
  %1 = "tfl.sub"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x2x2x4xf32>, tensor<2x4xf32>) -> tensor<256x2x2x4xf32>
  func.return %1 : tensor<256x2x2x4xf32>

  // CHECK: "tfl.depthwise_conv_2d"
  // CHECK: tfl.sub
}

// CHECK-LABEL: fuseAddIntoFollowingDepthwiseConv2d
func.func @fuseAddIntoFollowingDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<256x32x32x3xf32>, tensor<f32>) -> tensor<256x32x32x3xf32>

  %w = arith.constant dense<1.0> : tensor<3x3x3x16xf32>
  %bias = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %1 = "tfl.depthwise_conv_2d"(%0, %w, %bias) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<3x3x3x16xf32>
// CHECK-DAG: %[[b:.*]] = "tfl.pseudo_const"(){{.*}}dense<[4.150000e+01, 4.250000e+01, 4.350000e+01, 4.450000e+01, 4.550000e+01, 4.650000e+01, 4.750000e+01, 4.850000e+01, 4.950000e+01, 5.050000e+01, 5.150000e+01, 5.250000e+01, 5.350000e+01, 5.450000e+01, 5.550000e+01, 5.650000e+01]> : tensor<16xf32>
// CHECK-NEXT: %[[dc:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[w]], %[[b]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK-NEXT: return %[[dc]] : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: fuseAddWithRelu6IntoConv2d
func.func @fuseAddWithRelu6IntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x8x7x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  func.return %1 : tensor<256x8x7x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
  // CHECK-SAME: fused_activation_function = "RELU6"
}

// CHECK-LABEL: @fuseAddWithRelu6IntoDepthwiseConv2d
func.func @fuseAddWithRelu6IntoDepthwiseConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %cst_0 = arith.constant dense<1.5> : tensor<16xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst_0) {depth_multiplier = 4 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "RELU6"} : (tensor<256x30x30x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %1 : tensor<256x30x30x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst)
  // CHECK-SAME: fused_activation_function = "RELU6"
}

// CHECK-LABEL: fuseMulIntoConv2dWithQDQs
func.func @fuseMulIntoConv2dWithQDQs(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x8x7x3xf32> {
  %cst = arith.constant dense<1.5> : tensor<3xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %w = arith.constant dense<2.0> : tensor<3x3x3x3xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>} : (tensor<3x3x3x3xf32>) -> tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0,{1.0,2.0,3.0}>>) -> tensor<3x3x3x3xf32>
  %0 = "tfl.conv_2d"(%arg0, %dq, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  %1 = "tfl.mul"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x3xf32>, tensor<3xf32>) -> tensor<256x8x7x3xf32>
  func.return %1 : tensor<256x8x7x3xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant{{.*}}dense<3.000000e+00> : tensor<3x3x3x3xf32>
// CHECK-DAG: %[[cst:.*]] = arith.constant{{.*}}dense<[1.500000e+00, 3.000000e+00, 4.500000e+00]> : tensor<3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<3x3x3x3x!quant.uniform<i8<-127:127>:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[dq]], %[[cst]])
// CHECK: return %[[conv]] : tensor<256x8x7x3xf32>
}

// CHECK-LABEL: fuseMulIntoConv2dWithQDQsAndTranspose
func.func @fuseMulIntoConv2dWithQDQsAndTranspose(%arg0: tensor<1x6x6x3xf32>) -> (tensor<1x6x6x6xf32>){
  %cst = arith.constant dense<0.999994993> : tensor<1x1x1x6xf32>
  %cst_0 = arith.constant dense<"0x9465DEBE278B783E6E94913D38F7A2BE91AEA7BD202C85BE952E79BE3956663E5D78A73D6AAC123E2911133B953E6BBD813E833CFC41B5BDB202263DA20C84BE6E4832BE2B2AD1BEFAFDA93DBDA633BE6FF8943EEEE0603C5840693E04853EBD753CD73D2EB8563EE60A0DBE2A425EBE331A633E405A083E10D4F0BD293AF03DCB7C8F3E788714BD290204BED12A973B8956B2BD4559293EBB4A2DBE229D1FBDAA1E533E7D485D3EC4FC81BEEEE75D3E97E2B7BD74A88FBE384920BE3BC15B3E3ADC43BC853502BDFBD6AFBE8CBB82BC209A05BEC12D3D3E97A791BE2F290D3ED2DEBABD0C61D0BDD185C6BE947EFA3D9CA10FBE530BCD3C08F002BEF651D3BDD4E86D3E635E8E3E0F530C3D463204BEF6F9A6BEFC5F8D3D1B030D3EB0F1713EF76957BEBEAB763E6D459D3E9F2C07BE6E059C3EC58D4EBEE3D7593E598178BE8119D63E7434973DED4F69BE0D24873EA0A68EBEE36B81BE4CB1DABDF046E23D8154F13D684DCFBE8B854C3E475900BEDB4A973ECC8180BD6FC6A13D00CAD03C5762563E1B6E923EED06953EA717B33CB7A5123C086C06BE1B20C7BD862C433E0DDBAD3D3A641BBED0DA883EE81BACBECF565BBEE8A3BFBE3CB86CBECE3D7ABDFE0DA23D3283A63D9F530E3E3D213ABEA80051BC2E27553EFE3C1A3EA2769DBDC82340BDA5B628BEE273DE3CCBDB4BBE1ABAC7BD6963B0BE474A8FBE68396EBE35D8253CD61EABBDC877A53E299CA2BEA6A895BD1DBFADBC6B9D933E853995BE6CB3DDBE4C4CF43D6D5B723E7B9D0DBEF7A3BB3DF7947D3E082553BE5AE7B1BE939B343CEE92053EF733D33EC5CEB83D98D71A3EECD38B3E456A0ABE2042A7BCBB77A23DEB9501BE781C723ECD72B33DACA6813EFCEE923D4FBF223BC9A0AEBDADAFCA3C0759B33D"> : tensor<3x3x3x6xf32>
  %cst_1 = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst_2 = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<6xf32>
  %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>}> : (tensor<1x6x6x3xf32>) -> tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>) -> tensor<1x6x6x3xf32>
  %2 = "tfl.quantize"(%cst_0) <{qtype = tensor<3x3x3x6x!quant.uniform<i8:f32:3, {1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02}>>}> : (tensor<3x3x3x6xf32>) -> tensor<3x3x3x6x!quant.uniform<i8:f32:3, {1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02}>>
  %3 = "tfl.dequantize"(%2) : (tensor<3x3x3x6x!quant.uniform<i8:f32:3, {1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02,1.562500e-02}>>) -> tensor<3x3x3x6xf32>
  %4 = "tfl.transpose"(%3, %cst_2) : (tensor<3x3x3x6xf32>, tensor<4xi32>) -> tensor<6x3x3x3xf32>
  %5 = "tfl.pad"(%1, %cst_1) : (tensor<1x6x6x3xf32>, tensor<4x2xi32>) -> tensor<1x8x8x3xf32>
  %6 = "tfl.conv_2d"(%5, %4, %cst_3) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x8x8x3xf32>, tensor<6x3x3x3xf32>, tensor<6xf32>) -> tensor<1x6x6x6xf32>
  %7 = tfl.mul(%6, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x6x6x6xf32>, tensor<1x1x1x6xf32>) -> tensor<1x6x6x6xf32>
  %8 = "tfl.quantize"(%7) <{qtype = tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>}> : (tensor<1x6x6x6xf32>) -> tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>
  %9 = "tfl.dequantize"(%8) : (tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>) -> tensor<1x6x6x6xf32>
  return %9 : tensor<1x6x6x6xf32>

// CHECK:   %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<6xf32>
// CHECK:   %[[cst_0:.*]] = arith.constant dense<"0x4B65DEBED58A783E3E94913D03F7A2BE5AAEA7BDF42B85BE432E79BEED55663E2678A73D3AAC123EF910133B483E6BBD563E833CC141B5BD7C02263D770C84BE344832BEE629D1BEC2FDA93D82A633BE3EF8943EA4E0603C0B40693EC5843EBD2E3CD73DE8B7563EB80A0DBEE1415EBEE819633E135A083EC1D3F0BDDA39F03D9C7C8F3E478714BDFE0104BE9F2A973B4E56B2BD0D59293E824A2DBEEE9C1FBD651E533E34485D3E99FC81BEA5E75D3E5BE2B7BD45A88FBE034920BEF3C05B3EFADB43BC5A3502BDC1D6AFBE61BB82BCF49905BE832D3D3E67A791BE01290D3E95DEBABDC860D0BD9085C6BE427EFA3D6DA10FBE100BCD3CDDEF02BEB151D3BD86E86D3E345E8E3EE1520C3D1B3204BEBFF9A6BECE5F8D3DED020D3E61F1713EB06957BE6DAB763E39459D3E732C07BE3B059C3E818D4EBE9CD7593E078178BE3B19D63E4234973DA04F69BEE123873E71A68EBEB96B81BE04B1DABDA646E23D3254F13D244DCFBE48854C3E1D5900BEA94A973EA28180BD3AC6A13DBBC9D03C1162563EEB6D923EBC06953E6C17B33C87A5123CDC6B06BEDA1FC7BD462C433ED4DAAD3D07641BBEA3DA883EB01BACBE87565BBEA9A3BFBEEEB76CBE7C3D7ABDC90DA23DFB82A63D70530E3E00213ABE630051BCE826553ECB3C1A3E6E769DBD892340BD6EB628BE9973DE3C88DB4BBED8B9C7BD2F63B0BE184A8FBE1A396EBEFFD7253C9E1EABBD9277A53EF49BA2BE75A895BDE4BEADBC3B9D933E543995BE23B3DDBEFC4BF43D1D5B723E4D9D0DBEB9A3BB3DA4947D3EC32453BE20E7B1BE589B343CC292053EB233D33E88CEB83D65D71A3EBED38B3E186A0ABEE941A7BC8677A23DC09501BE291C723E9272B33D81A6813ECCEE923D1ABF223B90A0AEBD6AAFCA3CCC58B33D"> : tensor<3x3x3x6xf32>
// CHECK:   %[[cst_1:.*]] = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
// CHECK:   %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>}> : (tensor<1x6x6x3xf32>) -> tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>
// CHECK:   %1 = "tfl.dequantize"(%0) : (tensor<1x6x6x3x!quant.uniform<i8:f32, 1.562500e-02>>) -> tensor<1x6x6x3xf32>
// CHECK:   %2 = "tfl.quantize"(%[[cst_0:.*]]) <{qtype = tensor<3x3x3x6x!quant.uniform<i8:f32:3, {0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732}>>}> : (tensor<3x3x3x6xf32>) -> tensor<3x3x3x6x!quant.uniform<i8:f32:3, {0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732}>>
// CHECK:   %3 = "tfl.dequantize"(%2) : (tensor<3x3x3x6x!quant.uniform<i8:f32:3, {0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732,0.015624921768903732}>>) -> tensor<3x3x3x6xf32>
// CHECK:   %4 = "tfl.transpose"(%3, %[[cst_1]]) : (tensor<3x3x3x6xf32>, tensor<4xi32>) -> tensor<6x3x3x3xf32>
// CHECK:   %5 = "tfl.conv_2d"(%1, %4, %[[cst]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x3xf32>, tensor<6x3x3x3xf32>, tensor<6xf32>) -> tensor<1x6x6x6xf32>
// CHECK:   %6 = "tfl.quantize"(%5) <{qtype = tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>}> : (tensor<1x6x6x6xf32>) -> tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>
// CHECK:   %7 = "tfl.dequantize"(%6) : (tensor<1x6x6x6x!quant.uniform<i8:f32, 1.562500e-02>>) -> tensor<1x6x6x6xf32>
// CHECK:   return %7 : tensor<1x6x6x6xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnectedWithOptionalAttribute
func.func @fuseMulIntoFullyConnectedWithOptionalAttribute(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  func.return %1 : tensor<4x2xf32>

// CHECK-DAG:  %[[CONSTANT:.*]] = "tfl.pseudo_const"(){{.*}}dense<{{\[\[}}1.000000e+00, 2.000000e+00], [6.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG:  %[[CONSTANT0:.*]] = "tfl.pseudo_const"(){{.*}}dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) <{asymmetric_quantize_inputs = true,
}

// CHECK-LABEL: @fuseMulIntoFullyConnected
func.func @fuseMulIntoFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  func.return %1 : tensor<4x2xf32>

// CHECK-DAG:  %[[CONSTANT:.*]] = "tfl.pseudo_const"(){{.*}}dense<{{\[\[}}1.000000e+00, 2.000000e+00], [6.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG:  %[[CONSTANT0:.*]] = "tfl.pseudo_const"(){{.*}}dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK:  return %[[RES]] : tensor<4x2xf32>
}

// CHECK-LABEL: @DontFuseMulIntoFullyConnectedForLargeFilter
func.func @DontFuseMulIntoFullyConnectedForLargeFilter(%arg0: tensor<128x256000xf32>) -> tensor<128x1024xf32> {
  %cst0 = arith.constant dense<2.0> : tensor<1024x256000xf32>
  %cst1 = arith.constant dense<2.0> : tensor<1024xf32>
  %cst2 = arith.constant dense<2.0> : tensor<1024xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<128x256000xf32>, tensor<1024x256000xf32>, tensor<1024xf32>) -> tensor<128x1024xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<128x1024xf32>, tensor<1024xf32>) -> tensor<128x1024xf32>

  func.return %1 : tensor<128x1024xf32>

// CHECK:  %[[a:.*]] = "tfl.fully_connected"(%arg0, %cst, %cst_0) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK:  %[[b:.*]] = tfl.mul(%[[a]], %cst_0) <{fused_activation_function = "RELU6"}>
}


// CHECK-LABEL: @skipFuseMulIntoFullyConnected
func.func @skipFuseMulIntoFullyConnected(%arg0: tensor<4x2xf32>) -> (tensor<1x8xf32>, tensor<4x2xf32>) {
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %cst3 = arith.constant dense<[1, 8]> : tensor<2xi32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.reshape"(%0, %cst3) : (tensor<4x2xf32>, tensor<2xi32>) -> tensor<1x8xf32>
  %2 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  func.return %1, %2 : tensor<1x8xf32>, tensor<4x2xf32>
  // CHECK:  %cst = arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  // CHECK:  %cst_0 = arith.constant dense<2.000000e+00> : tensor<2xf32>
  // CHECK:  %cst_1 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK:  %cst_2 = arith.constant dense<[1, 8]> : tensor<2xi32>
  // CHECK:  %0 = "tfl.fully_connected"(%arg0, %cst, %cst_0) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  // CHECK:  %1 = "tfl.reshape"(%0, %cst_2) : (tensor<4x2xf32>, tensor<2xi32>) -> tensor<1x8xf32>
  // CHECK:  %2 = tfl.mul(%0, %cst_1) <{fused_activation_function = "RELU6"}> : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  // CHECK:  return %1, %2 : tensor<1x8xf32>, tensor<4x2xf32>
}

// CHECK-LABEL: @fuseAddIntoFollowingFullyConnectedWithQDQs
func.func @fuseAddIntoFollowingFullyConnectedWithQDQs(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %q = "tfl.quantize"(%cst0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0>>
  %dq = "tfl.dequantize"(%q) : (tensor<2x2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %dq, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG: %[[b:.*]] = "tfl.pseudo_const"(){{.*}}dense<[6.500000e+00, 1.250000e+01]> : tensor<2xf32>
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[w]])
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[dq]], %[[b]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseAddIntoFollowingFullyConnected
func.func @fuseAddIntoFollowingFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

// CHECK-DAG: %[[w:.*]] = arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
// CHECK-DAG: %[[b:.*]] = "tfl.pseudo_const"(){{.*}}dense<[6.500000e+00, 1.250000e+01]> : tensor<2xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
}

// CHECK-LABEL: @doNotFuseAddIntoFollowingFullyConnected
func.func @doNotFuseAddIntoFollowingFullyConnected(%arg0: tensor<4x2xf32>, %arg1: tensor<*xf32>) -> tensor<4x2xf32> {
  %cst1 = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst1) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst = arith.constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<*xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

// CHECK: tfl.add
// CHECK: "tfl.fully_connected"
}

// CHECK-LABEL: @fuseMulIntoFollowingFullyConnected
func.func @fuseMulIntoFollowingFullyConnected(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst2 = arith.constant dense<1.5> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

// CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<2xf32>
// CHECK-DAG: %[[w:.*]] = "tfl.pseudo_const"(){{.*}}dense<{{\[}}[1.500000e+00, 3.000000e+00], [4.500000e+00, 6.000000e+00]]> : tensor<2x2xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[w]], %[[b]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: return %[[fc]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnectedBroadcast
func.func @fuseMulIntoFullyConnectedBroadcast(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32> {
  %cst0 = arith.constant dense<[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  // %cst2 isn't broadcast-compatible to %cst0, but tf.Mul is able to fold them.
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  func.return %1 : tensor<1x2xf32>

// CHECK-DAG:  %[[CONSTANT:.*]] = "tfl.pseudo_const"(){{.*}}dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
// CHECK-DAG:  %[[CONSTANT0:.*]] = "tfl.pseudo_const"(){{.*}}dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %[[CONSTANT0]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK:  return %[[RES]] : tensor<1x2xf32>
}

// CHECK-LABEL: @fuseMulIntoFullyConnectedNoBias
func.func @fuseMulIntoFullyConnectedNoBias(%arg0: tensor<4x2xf32>, %arg1: none) -> tensor<4x2xf32> {
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %arg1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<4x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>

  func.return %1 : tensor<4x2xf32>

// CHECK-DAG:  %[[CONSTANT:.*]] = "tfl.pseudo_const"(){{.*}}dense<{{\[\[}}1.000000e+00, 2.000000e+00], [6.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>
// CHECK:  %[[RES:.*]] = "tfl.fully_connected"(%arg0, %[[CONSTANT]], %arg1) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
// CHECK:  return %[[RES]] : tensor<4x2xf32>
}

// CHECK-LABEL: @fuseMulIntoDepthwiseConv2d
func.func @fuseMulIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>

  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  func.return %1 : tensor<1x112x112x2xf32>

// CHECK-DAG:  %cst = arith.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00], [5.000000e+00, 1.200000e+01]], {{\[\[}}7.000000e+00, 1.600000e+01], [9.000000e+00, 2.000000e+01], [1.100000e+01, 2.400000e+01]], {{\[\[}}1.300000e+01, 2.800000e+01], [1.500000e+01, 3.200000e+01], [1.700000e+01, 3.600000e+01]]]]> : tensor<1x3x3x2xf32>
// CHECK-DAG:  %cst_0 = arith.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
// CHECK:  return %0
}

// CHECK-LABEL: @fuse4DMulIntoDepthwiseConv2d
func.func @fuse4DMulIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[[[[1.0, 2.0]]]]> : tensor<1x1x1x2xf32>

  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x112x112x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x112x112x2xf32>

  func.return %1 : tensor<1x112x112x2xf32>

// CHECK-DAG:  %cst = arith.constant dense<{{\[\[\[\[}}1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00], [5.000000e+00, 1.200000e+01]], {{\[\[}}7.000000e+00, 1.600000e+01], [9.000000e+00, 2.000000e+01], [1.100000e+01, 2.400000e+01]], {{\[\[}}1.300000e+01, 2.800000e+01], [1.500000e+01, 3.200000e+01], [1.700000e+01, 3.600000e+01]]]]> : tensor<1x3x3x2xf32>
// CHECK-DAG:  %cst_0 = arith.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x112x112x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
// CHECK:  return %0
}

// CHECK-LABEL: @notFuseMulIntoDepthwiseConv2d
func.func @notFuseMulIntoDepthwiseConv2d(%arg0: tensor<1x4x4x2xf32>) -> tensor<1x4x4x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[[3.1, 3.2], [3.1, 3.2], [3.1, 3.2], [3.1, 3.2]]> : tensor<4x2xf32>

  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x4x4x2xf32>, tensor<1x3x3x2xf32>, tensor<2xf32>) -> tensor<1x4x4x2xf32>
  // We cannot fuse this tfl.mul into the preceding conv op because %cst2 is not broadcast-compatible to %cst0.
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x4x4x2xf32>, tensor<4x2xf32>) -> tensor<1x4x4x2xf32>

  func.return %1 : tensor<1x4x4x2xf32>

// CHECK:  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0)
// CHECK:  %1 = tfl.mul(%0, %cst_1)
// CHECK:  return %1
}

// CHECK-LABEL: @FuseFullyConnectedAddWithNoBias
func.func @FuseFullyConnectedAddWithNoBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %cst = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %cst)
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithNoBiasWithQDQs
func.func @FuseFullyConnectedAddWithNoBiasWithQDQs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst1 = arith.constant dense<2.0> : tensor<40xf32>
  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.quantize"(%cst1) {qtype = tensor<2x1x!quant.uniform<i8:f32, 0.024986599940879671:92>>} : (tensor<40xf32>) -> tensor<40x!quant.uniform<i8:f32, 0.024986599940879671:92>>
  %2 = "tfl.dequantize"(%1) : (tensor<40x!quant.uniform<i8:f32, 0.024986599940879671:92>>) -> tensor<40xf32>
  %3 = "tfl.add"(%0, %2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>
  func.return %3 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[dq:.*]] = "tfl.dequantize"
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[dq]])
  // CHECK-NOT: tfl.add
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedReducedAddWithNoBias
func.func @FuseFullyConnectedReducedAddWithNoBias(%arg0: tensor<1024x1x126xf32>, %arg1: tensor<128x126xf32>) -> tensor<1024x1x128xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<1x1x128xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1024x1x126xf32>, tensor<128x126xf32>, none) -> (tensor<1024x1x128xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1024x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1024x1x128xf32>

  func.return %1 : tensor<1024x1x128xf32>

  // CHECK-DAG: %cst = arith.constant dense<2.000000e+00> : tensor<128xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %cst)
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithExistingBias
func.func @FuseFullyConnectedAddWithExistingBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddBroadcasted
func.func @FuseFullyConnectedAddBroadcasted(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst1 = arith.constant dense<2.0> : tensor<1x40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst1) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<1x40xf32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedMultiUseAddBroadcasted
func.func @FuseFullyConnectedMultiUseAddBroadcasted(%arg0: tensor<1x40x37xf32>, %arg1: tensor<4x37xf32>) -> (tensor<1x40x4xf32>, tensor<1x40x4xf32>) {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst1 = arith.constant dense<[[[2.0, 3.0, 4.0, 5.0]]]> : tensor<1x1x4xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> (tensor<1x40x4xf32>)
  %1 = "tfl.add"(%0, %cst1) {fused_activation_function = "NONE"} : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  %2 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> (tensor<1x40x4xf32>)
  %3 = "tfl.add"(%2, %cst1) {fused_activation_function = "NONE"} : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  func.return %1, %3 : tensor<1x40x4xf32>, tensor<1x40x4xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<{{.*}}>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: %[[fc1:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]], %[[fc1]]
}

// CHECK-LABEL: @FuseFullyConnectedMultiUseAddBroadcastedNagative
func.func @FuseFullyConnectedMultiUseAddBroadcastedNagative(%arg0: tensor<1x40x37xf32>, %arg1: tensor<4x37xf32>) -> (tensor<1x40x4xf32>, tensor<1x40x4xf32>, tensor<1x40x4xf32>) {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst1 = arith.constant dense<[[[2.0, 3.0, 4.0, 5.0]]]> : tensor<1x1x4xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> (tensor<1x40x4xf32>)
  %1 = "tfl.add"(%0, %cst1) {fused_activation_function = "NONE"} : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  %2 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> (tensor<1x40x4xf32>)
  %3 = "tfl.add"(%2, %cst1) {fused_activation_function = "NONE"} : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  %4 = "tfl.mul"(%2, %cst1) {fused_activation_function = "NONE"} : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  func.return %1, %3, %4 : tensor<1x40x4xf32>, tensor<1x40x4xf32>, tensor<1x40x4xf32>

  // CHECK:  %0 = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %cst = arith.constant dense<{{\[\[\[}}2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]]]> : tensor<1x1x4xf32>
  // CHECK:  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> tensor<1x40x4xf32>
  // CHECK:  %2 = tfl.add(%1, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  // CHECK:  %3 = "tfl.fully_connected"(%arg0, %arg1, %0) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x40x37xf32>, tensor<4x37xf32>, none) -> tensor<1x40x4xf32>
  // CHECK:  %4 = tfl.add(%3, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  // CHECK:  %5 = tfl.mul(%3, %cst) <{fused_activation_function = "NONE"}> : (tensor<1x40x4xf32>, tensor<1x1x4xf32>) -> tensor<1x40x4xf32>
  // CHECK:  return %2, %4, %5 : tensor<1x40x4xf32>, tensor<1x40x4xf32>, tensor<1x40x4xf32>
}

// CHECK-LABEL: @FuseFullyConnectedBroadcastedBiasAddWithQDQs
func.func @FuseFullyConnectedBroadcastedBiasAddWithQDQs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst1 = arith.constant dense<2.0> : tensor<1x40xf32>
  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.quantize"(%cst1) {qtype = tensor<2x1x!quant.uniform<i8:f32, 0.024986599940879671:92>>} : (tensor<1x40xf32>) -> tensor<1x40x!quant.uniform<i8:f32, 0.024986599940879671:92>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x40x!quant.uniform<i8:f32, 0.024986599940879671:92>>) -> tensor<1x40xf32>
  %3 = "tfl.add"(%0, %2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<1x40xf32>) -> tensor<40x40xf32>
  func.return %3 : tensor<40x40xf32>

  // CHECK: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[q:.*]] = "tfl.quantize"
  // CHECK-SAME: <{qtype = tensor<40x!quant.uniform<i8:f32, 0.024986599940879671:92>>}> : (tensor<40xf32>) -> tensor<40x!quant.uniform<i8:f32, 0.024986599940879671:92>>
  // CHECK: %[[dq:.*]] = "tfl.dequantize"
  // CHECK-SAME: (tensor<40x!quant.uniform<i8:f32, 0.024986599940879671:92>>) -> tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"
  // CHECK-SAME: (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<40x40xf32>
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddBroadcastedExistingBias
func.func @FuseFullyConnectedAddBroadcastedExistingBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<1x40xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<1x40xf32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}



// CHECK-LABEL: @FuseFullyConnectedAddWithNoBiasAndScalarRhs
func.func @FuseFullyConnectedAddWithNoBiasAndScalarRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<f32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithScalarRhs
func.func @FuseFullyConnectedAddWithScalarRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<f32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>

  func.return %1 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedAddNoBiasWithUnfusableRhs
func.func @FuseFullyConnectedAddNoBiasWithUnfusableRhs(%arg0: tensor<4x37xf32>, %arg1: tensor<4x37xf32>) -> tensor<4x4xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<[[2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3]]> : tensor<4x4xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x37xf32>, tensor<4x37xf32>, none) -> (tensor<4x4xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>

  // CHECK-DAG: %[[unit:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK-DAG: %[[filter:.*]] = arith.constant dense<{{.*}}> : tensor<4x4xf32>
  // CHECK: %[[fc_result:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[unit]])
  // CHECK: %[[add_result:.*]] = tfl.add %[[fc_result]], %[[filter]]
  // CHECK: return %[[add_result]]
}

// CHECK-LABEL: @FuseFullyConnectedAddWithUnfusableRhs
func.func @FuseFullyConnectedAddWithUnfusableRhs(%arg0: tensor<4x37xf32>, %arg1: tensor<4x37xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant dense<[2.0, 2.1, 2.2, 2.3]> : tensor<4xf32>
  %cst2 = arith.constant dense<[[2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3], [2.0, 2.1, 2.2, 2.3]]> : tensor<4x4xf32>

  %0 = "tfl.fully_connected" (%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x37xf32>, tensor<4x37xf32>, tensor<4xf32>) -> (tensor<4x4xf32>)
  %1 = "tfl.add"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>

  // CHECK-DAG: %[[bias:.*]] = arith.constant dense<{{.*}}> : tensor<4xf32>
  // CHECK-DAG: %[[filter:.*]] = arith.constant dense<{{.*}}> : tensor<4x4xf32>
  // CHECK: %[[fc_result:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[bias]])
  // CHECK: %[[add_result:.*]] = tfl.add %[[fc_result]], %[[filter]]
  // CHECK: return %[[add_result]]
}

// CHECK-LABEL: @FuseReshapeAroundBMMLHS
func.func @FuseReshapeAroundBMMLHS(%arg0: tensor<6x5x1024xf32>) -> tensor<6x5x8192xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<1024x8192xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<3xi32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst_1) : (tensor<6x5x1024xf32>, tensor<2xi32>) -> tensor<30x1024xf32>
  %1 = "tfl.batch_matmul"(%0, %cst) {adj_x = false, adj_y = false} : (tensor<30x1024xf32>, tensor<1024x8192xf32>) -> tensor<30x8192xf32>
  %2 = "tfl.reshape"(%1, %cst_0) : (tensor<30x8192xf32>, tensor<3xi32>) -> tensor<6x5x8192xf32>
  return %2 : tensor<6x5x8192xf32>
  // CHECK: %cst = arith.constant dense_resource<__elided__> : tensor<1024x8192xf32>
  // CHECK: %0 = "tfl.batch_matmul"(%arg0, %cst) <{adj_x = false, adj_y = false}> : (tensor<6x5x1024xf32>, tensor<1024x8192xf32>) -> tensor<6x5x8192xf32>
  // CHECK: return %0 : tensor<6x5x8192xf32>
}

// CHECK-LABEL: @FuseReshapeAroundBMMLHSNegative
func.func @FuseReshapeAroundBMMLHSNegative(%arg0: tensor<1x64xf32>, %arg1: tensor<1x64x1024xf32> ) -> (tensor<1x1024xf32> )  {
  %cst = arith.constant dense<[1, 1024]> : tensor<2xi32>
  %cst_0 = arith.constant dense<[1, 1, 64]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<1x64xf32>, tensor<3xi32>) -> tensor<1x1x64xf32>
  %1 = "tfl.batch_matmul"(%0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x1x64xf32>, tensor<1x64x1024xf32>) -> tensor<1x1x1024xf32>
  %2 = "tfl.reshape"(%1, %cst) : (tensor<1x1x1024xf32>, tensor<2xi32>) -> tensor<1x1024xf32>
  return %2 : tensor<1x1024xf32>
  // CHECK: %cst = arith.constant dense<[1, 1024]> : tensor<2xi32>
  // CHECK: %cst_0 = arith.constant dense<[1, 1, 64]> : tensor<3xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<1x64xf32>, tensor<3xi32>) -> tensor<1x1x64xf32>
  // CHECK: %1 = "tfl.batch_matmul"(%0, %arg1) <{adj_x = false, adj_y = false}> : (tensor<1x1x64xf32>, tensor<1x64x1024xf32>) -> tensor<1x1x1024xf32>
  // CHECK: %2 = "tfl.reshape"(%1, %cst) : (tensor<1x1x1024xf32>, tensor<2xi32>) -> tensor<1x1024xf32>
  // CHECK: return %2 : tensor<1x1024xf32>
}

// CHECK-LABEL: @FuseReshapeAroundBMMNagativeTest
func.func @FuseReshapeAroundBMMNagativeTest(%arg0: tensor<5x4x1x1024xf32>, %arg1: tensor<5x1024x8192xf32>) -> tensor<5x4x1x8192xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<3xi32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<4xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<5x4x1x1024xf32>, tensor<3xi32>) -> tensor<5x4x1024xf32>
  %1 = "tfl.batch_matmul"(%0, %arg1) {adj_x = false, adj_y = false} : (tensor<5x4x1024xf32>, tensor<5x1024x8192xf32>) -> tensor<5x4x8192xf32>
  %2 = "tfl.reshape"(%1, %cst_0) : (tensor<5x4x8192xf32>, tensor<4xi32>) -> tensor<5x4x1x8192xf32>
  return %2 : tensor<5x4x1x8192xf32>
  // CHECK: %cst = arith.constant dense_resource<__elided__> : tensor<3xi32>
  // CHECK: %cst_0 = arith.constant dense_resource<__elided__> : tensor<4xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst) : (tensor<5x4x1x1024xf32>, tensor<3xi32>) -> tensor<5x4x1024xf32>
  // CHECK: %1 = "tfl.batch_matmul"(%0, %arg1) <{adj_x = false, adj_y = false}> : (tensor<5x4x1024xf32>, tensor<5x1024x8192xf32>) -> tensor<5x4x8192xf32>
  // CHECK: %2 = "tfl.reshape"(%1, %cst_0) : (tensor<5x4x8192xf32>, tensor<4xi32>) -> tensor<5x4x1x8192xf32>
  // CHECK: return %2 : tensor<5x4x1x8192xf32>
}

// CHECK-LABEL: @FuseReshapeAroundBMMNagativeTest2
// Checks that the pattern matcher FuseReshapesAroundBatchMatMulLHS does not get
// applied for this case that does not pass the constraint around input rank.
func.func @FuseReshapeAroundBMMNagativeTest2(%arg0: tensor<2x1536xf32>) -> tensor<2x768xf32> {
  %cst = arith.constant dense_resource<__elided__> : tensor<3xi32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<2xi32>
  %402 = "tfl.reshape"(%arg0, %cst) : (tensor<2x1536xf32>, tensor<3xi32>) -> tensor<2x12x128xf32>
  %403 = "tfl.pseudo_qconst"() {qtype = tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>, value = dense<9> : tensor<128x64xi8>} : () -> tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>
  %404 = "tfl.batch_matmul"(%402, %403) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = true} : (tensor<2x12x128xf32>, tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>) -> tensor<2x12x64xf32>
  %405 = "tfl.reshape"(%404, %cst_0) : (tensor<2x12x64xf32>, tensor<2xi32>) -> tensor<2x768xf32>
  return %405 : tensor<2x768xf32>
  // CHECK: %cst = arith.constant dense_resource<__elided__> : tensor<3xi32>
  // CHECK: %cst_0 = arith.constant dense_resource<__elided__> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst) : (tensor<2x1536xf32>, tensor<3xi32>) -> tensor<2x12x128xf32>
  // CHECK: %1 = "tfl.pseudo_qconst"() <{qtype = tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>, value = dense<9> : tensor<128x64xi8>}> : () -> tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>
  // CHECK: %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = true}> : (tensor<2x12x128xf32>, tensor<128x64x!quant.uniform<i8:f32, 0.0047710379585623741>>) -> tensor<2x12x64xf32>
  // CHECK: %3 = "tfl.reshape"(%2, %cst_0) : (tensor<2x12x64xf32>, tensor<2xi32>) -> tensor<2x768xf32>
  // CHECK: return %3 : tensor<2x768xf32>
}

// CHECK-LABEL: @FuseReshapeAroundBMMNagativeTest3
// Checks that the pattern matcher FuseReshapesAroundBatchMatMulLHS does not get
// applied for this case that does not pass the constraint around input rank.
func.func @FuseReshapeAroundBMMNagativeTest3(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<10x5xf32>
  %cst_3 = arith.constant dense<5> : tensor<1xi32>
  %cst_4 = arith.constant dense<[1, 10]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst_4) : (tensor<10xf32>, tensor<2xi32>) -> tensor<1x10xf32>
  %1 = "tfl.batch_matmul"(%0, %cst_0) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x10xf32>, tensor<10x5xf32>) -> tensor<1x5xf32>
  %2 = "tfl.reshape"(%1, %cst_3) : (tensor<1x5xf32>, tensor<1xi32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>
  // CHECK: %cst = arith.constant dense_resource<__elided__> : tensor<10x5xf32>
  // CHECK: %cst_0 = arith.constant dense<5> : tensor<1xi32>
  // CHECK: %cst_1 = arith.constant dense<[1, 10]> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst_1) : (tensor<10xf32>, tensor<2xi32>) -> tensor<1x10xf32>
  // CHECK: %1 = "tfl.batch_matmul"(%0, %cst) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x10xf32>, tensor<10x5xf32>) -> tensor<1x5xf32>
  // CHECK: %2 = "tfl.reshape"(%1, %cst_0) : (tensor<1x5xf32>, tensor<1xi32>) -> tensor<5xf32>
  // CHECK: return %2 : tensor<5xf32>
}

// CHECK-LABEL: @convert_bmm_rhs_transpose_into_fc
// FOLD-LABEL: @convert_bmm_rhs_transpose_into_fc
func.func @convert_bmm_rhs_transpose_into_fc(%arg0: tensor<8x256xf32>, %arg1: tensor<256x256xf32>) -> (tensor<8x256xf32>) {
  %10 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %11 = "tfl.transpose"(%arg1, %10) : (tensor<256x256xf32>, tensor<2xi32>) -> tensor<256x256xf32>
  %14 = "tfl.batch_matmul"(%arg0, %11) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<8x256xf32>, tensor<256x256xf32>) -> tensor<8x256xf32>
  return %14 : tensor<8x256xf32>
  // CHECK:  %0 = "tfl.pseudo_const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK:  %1 = "tfl.transpose"(%arg1, %0) : (tensor<256x256xf32>, tensor<2xi32>) -> tensor<256x256xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%arg0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<8x256xf32>, tensor<256x256xf32>) -> tensor<8x256xf32>
  // CHECK:  return %2 : tensor<8x256xf32>

  // FOLD:  %0 = "tfl.no_value"() <{value}> : () -> none
  // FOLD:  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<8x256xf32>, tensor<256x256xf32>, none) -> tensor<8x256xf32>
  // FOLD:  return %1 : tensor<8x256xf32>
}

// CHECK-LABEL: @convert_bmm_rhs_transpose_into_fc_negative
func.func @convert_bmm_rhs_transpose_into_fc_negative(%arg0: tensor<2x1x256xf32>, %arg1: tensor<256x256x2xf32>) -> (tensor<2x1x256xf32>) {
  %10 = "tfl.pseudo_const"() <{value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %11 = "tfl.transpose"(%arg1, %10) : (tensor<256x256x2xf32>, tensor<3xi32>) -> tensor<2x256x256xf32>
  %14 = "tfl.batch_matmul"(%arg0, %11) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x1x256xf32>, tensor<2x256x256xf32>) -> tensor<2x1x256xf32>
  return %14 : tensor<2x1x256xf32>
  // CHECK:  %0 = "tfl.pseudo_const"() <{value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK:  %1 = "tfl.transpose"(%arg1, %0) : (tensor<256x256x2xf32>, tensor<3xi32>) -> tensor<2x256x256xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%arg0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<2x1x256xf32>, tensor<2x256x256xf32>) -> tensor<2x1x256xf32>
  // CHECK:  return %2 : tensor<2x1x256xf32>
}


// CHECK-LABEL: @convert_bmm_rhs_transpose_into_fc_negative_adjx_true
func.func @convert_bmm_rhs_transpose_into_fc_negative_adjx_true(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<16x16xf32>, tensor<2xi32>) -> tensor<16x16xf32>
  %1 = "tfl.batch_matmul"(%arg0, %0) <{adj_x = true, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %1 : tensor<16x16xf32>
  // CHECK:  %0 = "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = true, adj_y = true, asymmetric_quantize_inputs = false}> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  // CHECK:  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: @FuseBMMInputReshape_WithTwoRhsDims_DynamicShapedInput
func.func @FuseBMMInputReshape_WithTwoRhsDims_DynamicShapedInput(%arg0: tensor<8x1792x256xf32>, %arg1: tensor<?x?x1792xf32>) -> (tensor<?x?x8x256xf32>) {
  %cst = arith.constant dense<[1, 1, 8, 256]> : tensor<4xi32>
  %cst_0 = arith.constant dense<[1792, 2048]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, 1792]> : tensor<2xi32>
  %cst_2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst_2) : (tensor<8x1792x256xf32>, tensor<3xi32>) -> tensor<1792x8x256xf32>
  %1 = "tfl.reshape"(%arg1, %cst_1) : (tensor<?x?x1792xf32>, tensor<2xi32>) -> tensor<?x1792xf32>
  %2 = "tfl.reshape"(%0, %cst_0) : (tensor<1792x8x256xf32>, tensor<2xi32>) -> tensor<1792x2048xf32>
  %3 = "tfl.batch_matmul"(%1, %2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<?x1792xf32>, tensor<1792x2048xf32>) -> tensor<?x2048xf32>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<?x2048xf32>, tensor<4xi32>) -> tensor<?x?x8x256xf32>
  return %4 : tensor<?x?x8x256xf32>
  // CHECK:  %cst = arith.constant dense<[1, 1, 8, 256]> : tensor<4xi32>
  // CHECK:  %cst_0 = arith.constant dense<[1792, 2048]> : tensor<2xi32>
  // CHECK:  %cst_1 = arith.constant dense<[1, 1792]> : tensor<2xi32>
  // CHECK:  %cst_2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  // CHECK:  %0 = "tfl.transpose"(%arg0, %cst_2) : (tensor<8x1792x256xf32>, tensor<3xi32>) -> tensor<1792x8x256xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg1, %cst_1) : (tensor<?x?x1792xf32>, tensor<2xi32>) -> tensor<?x1792xf32>
  // CHECK:  %2 = "tfl.reshape"(%0, %cst_0) : (tensor<1792x8x256xf32>, tensor<2xi32>) -> tensor<1792x2048xf32>
  // CHECK:  %3 = "tfl.batch_matmul"(%1, %2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<?x1792xf32>, tensor<1792x2048xf32>) -> tensor<?x2048xf32>
  // CHECK:  %4 = "tfl.reshape"(%3, %cst) : (tensor<?x2048xf32>, tensor<4xi32>) -> tensor<?x?x8x256xf32>
  // CHECK:  return %4 : tensor<?x?x8x256xf32>
}

// CHECK-LABEL: @FuseBMMInputReshape_WithTwoRhsDims
func.func @FuseBMMInputReshape_WithTwoRhsDims(%arg0: tensor<8x1792x256xf32>, %arg1: tensor<1x1x1792xf32>) -> (tensor<1x1x8x256xf32>) {
  %cst = arith.constant dense<[1, 1, 8, 256]> : tensor<4xi32>
  %cst_0 = arith.constant dense<[1792, 2048]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, 1792]> : tensor<2xi32>
  %cst_2 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst_2) : (tensor<8x1792x256xf32>, tensor<3xi32>) -> tensor<1792x8x256xf32>
  %1 = "tfl.reshape"(%arg1, %cst_1) : (tensor<1x1x1792xf32>, tensor<2xi32>) -> tensor<1x1792xf32>
  %2 = "tfl.reshape"(%0, %cst_0) : (tensor<1792x8x256xf32>, tensor<2xi32>) -> tensor<1792x2048xf32>
  %3 = "tfl.batch_matmul"(%1, %2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x1792xf32>, tensor<1792x2048xf32>) -> tensor<1x2048xf32>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x2048xf32>, tensor<4xi32>) -> tensor<1x1x8x256xf32>
  return %4 : tensor<1x1x8x256xf32>
  // CHECK: %cst = arith.constant dense<[1, 1, 8, 256]> : tensor<4xi32>
  // CHECK: %cst_0 = arith.constant dense<[1792, 2048]> : tensor<2xi32>
  // CHECK: %cst_1 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  // CHECK: %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<8x1792x256xf32>, tensor<3xi32>) -> tensor<1792x8x256xf32>
  // CHECK: %1 = "tfl.reshape"(%0, %cst_0) : (tensor<1792x8x256xf32>, tensor<2xi32>) -> tensor<1792x2048xf32>
  // CHECK: %2 = "tfl.batch_matmul"(%arg1, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x1x1792xf32>, tensor<1792x2048xf32>) -> tensor<1x1x2048xf32>
  // CHECK: %3 = "tfl.reshape"(%2, %cst) : (tensor<1x1x2048xf32>, tensor<4xi32>) -> tensor<1x1x8x256xf32>
  // CHECK: return %3 : tensor<1x1x8x256xf32>
}

// CHECK-LABEL: @FuseBMMOutputReshape_WithTwoLHSContractionDims_DynamicShapedInput
func.func @FuseBMMOutputReshape_WithTwoLHSContractionDims_DynamicShapedInput(%arg0: tensor<8x256x1792xf32>, %arg1: tensor<?x?x8x256xf32>) -> (tensor<?x?x1792xf32>){
  %cst = arith.constant dense<[1, 128, 1792]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg1, %cst_1) : (tensor<?x?x8x256xf32>, tensor<2xi32>) -> tensor<?x2048xf32>
  %1 = "tfl.reshape"(%arg0, %cst_0) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<?x2048xf32>, tensor<2048x1792xf32>) -> tensor<?x1792xf32>
  %3 = "tfl.reshape"(%2, %cst) : (tensor<?x1792xf32>, tensor<3xi32>) -> tensor<?x?x1792xf32>
  return %3 : tensor<?x?x1792xf32>
  // CHECK:  %cst = arith.constant dense<[1, 128, 1792]> : tensor<3xi32>
  // CHECK:  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  // CHECK:  %cst_1 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg1, %cst_1) : (tensor<?x?x8x256xf32>, tensor<2xi32>) -> tensor<?x2048xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg0, %cst_0) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<?x2048xf32>, tensor<2048x1792xf32>) -> tensor<?x1792xf32>
  // CHECK:  %3 = "tfl.reshape"(%2, %cst) : (tensor<?x1792xf32>, tensor<3xi32>) -> tensor<?x?x1792xf32>
  // CHECK:  return %3 : tensor<?x?x1792xf32>
}

// CHECK-LABEL: @FuseBMMOutputReshape_WithTwoLHSContractionDims
func.func @FuseBMMOutputReshape_WithTwoLHSContractionDims(%arg0: tensor<8x256x1792xf32>, %arg1: tensor<1x128x8x256xf32>) -> (tensor<1x128x1792xf32>){
  %cst = arith.constant dense<[1, 128, 1792]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg1, %cst_1) : (tensor<1x128x8x256xf32>, tensor<2xi32>) -> tensor<128x2048xf32>
  %1 = "tfl.reshape"(%arg0, %cst_0) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  %2 = "tfl.batch_matmul"(%0, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x2048xf32>, tensor<2048x1792xf32>) -> tensor<128x1792xf32>
  %3 = "tfl.reshape"(%2, %cst) : (tensor<128x1792xf32>, tensor<3xi32>) -> tensor<1x128x1792xf32>
  return %3 : tensor<1x128x1792xf32>
  // CHECK:  %cst = arith.constant dense<[1, 128, 2048]> : tensor<3xi32>
  // CHECK:  %cst_0 = arith.constant dense<[2048, 1792]> : tensor<2xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<8x256x1792xf32>, tensor<2xi32>) -> tensor<2048x1792xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg1, %cst) : (tensor<1x128x8x256xf32>, tensor<3xi32>) -> tensor<1x128x2048xf32>
  // CHECK:  %2 = "tfl.batch_matmul"(%1, %0) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x128x2048xf32>, tensor<2048x1792xf32>) -> tensor<1x128x1792xf32>
  // CHECK:  return %2 : tensor<1x128x1792xf32>
}

// CHECK-LABEL: @FuseBMMOutputReshape_WithTwoLHSContractionDims_Negative
func.func @FuseBMMOutputReshape_WithTwoLHSContractionDims_Negative(%arg0: tensor<1x3872x1x128xf32>) -> tensor<1x3872x8x16xf32> {
  %cst_84 = arith.constant dense<[3872, 128]> : tensor<2xi32>
  %cst_82 = arith.constant dense<[1, 3872, 8, 16]> : tensor<4xi32>
  %cst_24 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
  %59 = "tfl.reshape"(%arg0, %cst_84) : (tensor<1x3872x1x128xf32>, tensor<2xi32>) -> tensor<3872x128xf32>
  %60 = "tfl.batch_matmul"(%59, %cst_24) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<3872x128xf32>, tensor<128x128xf32>) -> tensor<3872x128xf32>
  %67 = "tfl.reshape"(%60, %cst_82) : (tensor<3872x128xf32>, tensor<4xi32>) -> tensor<1x3872x8x16xf32>
  func.return %67: tensor<1x3872x8x16xf32>
  // CHECK:  %cst = arith.constant dense<[3872, 128]> : tensor<2xi32>
  // CHECK:  %cst_0 = arith.constant dense<[1, 3872, 8, 16]> : tensor<4xi32>
  // CHECK:  %cst_1 = arith.constant dense_resource<__elided__> : tensor<128x128xf32>
  // CHECK:  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<1x3872x1x128xf32>, tensor<2xi32>) -> tensor<3872x128xf32>
  // CHECK:  %1 = "tfl.batch_matmul"(%0, %cst_1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<3872x128xf32>, tensor<128x128xf32>) -> tensor<3872x128xf32>
  // CHECK:  %2 = "tfl.reshape"(%1, %cst_0) : (tensor<3872x128xf32>, tensor<4xi32>) -> tensor<1x3872x8x16xf32>
  // CHECK:  return %2 : tensor<1x3872x8x16xf32>
}

// CHECK-LABEL: @FuseBMMOutputReshape_WithThreeLHSContractionDims
func.func @FuseBMMOutputReshape_WithThreeLHSContractionDims(%arg0: tensor<2x8x256x1792xf32>, %arg1: tensor<1x2x128x8x256xf32>) -> (tensor<1x128x1792xf32>){
  %cst = arith.constant dense<[1, 128, 1792]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[4096, 1792]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[128, 4096]> : tensor<2xi32>
  %cst_2 = arith.constant dense<[0, 2, 1, 3, 4]> : tensor<5xi32>
  %0 = "tfl.transpose"(%arg1, %cst_2) : (tensor<1x2x128x8x256xf32>, tensor<5xi32>) -> tensor<1x128x2x8x256xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<1x128x2x8x256xf32>, tensor<2xi32>) -> tensor<128x4096xf32>
  %2 = "tfl.reshape"(%arg0, %cst_0) : (tensor<2x8x256x1792xf32>, tensor<2xi32>) -> tensor<4096x1792xf32>
  %3 = "tfl.batch_matmul"(%1, %2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x4096xf32>, tensor<4096x1792xf32>) -> tensor<128x1792xf32>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<128x1792xf32>, tensor<3xi32>) -> tensor<1x128x1792xf32>
  return %4 : tensor<1x128x1792xf32>
  // CHECK:  %cst = arith.constant dense<[1, 128, 4096]> : tensor<3xi32>
  // CHECK:  %cst_0 = arith.constant dense<[4096, 1792]> : tensor<2xi32>
  // CHECK:  %cst_1 = arith.constant dense<[0, 2, 1, 3, 4]> : tensor<5xi32>
  // CHECK:  %0 = "tfl.transpose"(%arg1, %cst_1) : (tensor<1x2x128x8x256xf32>, tensor<5xi32>) -> tensor<1x128x2x8x256xf32>
  // CHECK:  %1 = "tfl.reshape"(%arg0, %cst_0) : (tensor<2x8x256x1792xf32>, tensor<2xi32>) -> tensor<4096x1792xf32>
  // CHECK:  %2 = "tfl.reshape"(%0, %cst) : (tensor<1x128x2x8x256xf32>, tensor<3xi32>) -> tensor<1x128x4096xf32>
  // CHECK:  %3 = "tfl.batch_matmul"(%2, %1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x128x4096xf32>, tensor<4096x1792xf32>) -> tensor<1x128x1792xf32>
  // CHECK:  return %3 : tensor<1x128x1792xf32>
}

// CHECK-LABEL: @FuseReshapeAroundBMMRHS
func.func @FuseReshapeAroundBMMRHS(%arg0: tensor<1x3x6x5x1024xf32>) -> tensor<1x3x6x5x8192xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "inputs", outputs = "Identity_1"}} {
  %cst = arith.constant dense_resource<__elided__> : tensor<1x1024x8192xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<5xi32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst_1) : (tensor<1x3x6x5x1024xf32>, tensor<3xi32>) -> tensor<1x90x1024xf32>
  %1 = "tfl.batch_matmul"(%0, %cst) {adj_x = false, adj_y = false} : (tensor<1x90x1024xf32>, tensor<1x1024x8192xf32>) -> tensor<1x90x8192xf32>
  %2 = "tfl.reshape"(%1, %cst_0) : (tensor<1x90x8192xf32>, tensor<5xi32>) -> tensor<1x3x6x5x8192xf32>
  return %2 : tensor<1x3x6x5x8192xf32>
  // CHECK: %cst = arith.constant dense_resource<__elided__> : tensor<1x1024x8192xf32>
  // CHECK: %0 = "tfl.batch_matmul"(%arg0, %cst) <{adj_x = false, adj_y = false}> : (tensor<1x3x6x5x1024xf32>, tensor<1x1024x8192xf32>) -> tensor<1x3x6x5x8192xf32>
  // CHECK: return %0 : tensor<1x3x6x5x8192xf32>
}

// CHECK-LABEL: @FuseTransposeIntoBMM_LHS
func.func @FuseTransposeIntoBMM_LHS(%arg0: tensor<1x4x1440x256xf32>, %arg1: tensor<1x1440x256xf32>) -> tensor<1x4x256x256xf32> {
  %cst_1 = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
  %32 = "tfl.transpose"(%arg1, %cst_1) : (tensor<1x1440x256xf32>, tensor<3xi32>) -> tensor<1x256x1440xf32>
  %33 = "tfl.batch_matmul"(%32, %arg0) {adj_x = false, adj_y = false} : (tensor<1x256x1440xf32>, tensor<1x4x1440x256xf32>) -> tensor<1x4x256x256xf32>
  return %33 : tensor<1x4x256x256xf32>
  // CHECK: %0 = "tfl.batch_matmul"(%arg1, %arg0) <{adj_x = true, adj_y = false}> : (tensor<1x1440x256xf32>, tensor<1x4x1440x256xf32>) -> tensor<1x4x256x256xf32>
  // CHECK: return %0 : tensor<1x4x256x256xf32>
}

// CHECK-LABEL: @FuseSqueezingReshapesAroundBMM
func.func @FuseSqueezingReshapesAroundBMM(%arg0: tensor<1x128x8x256xf32>) -> (tensor<1x128xf32>) {
  %cst_2 = arith.constant dense<9.0> : tensor<2048x1792xf32>
  %cst_14 = arith.constant dense<[1, 128, 1792]> : tensor<3xi32>
  %cst_15 = arith.constant dense<[128, 2048]> : tensor<2xi32>
  %cst_16 = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %cst_15) : (tensor<1x128x8x256xf32>, tensor<2xi32>) -> tensor<128x2048xf32>
  %1 = "tfl.batch_matmul"(%0, %cst_2) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<128x2048xf32>, tensor<2048x1792xf32>) -> tensor<128x1792xf32>
  %2 = "tfl.reshape"(%1, %cst_14) : (tensor<128x1792xf32>, tensor<3xi32>) -> tensor<1x128x1792xf32>
  %3 = tfl.mul %2, %2 {fused_activation_function = "NONE"} : tensor<1x128x1792xf32>
  %4 = "tfl.sum"(%3, %cst_16) <{keep_dims = false}> : (tensor<1x128x1792xf32>, tensor<1xi32>) -> tensor<1x128xf32>
  return %4 : tensor<1x128xf32>
  // CHECK:  %cst = arith.constant dense<9.000000e+00> : tensor<2048x1792xf32>
  // CHECK:  %cst_0 = arith.constant dense<2> : tensor<1xi32>
  // CHECK:  %cst_1 = arith.constant dense<[1, 128, 2048]> : tensor<3xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg0, %cst_1) : (tensor<1x128x8x256xf32>, tensor<3xi32>) -> tensor<1x128x2048xf32>
  // CHECK:  %1 = "tfl.batch_matmul"(%0, %cst) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x128x2048xf32>, tensor<2048x1792xf32>) -> tensor<1x128x1792xf32>
  // CHECK:  %2 = tfl.mul %1, %1 {fused_activation_function = "NONE"} : tensor<1x128x1792xf32>
  // CHECK:  %3 = "tfl.sum"(%2, %cst_0) <{keep_dims = false}> : (tensor<1x128x1792xf32>, tensor<1xi32>) -> tensor<1x128xf32>
  // CHECK:  return %3 : tensor<1x128xf32>
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAddConst
// FOLD-LABEL: @FuseFullyConnectedReshapeAddConst
func.func @FuseFullyConnectedReshapeAddConst(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40x40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<40xf32>
  %shape1 = arith.constant dense<[1, 40, 40]> : tensor<3xi32>
  %shape2 = arith.constant dense<[40, 40]> : tensor<2xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40x40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<40x40xf32>, tensor<3xi32>) -> tensor<1x40x40xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x40xf32>, tensor<40xf32>) -> tensor<1x40x40xf32>
  %3 = "tfl.reshape"(%2, %shape2) : (tensor<1x40x40xf32>, tensor<2xi32>) -> tensor<40x40xf32>

  func.return %3 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: %[[rs2:.*]] = "tfl.reshape"(%[[rs1]]
  // CHECK: return %[[rs2]]

  // FOLD: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // FOLD: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // FOLD: return %[[fc]]
}

// CHECK-LABEL: @RemoveRedundantReshapeUsedAsInputToBinaryOp
func.func @RemoveRedundantReshapeUsedAsInputToBinaryOp(%arg0: tensor<128xf32>, %arg1: tensor<1x512x512x128xf32>, %arg2: tensor<1x512x512x128xf32>) -> (tensor<1x512x512x128xf32>, tensor<1x512x512x128xf32>) {
  %cst_10 = arith.constant  dense<[1, 1, 1, 128]> : tensor<4xi32>

  %894 = "tfl.reshape"(%arg0, %cst_10) : (tensor<128xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
  %895 = "tfl.mul"(%894, %arg1) {fused_activation_function = "NONE"} : (tensor<1x1x1x128xf32>, tensor<1x512x512x128xf32>) -> tensor<1x512x512x128xf32>
  %896 = "tfl.mul"(%arg2, %894) {fused_activation_function = "NONE"} : (tensor<1x512x512x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x512x512x128xf32>

  return %895, %896 : tensor<1x512x512x128xf32>, tensor<1x512x512x128xf32>

  // CHECK:  %0 = tfl.mul(%arg0, %arg1)
  // CHECK:  %1 = tfl.mul(%arg2, %arg0)
  // CHECK:  return %0, %1
}

// CHECK-LABEL: @RetainRedundantReshapeUseInNonBinaryOp
func.func @RetainRedundantReshapeUseInNonBinaryOp(%arg0: tensor<128xf32>, %arg1: tensor<1x512x512x128xf32>, %arg2: tensor<1x512x512x128xf32>) -> (tensor<1x512x512x128xf32>, tensor<128xf32>) {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %cst_10 = arith.constant  dense<[1, 1, 1, 128]> : tensor<4xi32>
  %894 = "tfl.reshape"(%arg0, %cst_10) : (tensor<128xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
  %895 = "tfl.mul"(%894, %arg1) {fused_activation_function = "NONE"} : (tensor<1x1x1x128xf32>, tensor<1x512x512x128xf32>) -> tensor<1x512x512x128xf32>
  %896 = "tfl.reduce_max"(%894, %cst) {keep_dims = false} : (tensor<1x1x1x128xf32>, tensor<1xi32>) -> tensor<128xf32>
  return %895, %896 : tensor<1x512x512x128xf32>, tensor<128xf32>

  // CHECK-DAG: %cst = arith.constant dense<0> : tensor<1xi32>
  // CHECK-DAG: %cst_0 = arith.constant dense<[1, 1, 1, 128]> : tensor<4xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<128xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
  // CHECK: %1 = tfl.mul(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x128xf32>, tensor<1x512x512x128xf32>) -> tensor<1x512x512x128xf32>
  // CHECK: %2 = "tfl.reduce_max"(%0, %cst) <{keep_dims = false}> : (tensor<1x1x1x128xf32>, tensor<1xi32>) -> tensor<128xf32>
  // CHECK: return %1, %2
}

// CHECK-LABEL: @FuseTransposeReshapeTranspose
func.func @FuseTransposeReshapeTranspose(%arg0: tensor<1x16x256xf32>) -> tensor<16x256xf32> {
  %cst_10 = arith.constant dense<[0, 2, 1]> : tensor<3xi32>
  %cst_3 = arith.constant dense<[256, 16]> : tensor<2xi32>
  %cst_6 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %2057 = "tfl.transpose"(%arg0, %cst_10) : (tensor<1x16x256xf32>, tensor<3xi32>) -> tensor<1x256x16xf32>
  %2058 = "tfl.reshape"(%2057, %cst_3) : (tensor<1x256x16xf32>, tensor<2xi32>) -> tensor<256x16xf32>
  %2059 = "tfl.transpose"(%2058, %cst_6) : (tensor<256x16xf32>, tensor<2xi32>) -> tensor<16x256xf32>
  return %2059: tensor<16x256xf32>
  // CHECK-DAG: %cst = arith.constant dense<[16, 256]> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst) : (tensor<1x16x256xf32>, tensor<2xi32>) -> tensor<16x256xf32>
  // CHECK: return %0
}

// CHECK-LABEL: @FoldDoubleTranspose
func.func @FoldDoubleTranspose(%arg0: tensor<1x4x1440x256xf32>) -> tensor<1x1440x256x4xf32> {
    %cst_12 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %cst_18 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %2112 = "tfl.transpose"(%arg0, %cst_18) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x4x256xf32>
    %2114 = "tfl.transpose"(%2112, %cst_12) : (tensor<1x1440x4x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
    return %2114 : tensor<1x1440x256x4xf32>
  // CHECK-DAG: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  // CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
  // CHECK: return %0
}

// CHECK-LABEL: @FoldMultpleTranspose
func.func @FoldMultpleTranspose(%arg0: tensor<1x4x1440x256xf32>) -> tensor<1x256x4x1440xf32> {
    %cst_11 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %cst_12 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %cst_18 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %2112 = "tfl.transpose"(%arg0, %cst_11) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x1440x256x4xf32>
    %2113 = "tfl.transpose"(%2112, %cst_18) : (tensor<1x1440x256x4xf32>, tensor<4xi32>) -> tensor<1x256x1440x4xf32>
    %2114 = "tfl.transpose"(%2113, %cst_12) : (tensor<1x256x1440x4xf32>, tensor<4xi32>) -> tensor<1x256x4x1440xf32>
    return %2114 : tensor<1x256x4x1440xf32>
  // CHECK-DAG: %cst = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  // CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x4x1440x256xf32>, tensor<4xi32>) -> tensor<1x256x4x1440xf32>
  // CHECK: return %0
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAddConstWithOptionalAttribute
// FOLD-LABEL: @FuseFullyConnectedReshapeAddConstWithOptionalAttribute
func.func @FuseFullyConnectedReshapeAddConstWithOptionalAttribute(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40x40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<40xf32>
  %shape1 = arith.constant dense<[1, 40, 40]> : tensor<3xi32>
  %shape2 = arith.constant dense<[40, 40]> : tensor<2xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40x40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<40x40xf32>, tensor<3xi32>) -> tensor<1x40x40xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x40xf32>, tensor<40xf32>) -> tensor<1x40x40xf32>
  %3 = "tfl.reshape"(%2, %shape2) : (tensor<1x40x40xf32>, tensor<2xi32>) -> tensor<40x40xf32>

  func.return %3 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{asymmetric_quantize_inputs = true,

  // FOLD: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // FOLD: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{asymmetric_quantize_inputs = true,
}

// CHECK-LABEL: @MoveReshapeAfterFullyConnected
// FOLD-LABEL: @MoveReshapeAfterFullyConnected
func.func @MoveReshapeAfterFullyConnected(%arg0: tensor<4x4x10xf32>)->(tensor<16x20xf32>) {
  %0 = "tfl.no_value"() <{value}> : () -> none
  %1 = "tfl.pseudo_const"() <{value = dense<[2, 2, 4, 10]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %2 = "tfl.reshape"(%arg0, %1) : (tensor<4x4x10xf32>, tensor<4xi32>) -> tensor<2x2x4x10xf32>
  %3 = "tfl.pseudo_const"() <{value = dense<9.0> : tensor<20x10xf32>}> : () -> tensor<20x10xf32>
  %4 = "tfl.fully_connected"(%2, %3, %0) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> :(tensor<2x2x4x10xf32>, tensor<20x10xf32>, none) -> tensor<2x2x4x20xf32>
  %5 = "tfl.pseudo_const"() <{value = dense<[16, 20]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %6 = "tfl.reshape"(%4, %5) : (tensor<2x2x4x20xf32>, tensor<2xi32>) -> tensor<16x20xf32>
  return %6 : tensor<16x20xf32>
  // CHECK-DAG:  %[[S0:.*]] = arith.constant dense<[16, 10]> : tensor<2xi32>
  // CHECK:  %[[FILTER:.*]] = "tfl.pseudo_const"() <{value = dense<9.000000e+00> : tensor<20x10xf32>}> : () -> tensor<20x10xf32>
  // CHECK:  %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %[[S1:.*]] = "tfl.pseudo_const"() <{value = dense<[2, 2, 4, 10]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:  %[[R0:.*]] = "tfl.reshape"(%arg0, %[[S1]]) : (tensor<4x4x10xf32>, tensor<4xi32>) -> tensor<2x2x4x10xf32>
  // CHECK:  %[[R1:.*]] = "tfl.reshape"(%[[R0]], %[[S0]]) : (tensor<2x2x4x10xf32>, tensor<2xi32>) -> tensor<16x10xf32>
  // CHECK:  %[[RESULT:.*]] = "tfl.fully_connected"(%[[R1]], %[[FILTER]], %[[BIAS]]) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<16x10xf32>, tensor<20x10xf32>, none) -> tensor<16x20xf32>
  // CHECK:  return %[[RESULT]] : tensor<16x20xf32>

  // FOLD-DAG:  %[[FILTER:.*]] =  arith.constant dense<9.000000e+00> : tensor<20x10xf32>
  // FOLD:  %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // FOLD:  %[[SHAPE:.*]] = arith.constant dense<[16, 10]> : tensor<2xi32>
  // FOLD:  %[[INPUT:.*]] = "tfl.reshape"(%arg0, %[[SHAPE]]) : (tensor<4x4x10xf32>, tensor<2xi32>) -> tensor<16x10xf32>
  // FOLD:  %[[RESULT:.*]] = "tfl.fully_connected"(%[[INPUT]], %[[FILTER]], %[[BIAS]]) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<16x10xf32>, tensor<20x10xf32>, none) -> tensor<16x20xf32>
  // FOLD:  return %[[RESULT]] : tensor<16x20xf32>
}

// CHECK-LABEL: @EnableFullyConnectedKeepNumDimsBeforeReshape
func.func @EnableFullyConnectedKeepNumDimsBeforeReshape(%arg0: tensor<2x2x4x10xf32>)->(tensor<1x320xf32>) {
  %0 = "tfl.no_value"() <{value}> : () -> none
  %1 = "tfl.pseudo_const"() <{value = dense<9.0> : tensor<20x10xf32>}> : () -> tensor<20x10xf32>
  %2 = "tfl.fully_connected"(%arg0, %1, %0) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> :(tensor<2x2x4x10xf32>, tensor<20x10xf32>, none) -> tensor<16x20xf32>
  %3 = "tfl.pseudo_const"() <{value = dense<[1, 320]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %4 = "tfl.reshape"(%2, %3) : (tensor<16x20xf32>, tensor<2xi32>) -> tensor<1x320xf32>
  return %4 : tensor<1x320xf32>
  // CHECK-DAG:  %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 320]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK:  %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %[[FILTER:.*]] = "tfl.pseudo_const"() <{value = dense<9.000000e+00> : tensor<20x10xf32>}> : () -> tensor<20x10xf32>
  // CHECK:  %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[FILTER]], %[[BIAS]]) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<2x2x4x10xf32>, tensor<20x10xf32>, none) -> tensor<2x2x4x20xf32>
  // CHECK:  %[[RESULT:.*]] = "tfl.reshape"(%[[FC]], %[[CST]]) : (tensor<2x2x4x20xf32>, tensor<2xi32>) -> tensor<1x320xf32>
  // CHECK:  return %[[RESULT]] : tensor<1x320xf32>
}

// CHECK-LABEL: @fuse_fc_and_lhs_reshape_dynamic_shape_input
func.func @fuse_fc_and_lhs_reshape_dynamic_shape_input(%arg0: tensor<?x?x14336xf32>) -> tensor<?x1792xf32> {
  %3 = "tfl.no_value"() <{value}> : () -> none
  %41 = "tfl.pseudo_const"() <{value = dense<[128, 14336]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %42 = "tfl.reshape"(%arg0, %41) : (tensor<?x?x14336xf32>, tensor<2xi32>) -> tensor<?x14336xf32>
  %43 = "tfl.pseudo_const"() <{value = dense<9.0> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  %44 = "tfl.fully_connected"(%42, %43, %3) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<?x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<?x1792xf32>
  return %44 : tensor<?x1792xf32>
  // CHECK:  %0 = "tfl.pseudo_const"() <{value = dense<9.000000e+00> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  // CHECK:  %1 = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %2 = "tfl.pseudo_const"() <{value = dense<[128, 14336]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK:  %3 = "tfl.reshape"(%arg0, %2) : (tensor<?x?x14336xf32>, tensor<2xi32>) -> tensor<?x14336xf32>
  // CHECK:  %4 = "tfl.fully_connected"(%3, %0, %1) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<?x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<?x1792xf32>
  // CHECK:  return %4 : tensor<?x1792xf32>
}

// CHECK-LABEL: @fuse_fc_and_lhs_reshape
// FOLD-LABEL: @fuse_fc_and_lhs_reshape
func.func @fuse_fc_and_lhs_reshape(%arg0: tensor<1x128x14336xf32>) -> tensor<128x1792xf32> {
  %3 = "tfl.no_value"() <{value}> : () -> none
  %41 = "tfl.pseudo_const"() <{value = dense<[128, 14336]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %42 = "tfl.reshape"(%arg0, %41) : (tensor<1x128x14336xf32>, tensor<2xi32>) -> tensor<128x14336xf32>
  %43 = "tfl.pseudo_const"() <{value = dense<9.0> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  %44 = "tfl.fully_connected"(%42, %43, %3) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<128x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<128x1792xf32>
  return %44 : tensor<128x1792xf32>
  // CHECK:  %0 = "tfl.pseudo_const"() <{value = dense<9.000000e+00> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  // CHECK:  %1 = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %2 = "tfl.pseudo_const"() <{value = dense<[128, 14336]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK:  %3 = "tfl.reshape"(%arg0, %2) : (tensor<1x128x14336xf32>, tensor<2xi32>) -> tensor<128x14336xf32>
  // CHECK:  %4 = "tfl.fully_connected"(%3, %0, %1) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<128x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<128x1792xf32>
  // CHECK:  return %4 : tensor<128x1792xf32>

  //FOLD:  %cst = arith.constant dense<9.000000e+00> : tensor<1792x14336xf32>
  //FOLD:  %0 = "tfl.no_value"() <{value}> : () -> none
  //FOLD:  %1 = "tfl.fully_connected"(%arg0, %cst, %0) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x128x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<128x1792xf32>
  //FOLD:  return %1 : tensor<128x1792xf32>
}

// CHECK-LABEL: @fuse_fc_and_lhs_reshape_negative
func.func @fuse_fc_and_lhs_reshape_negative(%arg0: tensor<1x2x3x128x14336xf32>)->(tensor<2x3x128x1792xf32>){
  %3 = "tfl.no_value"() <{value}> : () -> none
  %41 = "tfl.pseudo_const"() <{value = dense<[2, 3, 128, 14336]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %42 = "tfl.reshape"(%arg0, %41) : (tensor<1x2x3x128x14336xf32>, tensor<4xi32>) -> tensor<2x3x128x14336xf32>
  %43 = "tfl.pseudo_const"() <{value = dense<9.0> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  %44 = "tfl.fully_connected"(%42, %43, %3) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<2x3x128x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<2x3x128x1792xf32>
  return %44 : tensor<2x3x128x1792xf32>
  // CHECK:  %0 = "tfl.pseudo_const"() <{value = dense<9.000000e+00> : tensor<1792x14336xf32>}> : () -> tensor<1792x14336xf32>
  // CHECK:  %1 = "tfl.no_value"() <{value}> : () -> none
  // CHECK:  %2 = "tfl.pseudo_const"() <{value = dense<[2, 3, 128, 14336]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:  %3 = "tfl.reshape"(%arg0, %2) : (tensor<1x2x3x128x14336xf32>, tensor<4xi32>) -> tensor<2x3x128x14336xf32>
  // CHECK:  %4 = "tfl.fully_connected"(%3, %0, %1) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<2x3x128x14336xf32>, tensor<1792x14336xf32>, none) -> tensor<2x3x128x1792xf32>
  // CHECK:  return %4 : tensor<2x3x128x1792xf32>
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAddConstWithActivation
// FOLD-LABEL: @FuseFullyConnectedReshapeAddConstWithActivation
func.func @FuseFullyConnectedReshapeAddConstWithActivation(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<3.0> : tensor<40x40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<40xf32>
  %shape1 = arith.constant dense<[1, 40, 40]> : tensor<3xi32>
  %shape2 = arith.constant dense<[40, 40]> : tensor<2xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40x40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<40x40xf32>, tensor<3xi32>) -> tensor<1x40x40xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x40x40xf32>, tensor<40xf32>) -> tensor<1x40x40xf32>
  %3 = "tfl.reshape"(%2, %shape2) : (tensor<1x40x40xf32>, tensor<2xi32>) -> tensor<40x40xf32>

  func.return %3 : tensor<40x40xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: %[[rs2:.*]] = "tfl.reshape"(%[[rs1]]
  // CHECK: return %[[rs2]]

  // FOLD: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40x40xf32>
  // FOLD: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
  // FOLD: return %[[fc]]
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAdd2DConst
func.func @FuseFullyConnectedReshapeAdd2DConst(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40x4x10xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<4x10xf32>
  %shape = arith.constant dense<[1, 40, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x40xf32>, tensor<4xi32>) -> tensor<1x40x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x4x10xf32>, tensor<4x10xf32>) -> tensor<1x40x4x10xf32>

  func.return %2 : tensor<1x40x4x10xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: return %[[rs]]
}

// CHECK-LABEL: @FuseFCReshapeAdd2DConst2
func.func @FuseFCReshapeAdd2DConst2(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40x4x10xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<1x1x4x10xf32>
  %shape = arith.constant dense<[1, 40, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x40xf32>, tensor<4xi32>) -> tensor<1x40x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x4x10xf32>, tensor<1x1x4x10xf32>) -> tensor<1x40x4x10xf32>

  func.return %2 : tensor<1x40x4x10xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: return %[[rs]]
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAdd2DConstWithActivation
func.func @FuseFullyConnectedReshapeAdd2DConstWithActivation(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40x4x10xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<4x10xf32>
  %shape = arith.constant dense<[1, 40, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x40xf32>, tensor<4xi32>) -> tensor<1x40x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x40x4x10xf32>, tensor<4x10xf32>) -> tensor<1x40x4x10xf32>

  func.return %2 : tensor<1x40x4x10xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: return %[[rs]]
}

// CHECK-LABEL: @FuseFCReshapeAdd2DConstWithActvtn2
func.func @FuseFCReshapeAdd2DConstWithActvtn2(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40x4x10xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<1x1x4x10xf32>
  %shape = arith.constant dense<[1, 40, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x40xf32>, tensor<4xi32>) -> tensor<1x40x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "RELU6"} : (tensor<1x40x4x10xf32>, tensor<1x1x4x10xf32>) -> tensor<1x40x4x10xf32>

  func.return %2 : tensor<1x40x4x10xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<2.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]]) <{fused_activation_function = "RELU6", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: return %[[rs]]
}

// CHECK-LABEL: @FuseFullyConnectedReshapeAdd2DConstWithExistingBias
func.func @FuseFullyConnectedReshapeAdd2DConstWithExistingBias(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40x4x10xf32> {
  %cst = arith.constant dense<3.0> : tensor<40xf32>
  %cst2 = arith.constant dense<2.0> : tensor<4x10xf32>
  %shape = arith.constant dense<[1, 40, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> (tensor<40x40xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x40xf32>, tensor<4xi32>) -> tensor<1x40x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x40x4x10xf32>, tensor<4x10xf32>) -> tensor<1x40x4x10xf32>

  func.return %2 : tensor<1x40x4x10xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<40xf32>
  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %[[cst]])
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: return %[[rs]]
}

// CHECK-LABEL: @NotFuseFullyConnectedReshapeAdd2DConstIfLastDimIsNotNumElementsOfRhs
func.func @NotFuseFullyConnectedReshapeAdd2DConstIfLastDimIsNotNumElementsOfRhs(%arg0: tensor<40x37xf32>, %arg1: tensor<20x37xf32>) -> tensor<1x20x4x10xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %cst2 = arith.constant dense<2.0> : tensor<4x10xf32>
  %shape = arith.constant dense<[1, 20, 4, 10]> : tensor<4xi32>

  %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<20x37xf32>, none) -> (tensor<40x20xf32>)
  %1 = "tfl.reshape"(%0, %shape) : (tensor<40x20xf32>, tensor<4xi32>) -> tensor<1x20x4x10xf32>
  %2 = "tfl.add"(%1, %cst2) {fused_activation_function = "NONE"} : (tensor<1x20x4x10xf32>, tensor<4x10xf32>) -> tensor<1x20x4x10xf32>

  func.return %2 : tensor<1x20x4x10xf32>

  // CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1
  // CHECK: %[[rs:.*]] = "tfl.reshape"(%[[fc]]
  // CHECK: %[[add:.*]] = tfl.add(%[[rs]]
  // CHECK: return %[[add]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfNotBroadcastableAfter
func.func @NotReorderReshapeAddIfNotBroadcastableAfter(%arg0: tensor<40x10x4xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<2.0> : tensor<40xf32>
  %shape = arith.constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x10x4xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<40xf32>) -> tensor<40x40xf32>
  func.return %2 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.add(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfNotTailingDimAfter
func.func @NotReorderReshapeAddIfNotTailingDimAfter(%arg0: tensor<1x30x1x96xf32>) -> tensor<1x30x96xf32> {
  %cst = arith.constant dense<2.0> : tensor<1x30x96xf32>
  %shape = arith.constant dense<[1, 30, 96]> : tensor<3xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<1x30x1x96xf32>, tensor<3xi32>) -> tensor<1x30x96xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<1x30x96xf32>, tensor<1x30x96xf32>) -> tensor<1x30x96xf32>
  func.return %2 : tensor<1x30x96xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.add %[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAddIf5DInputs
func.func @NotReorderReshapeAddIf5DInputs(%arg0: tensor<2x1x1x1x1xf32>) -> tensor<1x1x1x1x2xf32> {
  %cst = arith.constant dense<2.0> : tensor<1x1x1x1x2xf32>
  %shape = arith.constant dense<[1, 1, 1, 1, 2]> : tensor<5xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<2x1x1x1x1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x2xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<1x1x1x1x2xf32>, tensor<1x1x1x1x2xf32>) -> tensor<1x1x1x1x2xf32>
  func.return %2 : tensor<1x1x1x1x2xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.add %[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeFloorDivIf5DInputs
func.func @NotReorderReshapeFloorDivIf5DInputs(%arg0: tensor<2x1x1x1x1xf32>) -> tensor<1x1x1x1x2xf32> {
  %cst = arith.constant dense<2.0> : tensor<1x1x1x1x2xf32>
  %shape = arith.constant dense<[1, 1, 1, 1, 2]> : tensor<5xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<2x1x1x1x1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x2xf32>
  %2 = "tfl.floor_div"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<1x1x1x1x2xf32>, tensor<1x1x1x1x2xf32>) -> tensor<1x1x1x1x2xf32>
  func.return %2 : tensor<1x1x1x1x2xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.floor_div %[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfNotTailingDim
func.func @NotReorderReshapeAddIfNotTailingDim(%arg0: tensor<40x40x1xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<2.0> : tensor<1x40xf32>
  %shape = arith.constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x40x1xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<40x40xf32>, tensor<1x40xf32>) -> tensor<40x40xf32>
  func.return %2 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.add(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAddIfHighDim
func.func @NotReorderReshapeAddIfHighDim(%arg0: tensor<1x1x1x1x30x96xf32>) -> tensor<1x30x96xf32> {
  %cst = arith.constant dense<2.0> : tensor<f32>
  %shape = arith.constant dense<[1, 30, 96]> : tensor<3xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<1x1x1x1x30x96xf32>, tensor<3xi32>) -> tensor<1x30x96xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<1x30x96xf32>, tensor<f32>) -> tensor<1x30x96xf32>
  func.return %2 : tensor<1x30x96xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = tfl.add(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @NotReorderReshapeAdd2DConstIfInputIsNotDefinedByFullyConnected
func.func @NotReorderReshapeAdd2DConstIfInputIsNotDefinedByFullyConnected(%arg0: tensor<8x15xf32>) -> tensor<1x8x3x5xf32> {
  %cst = arith.constant dense<2.0> : tensor<3x5xf32>
  %shape = arith.constant dense<[1, 8, 3, 5]> : tensor<4xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<8x15xf32>, tensor<4xi32>) -> tensor<1x8x3x5xf32>
  %2 = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<1x8x3x5xf32>, tensor<3x5xf32>) -> tensor<1x8x3x5xf32>
  func.return %2 : tensor<1x8x3x5xf32>

  // CHECK: %[[rs:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[add:.*]] = tfl.add(%[[rs]]
  // CHECK: return %[[add]]
}

// CHECK-LABEL: @ReorderElementwiseValueOpAndMoveOp
func.func @ReorderElementwiseValueOpAndMoveOp(%arg0: tensor<40x40x1xf32>) -> tensor<40x40xf32> {
  %shape = arith.constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x40x1xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.relu"(%1) : (tensor<40x40xf32>) -> tensor<40x40xf32>
  func.return %2 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.relu"(%arg0
  // CHECK: %[[rs2:.*]] = "tfl.reshape"(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @MinimumOfReluAnd6ToRelu6
func.func @MinimumOfReluAnd6ToRelu6(%arg0: tensor<40x40xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<6.0> : tensor<f32>
  %2 = "tfl.relu"(%arg0) : (tensor<40x40xf32>) -> tensor<40x40xf32>
  %3 = "tfl.minimum"(%2, %cst) : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>
  func.return %3 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.relu6"(%arg0
  // CHECK: return %[[rs1]]
}

// CHECK-LABEL: @MinimumOfReluAnd6ToRelu6_2
func.func @MinimumOfReluAnd6ToRelu6_2(%arg0: tensor<40x40xf32>) -> tensor<40x40xf32> {
  %cst = arith.constant dense<6.0> : tensor<f32>
  %2 = "tfl.minimum"(%arg0, %cst) : (tensor<40x40xf32>, tensor<f32>) -> tensor<40x40xf32>
  %3 = "tfl.relu"(%2) : (tensor<40x40xf32>) -> tensor<40x40xf32>

  func.return %3 : tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.relu6"(%arg0
  // CHECK: return %[[rs1]]
}

// CHECK-LABEL: @NotReorderElementwiseValueOpAndMoveOp
func.func @NotReorderElementwiseValueOpAndMoveOp(%arg0: tensor<40x40x1xf32>) -> (tensor<40x40xf32>, tensor<40x40xf32>) {
  %shape = arith.constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x40x1xf32>, tensor<2xi32>) -> tensor<40x40xf32>
  %2 = "tfl.relu"(%1) : (tensor<40x40xf32>) -> tensor<40x40xf32>
  func.return %1, %2 : tensor<40x40xf32>, tensor<40x40xf32>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = "tfl.relu"(%[[rs1]]
  // CHECK: return %[[rs1]], %[[rs2]]
}

// CHECK-LABEL: @NotReorderElementwiseValueOpAndMoveOpDifferentQuantParams
func.func @NotReorderElementwiseValueOpAndMoveOpDifferentQuantParams(%arg0: tensor<40x40x1x!quant.uniform<u8:f32, 0.024701418355107307:175>>) -> (tensor<40x40x!quant.uniform<u8:f32, 3.906250e-03>>) {
  %shape = arith.constant dense<[40, 40]> : tensor<2xi32>
  %1 = "tfl.reshape"(%arg0, %shape) : (tensor<40x40x1x!quant.uniform<u8:f32, 0.024701418355107307:175>>, tensor<2xi32>) -> tensor<40x40x!quant.uniform<u8:f32, 0.024701418355107307:175>>
  %2 = "tfl.logistic"(%1) : (tensor<40x40x!quant.uniform<u8:f32, 0.024701418355107307:175>>) -> tensor<40x40x!quant.uniform<u8:f32, 3.906250e-03>>
  func.return %2 : tensor<40x40x!quant.uniform<u8:f32, 3.906250e-03>>

  // CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
  // CHECK: %[[rs2:.*]] = "tfl.logistic"(%[[rs1]]
  // CHECK: return %[[rs2]]
}

// CHECK-LABEL: @FuseFullyConnectedRelu
func.func @FuseFullyConnectedRelu(%arg0: tensor<1x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "tfl.fully_connected" (%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x256xf32>, tensor<128x256xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "tfl.relu"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

  // CHECK: %[[RES:[0-9].*]] = "tfl.fully_connected"
  // CHECK-SAME: fused_activation_function = "RELU"
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @FuseFullyConnectedRelu6
func.func @FuseFullyConnectedRelu6(%arg0: tensor<1x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "tfl.fully_connected" (%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x256xf32>, tensor<128x256xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "tfl.relu6"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

  // CHECK: %[[RES:[0-9].*]] = "tfl.fully_connected"
  // CHECK-SAME: fused_activation_function = "RELU6"
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @FuseFullyConnectedRelu1
func.func @FuseFullyConnectedRelu1(%arg0: tensor<1x256xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
  %0 = "tfl.fully_connected" (%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x256xf32>, tensor<128x256xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

  // CHECK: %[[RES:[0-9].*]] = "tfl.fully_connected"
  // CHECK-SAME: fused_activation_function = "RELU_N1_TO_1"
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @HardSwishPattern
func.func @HardSwishPattern(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %three = arith.constant dense<3.> : tensor<f32>
  %six = arith.constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.add"(%arg0, %three)  {fused_activation_function = "RELU6"}  : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.mul"(%1, %six)  {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %2: tensor<1xf32>
  // CHECK: %0 = "tfl.hard_swish"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
}

// CHECK-LABEL: @HardSwishPatternTwo
func.func @HardSwishPatternTwo(%arg0: tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32> {
  %three = arith.constant dense<3.000000e+00> : tensor<f32>
  %six = arith.constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.add"(%arg0, %three) {fused_activation_function = "NONE"} : (tensor<1x128x128x3xf32>, tensor<f32>) -> tensor<1x128x128x3xf32>
  %1 = "tfl.relu6"(%0) : (tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32>
  %2 = tfl.mul %arg0, %1 {fused_activation_function = "NONE"} : tensor<1x128x128x3xf32>
  %3 = "tfl.mul"(%2, %six) {fused_activation_function = "NONE"} : (tensor<1x128x128x3xf32>, tensor<f32>) -> tensor<1x128x128x3xf32>
  func.return %3 : tensor<1x128x128x3xf32>
  // CHECK: %0 = "tfl.hard_swish"(%arg0) : (tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32>
}

// CHECK-LABEL: @HardSwishPatternThree
func.func @HardSwishPatternThree(%arg0: tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32> {
  %three = arith.constant dense<3.000000e+00> : tensor<f32>
  %six = arith.constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.add"(%arg0, %three) {fused_activation_function = "RELU6"} : (tensor<1x128x128x3xf32>, tensor<f32>) -> tensor<1x128x128x3xf32>
  %1 = tfl.mul %arg0, %0 {fused_activation_function = "NONE"} : tensor<1x128x128x3xf32>
  %2 = "tfl.mul"(%1, %six) {fused_activation_function = "NONE"} : (tensor<1x128x128x3xf32>, tensor<f32>) -> tensor<1x128x128x3xf32>
  func.return %2 : tensor<1x128x128x3xf32>
  // CHECK: %0 = "tfl.hard_swish"(%arg0) : (tensor<1x128x128x3xf32>) -> tensor<1x128x128x3xf32>
}

// CHECK-LABEL: @HardSwishPatternFour
func.func @HardSwishPatternFour(%arg0: tensor<?x1x1x1024xf32>) -> tensor<?x1x1x1024xf32> {
  %three = arith.constant dense<3.000000e+00> : tensor<f32>
  %six = arith.constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.add"(%arg0, %three) {fused_activation_function = "NONE"} : (tensor<?x1x1x1024xf32>, tensor<f32>) -> tensor<?x1x1x1024xf32>
  %1 = "tfl.relu6"(%0) : (tensor<?x1x1x1024xf32>) -> tensor<?x1x1x1024xf32>
  %2 = "tfl.mul"(%1, %six) {fused_activation_function = "NONE"} : (tensor<?x1x1x1024xf32>, tensor<f32>) -> tensor<?x1x1x1024xf32>
  %3 = tfl.mul %2, %arg0 {fused_activation_function = "NONE"} : tensor<?x1x1x1024xf32>
  func.return %3 : tensor<?x1x1x1024xf32>
  // CHECK: %0 = "tfl.hard_swish"(%arg0) : (tensor<?x1x1x1024xf32>) -> tensor<?x1x1x1024xf32>
}

// CHECK-LABEL: @HardSwishPatternFail
func.func @HardSwishPatternFail(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %three = arith.constant dense<4.> : tensor<f32>
  %six = arith.constant dense<0.1666666666666> : tensor<f32>
  %0 = "tfl.sub"(%arg0, %three)  {fused_activation_function = "RELU6"}  : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%arg0, %0) {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.mul"(%1, %six)  {fused_activation_function = "NONE"} :  (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %2: tensor<1xf32>
  // CHECK: %0 = tfl.sub(%arg0, %cst) <{fused_activation_function = "RELU6"}> : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
}

// CHECK-LABEL: @L2NormalizePattern
func.func @L2NormalizePattern(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.rsqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.mul"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  func.return %3: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern1
func.func @L2NormalizePattern1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  func.return %3: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern2
func.func @L2NormalizePattern2(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.rsqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.mul"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  func.return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern3
func.func @L2NormalizePattern3(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  func.return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern4
func.func @L2NormalizePattern4(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.maximum"(%1, %cst_1) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  func.return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @L2NormalizePattern5
func.func @L2NormalizePattern5(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1.0e-4]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.maximum"(%1, %cst_1) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  func.return %4: tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.l2_normalization"([[INPUT:%.*]]) <{fused_activation_function = "NONE"}> : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @InvalidL2NormalizePattern
// Div and square ops must take the same argument to be eligible.
func.func @InvalidL2NormalizePattern(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg1, %2) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  func.return %3: tensor<2xf32>
  // CHECK: %3 = tfl.div([[INPUT:%.*]], %2) <{fused_activation_function = "NONE"}> : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: return %3
}

// CHECK-LABEL: @InvalidL2NormalizePattern2
// Epsilon in the add must be < 1e-3
func.func @InvalidL2NormalizePattern2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1.0e-1]> : tensor<1xf32>
  %0 = "tfl.square"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.add"(%1, %cst_1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.sqrt"(%2) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  func.return %4 : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = tfl.div([[INPUT:%.*]], %3) <{fused_activation_function = "NONE"}> : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @InvalidL2NormalizePattern3
// Axis must be last dimension.
func.func @InvalidL2NormalizePattern3(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<[0]> : tensor<1xi32>
  %0 = "tfl.square"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = false} : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.sqrt"(%1) : (tensor<f32>) -> tensor<f32>
  %3 = "tfl.div"(%arg0, %2) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  func.return %3: tensor<2x2xf32>
  // CHECK: %[[RES:[0-9].*]] = tfl.div([[INPUT:%.*]], %2) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseDivIntoConv2d
func.func @fuseDivIntoConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x28x23x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]> : tensor<2x2x2x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x28x23x2xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x28x23x2xf32>, tensor<2xf32>) -> tensor<1x28x23x2xf32>

  func.return %1 : tensor<1x28x23x2xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], {{\[\[}}5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]], {{\[\[\[}}4.500000e+00, 5.000000e+00], [5.500000e+00, 6.000000e+00]], {{\[\[}}6.500000e+00, 7.000000e+00], [7.500000e+00, 8.000000e+00]]]]> : tensor<2x2x2x2xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %cst, %cst_0) <{dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x28x23x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseDivIntoDepthwiseConv2d
func.func @fuseDivIntoDepthwiseConv2d(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]> : tensor<2x2x2x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.depthwise_conv_2d"(%arg0, %cst0, %cst1) {depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  func.return %1 : tensor<1x112x112x2xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<{{\[\[\[\[}}1.000000e+00, 1.000000e+00], [3.000000e+00, 2.000000e+00]], {{\[\[}}5.000000e+00, 3.000000e+00], [7.000000e+00, 4.000000e+00]]], {{\[\[\[}}9.000000e+00, 5.000000e+00], [1.100000e+01, 6.000000e+00]], {{\[\[}}1.300000e+01, 7.000000e+00], [1.500000e+01, 8.000000e+00]]]]> : tensor<2x2x2x2xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.depthwise_conv_2d"(%arg0, %cst, %cst_0) <{depth_multiplier = 1 : i32, dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<1x112x112x2xf32>, tensor<2x2x2x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseDivIntoConv2d_Scalar
func.func @fuseDivIntoConv2d_Scalar(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x28x23x1xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<2xf32>
  %cst2 = arith.constant dense<2.0> : tensor<f32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x28x23x1xf32>
  %1 = "tfl.div"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x28x23x1xf32>, tensor<f32>) -> tensor<1x28x23x1xf32>

  func.return %1 : tensor<1x28x23x1xf32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<{{\[\[\[\[}}5.000000e-01, 1.000000e+00], [1.500000e+00, 2.000000e+00]], {{\[\[}}2.500000e+00, 3.000000e+00], [3.500000e+00, 4.000000e+00]]]]> : tensor<1x2x2x2xf32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<5.000000e-01> : tensor<2xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %[[CST1]], %[[CST2]]) <{dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x28x23x1xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: @fuseMulIntoConv2d_Scalar
func.func @fuseMulIntoConv2d_Scalar(%arg0: tensor<1x112x112x2xf32>) -> tensor<1x28x23x1xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]> : tensor<1x2x2x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<1xf32>
  %cst2 = arith.constant dense<2.0> : tensor<f32>
  %0 = "tfl.conv_2d"(%arg0, %cst0, %cst1) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<1xf32>) -> tensor<1x28x23x1xf32>
  %1 = "tfl.mul"(%0, %cst2) {fused_activation_function = "NONE"} : (tensor<1x28x23x1xf32>, tensor<f32>) -> tensor<1x28x23x1xf32>

  func.return %1 : tensor<1x28x23x1xf32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<{{\[\[\[\[}}2.000000e+00, 4.000000e+00], [6.000000e+00, 8.000000e+00]], {{\[\[}}1.000000e+01, 1.200000e+01], [1.400000e+01, 1.600000e+01]]]]> : tensor<1x2x2x2xf32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : tensor<1xf32>
  // CHECK: %[[RES:[0-9].*]] = "tfl.conv_2d"(%arg0, %[[CST1]], %[[CST2]]) <{dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}> : (tensor<1x112x112x2xf32>, tensor<1x2x2x2xf32>, tensor<1xf32>) -> tensor<1x28x23x1xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: fuseTileWithBinaryOp
func.func @fuseTileWithBinaryOp(%arg0: tensor<1x1xf32>) -> tensor<1x2xf32> {
  %cst = arith.constant dense<[[1,2]]> : tensor<1x2xi32>
  %cst1 = arith.constant dense<[[3.0, 4.0]]> : tensor<1x2xf32>
  %0 = "tfl.sqrt"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.tile"(%0, %cst) : (tensor<1x1xf32>, tensor<1x2xi32>) -> tensor<1x2xf32>
  %2 = "tfl.add"(%cst1, %1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
  func.return %2 : tensor<1x2xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<{{\[\[}}3.000000e+00, 4.000000e+00]]> : tensor<1x2xf32>
  // CHECK: %[[SQRT:[0-9].*]] = "tfl.sqrt"
  // CHECK: %[[RES:[0-9].*]] = tfl.add(%[[SQRT]], %[[cst]])
}

// CHECK-LABEL: fuseTileWithBinaryOp1
func.func @fuseTileWithBinaryOp1(%arg0: tensor<1x1xf32>, %arg1: tensor<1x128xf32>) -> tensor<1x128xf32> {
  %cst_0 = arith.constant dense<1.0> : tensor<f32>
  %cst_1 = arith.constant dense<[1, 128]> : tensor<2xi32>
  %0 = "tfl.add"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %1 = "tfl.sqrt"(%0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "tfl.tile"(%1, %cst_1) : (tensor<1x1xf32>, tensor<2xi32>) -> tensor<1x128xf32>
  %3 = "tfl.div"(%2, %arg1) {fused_activation_function = "NONE"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  func.return %3 : tensor<1x128xf32>

  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[ADD:[0-9].*]] = tfl.add(%arg0, %[[cst]]) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  // CHECK: %[[SQRT:[0-9].*]] = "tfl.sqrt"(%[[ADD]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  // CHECK: %[[RES:[0-9].*]] = tfl.div(%[[SQRT]], %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x1xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: notFuseTileWithBinaryOpOn5DInputs
func.func @notFuseTileWithBinaryOpOn5DInputs(%arg0: tensor<1x1xf32>) -> tensor<1x1x1x1x2xf32> {
  %cst = arith.constant dense<[1, 1, 1, 1, 2]> : tensor<5xi32>
  %cst1 = arith.constant dense<3.0> : tensor<1x1x1x1x2xf32>
  %0 = "tfl.sqrt"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.tile"(%0, %cst) : (tensor<1x1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x2xf32>
  %2 = "tfl.add"(%cst1, %1) {fused_activation_function = "NONE"} : (tensor<1x1x1x1x2xf32>, tensor<1x1x1x1x2xf32>) -> tensor<1x1x1x1x2xf32>
  func.return %2 : tensor<1x1x1x1x2xf32>

  // CHECK: "tfl.sqrt"
  // CHECK: "tfl.tile"
  // CHECK: tfl.add
}

// CHECK-LABEL: notFuseTileWithBinaryOp1On5DInputs
func.func @notFuseTileWithBinaryOp1On5DInputs(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1x1x1x128xf32>) -> tensor<1x1x1x1x128xf32> {
  %cst_0 = arith.constant dense<1.0> : tensor<f32>
  %cst_1 = arith.constant dense<[1, 1, 1, 1, 128]> : tensor<5xi32>
  %0 = "tfl.add"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %1 = "tfl.sqrt"(%0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "tfl.tile"(%1, %cst_1) : (tensor<1x1xf32>, tensor<5xi32>) -> tensor<1x1x1x1x128xf32>
  %3 = "tfl.div"(%2, %arg1) {fused_activation_function = "NONE"} : (tensor<1x1x1x1x128xf32>, tensor<1x1x1x1x128xf32>) -> tensor<1x1x1x1x128xf32>
  func.return %3 : tensor<1x1x1x1x128xf32>

  // CHECK: tfl.add
  // CHECK: "tfl.sqrt"
  // CHECK: "tfl.tile"
  // CHECK: tfl.div
}

// CHECK-LABEL: InvalidFuseTileWithBinaryOp
func.func @InvalidFuseTileWithBinaryOp(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
  %cst = arith.constant dense<[[1,2]]> : tensor<1x2xi32>
  %cst1 = arith.constant dense<[[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]> : tensor<1x6xf32>
  %0 = "tfl.sqrt"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.tile"(%0, %cst) : (tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<2x6xf32>
  %2 = "tfl.add"(%cst1, %1) {fused_activation_function = "NONE"} : (tensor<1x6xf32>, tensor<2x6xf32>) -> tensor<2x6xf32>
  func.return %2 : tensor<2x6xf32>

  // CHECK: %[[TILE:[0-9].*]] = "tfl.tile"
}

// CHECK-LABEL: InvalidFuseTileAlreadyBroadcastAlongTileDim
func.func @InvalidFuseTileAlreadyBroadcastAlongTileDim(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x6x8x1xf32> {
  %cst_1 = arith.constant dense<[1, 6, 8, 1]> : tensor<4xi32>
  %cst_2 = arith.constant dense<[1, 1, 1, 46]> : tensor<4xi32>
  %cst_20 = arith.constant dense<4.600000e+01> : tensor<f32>
  %0 = "tfl.tile"(%arg0, %cst_1) : (tensor<1x1x1x1xf32>, tensor<4xi32>) -> tensor<1x6x8x1xf32>
  %1 = "tfl.mul"(%0, %cst_20) {fused_activation_function = "NONE"} : (tensor<1x6x8x1xf32>, tensor<f32>) -> tensor<1x6x8x1xf32>
  func.return %1 : tensor<1x6x8x1xf32>

  // CHECK: %[[TILE:[0-9].*]] = "tfl.tile"
}

// CHECK-LABEL: FuseHardswish
func.func @FuseHardswish(%arg0: tensor<1x112x112x16xf32>) -> tensor<1x56x56x16xf32> {
  %cst_0 = arith.constant dense<3.0> : tensor<f32>
  %cst_1 = arith.constant dense<0.166666666> : tensor<f32>
  %w = arith.constant dense<1.0> : tensor<1x3x3x16xf32>
  %b = arith.constant dense<10.0> : tensor<16xf32>
  %2 = "tfl.add"(%arg0, %cst_0) {fused_activation_function = "RELU6"} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
  %3 = "tfl.mul"(%2, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x112x112x16xf32>, tensor<f32>) -> tensor<1x112x112x16xf32>
  %4 = tfl.mul %arg0, %3 {fused_activation_function = "NONE"} : tensor<1x112x112x16xf32>
  %5 = "tfl.depthwise_conv_2d"(%4, %w, %b) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x16xf32>, tensor<1x3x3x16xf32>, tensor<16xf32>) -> tensor<1x56x56x16xf32>
  func.return %5 : tensor<1x56x56x16xf32>

// CHECK: tfl.hard_swish
// CHECK: tfl.depthwise_conv_2d
}

// CHECK-LABEL: squeezeToReshape
func.func @squeezeToReshape(%arg0: tensor<1x1x2xf32>) -> tensor<2xf32> {
  %0 = "tfl.squeeze"(%arg0) : (tensor<1x1x2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>

  // CHECK-DAG: [[CONST:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<1x1x2xf32>, tensor<1xi32>) -> tensor<2xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: expandDimsToReshape
func.func @expandDimsToReshape(%arg0: tensor<6x6x256xf32>) -> tensor<6x6x256x1xf32> {
  %cst = arith.constant dense<-1> : tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<6x6x256xf32>, tensor<i32>) -> tensor<6x6x256x1xf32>
  func.return %0 : tensor<6x6x256x1xf32>

  // CHECK-DAG: [[CONST:.*]] = arith.constant dense<[6, 6, 256, 1]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<6x6x256xf32>, tensor<4xi32>) -> tensor<6x6x256x1xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: convertTrivialTransposeToReshape
func.func @convertTrivialTransposeToReshape(%arg0: tensor<6x6x256x1xf32>) -> tensor<1x6x6x256xf32> {
  %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  func.return %0 : tensor<1x6x6x256xf32>

  // CHECK-DAG: [[CONST:.*]] = arith.constant {{.*}}dense<[1, 6, 6, 256]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: doNotConvertNonTrivialTransposeToReshape
func.func @doNotConvertNonTrivialTransposeToReshape(%arg0: tensor<6x6x256x1xf32>) -> tensor<1x6x6x256xf32> {
  // Note: The dimension 0 and 1 are swapped, so it's not trivial
  // (elements are not in the same order).
  %cst = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<6x6x256x1xf32>, tensor<4xi32>) -> tensor<1x6x6x256xf32>
  func.return %0 : tensor<1x6x6x256xf32>

  // CHECK-DAG: [[CONST:.*]] = arith.constant dense<[3, 1, 0, 2]> : tensor<4xi32>
  // CHECK: %[[RESULT:.*]] = "tfl.transpose"(%arg0, %[[CONST:.*]])
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: Relu
func.func @Relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>

  // CHECK: %[[RESULT:.*]] = "tfl.relu"(%arg0)
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: Relu1
func.func @Relu1(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<-1.0> : tensor<f32>
  %cst1 = arith.constant dense<1.0> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.minimum"(%0, %cst1) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>

  // CHECK: %[[relu_n1_to_1:[0-9].*]] = "tfl.relu_n1_to_1"
}

// CHECK-LABEL: Relu1_2
func.func @Relu1_2(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<-1.0> : tensor<f32>
  %cst1 = arith.constant dense<1.0> : tensor<f32>
  %0 = "tfl.minimum"(%arg0, %cst1) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %cst) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>

  // CHECK: %[[relu_n1_to_1:[0-9].*]] = "tfl.relu_n1_to_1"
}

// CHECK-LABEL: fuse_relu_to_add
func.func @fuse_relu_to_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "RELU_N1_TO_1"}
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: leaky_relu_fusion
func.func @leaky_relu_fusion(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<0.2> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %alpha) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.leaky_relu"
}

// CHECK-LABEL: leaky_relu_not_fused
// Should not fuse to LeakyRelu, since alpha > 1.
func.func @leaky_relu_not_fused(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<1.2> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %alpha) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  %1 = "tfl.maximum"(%0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.maximum"
}

// CHECK-LABEL: prelu_fusion
func.func @prelu_fusion(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<-0.2> : tensor<3xf32>
  %0 = "tfl.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tfl.relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tfl.mul"(%alpha, %2) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.prelu"
}

// CHECK-LABEL: prelu_not_fused
// Rank of alpha should be one less than input for PReLU, which is not the case.
func.func @prelu_not_fused(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<-0.2> : tensor<f32>
  %0 = "tfl.relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tfl.neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tfl.relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tfl.mul"(%alpha, %2) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tfl.relu"
}

// CHECK-LABEL: NotfuseAddIntoConv2d_MultipleUsers
func.func @NotfuseAddIntoConv2d_MultipleUsers(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> (tensor<256x8x7x16xf32>, tensor<256x8x7x16xf32>) {
  %cst = arith.constant dense<1.5> : tensor<16xf32>
  %cst_1 = arith.constant dense<3.5> : tensor<16xf32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %2 = "tfl.add"(%0, %cst_1) {fused_activation_function = "NONE"} : (tensor<256x8x7x16xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  func.return %1, %2 : tensor<256x8x7x16xf32>, tensor<256x8x7x16xf32>

  // CHECK: %[[tfl_conv2d:[0-9].*]] = "tfl.conv_2d"
  // CHECK: tfl.add
  // CHECK-NEXT: tfl.add
}

func.func @FusingaddRelu(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.Relu"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tf.Add"(%3, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "tf.Relu6"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  %6 = "tfl.add"(%5, %3) {fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %7 = "tf.Relu6"(%6) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %7: tensor<1xf32>

// Fusing-LABEL: FusingaddRelu
// Fusing:  %[[add:[0-9].*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xf32>
// Fusing:  %[[add1:[0-9].*]] = tfl.add %arg0, %[[add]] {fused_activation_function = "RELU"} : tensor<1xf32>
// Fusing:  %[[relu:[0-9].*]] = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
// Fusing:  %[[add2:[0-9].*]] = tfl.add %[[relu]], %[[add1]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  %[[add3:[0-9].*]] = tfl.add %[[add2]], %[[relu]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  return

// NoFusing-LABEL: FusingaddRelu
// NoFusing:  %[[add:[0-9].*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xf32>
// NoFusing:  %[[add1:[0-9].*]] = tfl.add %arg0, %[[add]] {fused_activation_function = "RELU"} : tensor<1xf32>
// NoFusing:  %[[relu:[0-9].*]] = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
// NoFusing:  %[[add2:[0-9].*]] = tfl.add %[[relu]], %[[add1]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// NoFusing:  %[[add3:[0-9].*]] = tfl.add %[[add2]], %[[relu]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// NoFusing:  return
}

func.func @FusingbiasAdd(%arg0: tensor<1x10x10x32xf32>, %arg1: tensor<32xf32>) -> tensor<1x10x10x32xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  %1 = "tf.BiasAdd"(%0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  %2 = "tf.Relu6"(%1) : (tensor<1x10x10x32xf32>) -> tensor<1x10x10x32xf32>
  func.return %2 : tensor<1x10x10x32xf32>

// Fusing-LABEL: FusingbiasAdd
// Fusing:  %[[add:[0-9].*]] = tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
// Fusing:  %[[add1:[0-9].*]] = tfl.add(%[[add]], %arg1) <{fused_activation_function = "RELU6"}> : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
}

func.func @FusingdivRelu(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.Div"(%arg0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tf.Relu"(%1) : (tensor<1xf32>) -> tensor<1xf32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %4 = "tf.Div"(%3, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "tf.Relu6"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %5: tensor<1xf32>

// Fusing-LABEL: FusingdivRelu
// Fusing:  %[[div:[0-9].*]] = tfl.div %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xf32>
// Fusing:  %[[div1:[0-9].*]] = tfl.div %arg0, %[[div]] {fused_activation_function = "RELU"} : tensor<1xf32>
// Fusing:  %[[relu:[0-9].*]] = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
// Fusing:  %[[div2:[0-9].*]] = tfl.div %[[relu]], %[[div1]] {fused_activation_function = "RELU6"} : tensor<1xf32>
// Fusing:  return
}

func.func @ReorderAddWithConstant(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<2x2xf32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.add"(%0, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

  // CHECK-LABEL: ReorderAddWithConstant
  // CHECK-DAG: %[[CONST:.*]] = arith.constant dense<3.000000e+00> : tensor<2x2xf32>
  // CHECK: %[[RESULT:.*]] = tfl.add %arg0, %[[CONST]] {fused_activation_function = "NONE"} : tensor<2x2xf32>
}

func.func @NotReorderAddWithConstantOn5D(%arg0: tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2x2x2x2xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<2x2x2x2x2xf32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
  %1 = "tfl.add"(%0, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
  func.return %1 : tensor<2x2x2x2x2xf32>

  // CHECK-LABEL: NotReorderAddWithConstantOn5D
  // CHECK: tfl.add
  // CHECK: tfl.add
}

func.func @NotReorderAddWithUnranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<2x2xf32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<2x2xf32>) -> tensor<*xf32>
  %1 = "tfl.add"(%0, %cst_1) {fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<2x2xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>

  // CHECK-LABEL: NotReorderAddWithUnranked
  // CHECK: tfl.add
  // CHECK: tfl.add
}

func.func @RemoveCast(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.cast"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

  // CHECK-LABEL: RemoveCast
  // CHECK: return %arg0
}

func.func @DontRemoveCastToReturn(%arg0: tensor<2x2xf32>) -> tensor<?x?xf32> {
  %1 = "tfl.cast"(%arg0) : (tensor<2x2xf32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
  // CHECK-LABEL: DontRemoveCastToReturn
  // CHECK: %[[CAST:.*]] = "tfl.cast
  // CHECK: return %[[CAST]]
}
func.func @squaredDifferenceReluRemoveRelu(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.relu"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1: tensor<1xf32>

// CHECK-LABEL: squaredDifferenceReluRemoveRelu
// CHECK:  %[[RESULT:.*]] = tfl.squared_difference %arg0, %arg1 : tensor<1xf32>
// CHECK:  return %[[RESULT]]
}

func.func @ConvertSqueezeToReshapeWithDynamicDimension(%arg0: tensor<?x1x8x3xf32>) -> tensor<?x8x3xf32> {
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [1]}: (tensor<?x1x8x3xf32>) -> tensor<?x8x3xf32>
  func.return %0: tensor<?x8x3xf32>

// CHECK-LABEL: ConvertSqueezeToReshapeWithDynamicDimension
// CHECK-DAG: [[CONST:.*]] = arith.constant dense<[-1, 8, 3]> : tensor<3xi32>
// CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<?x1x8x3xf32>, tensor<3xi32>) -> tensor<?x8x3xf32>
// CHECK:  return %[[RESULT]]
}

func.func @ConvertSqueezeToReshapeWithDynamicDimension2(%arg0: tensor<?x1x8x3xf32>) -> tensor<1x8x3xf32> {
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [0]}: (tensor<?x1x8x3xf32>) -> tensor<1x8x3xf32>
  func.return %0: tensor<1x8x3xf32>

// CHECK-LABEL: ConvertSqueezeToReshapeWithDynamicDimension2
// CHECK-DAG: [[CONST:.*]] = arith.constant dense<[1, 8, 3]> : tensor<3xi32>
// CHECK: %[[RESULT:.*]] = "tfl.reshape"(%arg0, %[[CONST:.*]]) : (tensor<?x1x8x3xf32>, tensor<3xi32>) -> tensor<1x8x3xf32>
// CHECK:  return %[[RESULT]]
}

func.func @DontConvertSqueezeToReshape(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [0]}: (tensor<*xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>

// CHECK-LABEL: DontConvertSqueezeToReshape
// CHECK: %[[RESULT:.*]] = "tfl.squeeze"(%arg0)
// CHECK:  return %[[RESULT]]
}

func.func @ConvertSqueezeToReshapeOnMultiDynamicDims(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [0]}: (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0: tensor<?x?xf32>

// CHECK-LABEL: ConvertSqueezeToReshapeOnMultiDynamicDims
// CHECK:  return %arg0
}

func.func @ConvertPow1ToIdentity(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

// CHECK-LABEL: ConvertPow1ToIdentity
// CHECK: return %arg0
}

func.func @ConvertPow2ToSquare(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

// CHECK-LABEL: ConvertPow2ToSquare
// CHECK: %[[RESULT:.*]] = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK: return %[[RESULT]]
}

func.func @ConvertPowHalfToSqrt(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<0.500000e+00> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

// CHECK-LABEL: ConvertPowHalfToSqrt
// CHECK: %[[RESULT:.*]] = "tfl.sqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: return %[[RESULT]]
}

func.func @ConvertPowMinusHalfToRsqrt(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<-0.500000e+00> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>

// CHECK-LABEL: ConvertPowMinusHalfToRsqrt
// CHECK: %[[RESULT:.*]] = "tfl.rsqrt"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK: return %[[RESULT]]
}

func.func @ConvertIdentityGatherNdOp(%arg0: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst = arith.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>
  %0 = "tfl.gather_nd"(%arg0, %cst) : (tensor<4x3xf32>, tensor<4x1xi32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>

// CHECK-LABEL: ConvertIdentityGatherNdOp
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK-NEXT: return %[[ARG]] : tensor<4x3xf32>
}

func.func @ConvertIdentityGatherNdOp3D(%arg0: tensor<4x3x4xf32>) -> tensor<4x3x4xf32> {
  %cst = arith.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>
  %0 = "tfl.gather_nd"(%arg0, %cst) : (tensor<4x3x4xf32>, tensor<4x1xi32>) -> tensor<4x3x4xf32>
  func.return %0 : tensor<4x3x4xf32>

// CHECK-LABEL: ConvertIdentityGatherNdOp3D
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x3x4xf32>) -> tensor<4x3x4xf32>
// CHECK-NEXT: return %[[ARG]] : tensor<4x3x4xf32>
}

func.func @ConvertIdentityScatterNd(%arg0: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %cst = arith.constant dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>
  %shape = arith.constant dense<[4, 3]> : tensor<2xi32>
  %0 = "tfl.scatter_nd"(%cst, %arg0, %shape) : (tensor<4x1xi32>, tensor<4x3xf32>, tensor<2xi32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>

// CHECK-LABEL: ConvertIdentityScatterNd
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK-NEXT: return %[[ARG]] : tensor<4x3xf32>
}

func.func @DontConvertIdentityScatterNdWithLargerOutputDimSize(%arg0: tensor<1xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[[0]]> : tensor<1x1xi32>
  %shape = arith.constant dense<[2]> : tensor<1xi32>
  %0 = "tfl.scatter_nd"(%cst, %arg0, %shape) : (tensor<1x1xi32>, tensor<1xf32>, tensor<1xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>

// CHECK-LABEL: DontConvertIdentityScatterNdWithLargerOutputDimSize
// CHECK: %[[SCATTER_ND:.*]] = "tfl.scatter_nd"(%{{.*}}, %arg0, %{{.*}})
// CHECK-NEXT: return %[[SCATTER_ND]] : tensor<2xf32>
}

func.func @ReshapeAddUnknownShape(%arg0: tensor<*xf32>) -> tensor<3x4xf32> {
  %cst = arith.constant dense<[3, 4]> : tensor<2xi32>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<3x4xf32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<*xf32>, tensor<2xi32>) -> tensor<3x4xf32>
  %1 = "tfl.add"(%0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %1 : tensor<3x4xf32>
// CHECK-LABEL: ReshapeAddUnknownShape
// CHECK: %[[rs1:.*]] = "tfl.reshape"(%arg0
// CHECK: %[[rs2:.*]] = tfl.add %[[rs1]]
// CHECK: return %[[rs2]]
}

func.func @FoldSumKeepDim(%arg0: tensor<8x128xf32>) -> tensor<8x1xf32> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %cst_1 = arith.constant dense<[8, 1]> : tensor<2xi32>
  %0 = "tfl.sum"(%arg0, %cst) {keep_dims = false} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<8xf32>, tensor<2xi32>) -> tensor<8x1xf32>
  func.return %1 : tensor<8x1xf32>

// CHECK-LABEL: FoldSumKeepDim
// CHECK: %[[RESULT:.*]] = "tfl.sum"(%arg0, %cst) <{keep_dims = true}> : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8x1xf32>
// CHECK: return %[[RESULT]] : tensor<8x1xf32>
}

func.func @FoldReduceMinKeepDim(%arg0: tensor<8x128xf32>) -> tensor<1x128xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1, 128]> : tensor<2xi32>
  %0 = "tfl.reduce_min"(%arg0, %cst) {keep_dims = false} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<128xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<128xf32>, tensor<2xi32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

// CHECK-LABEL: FoldReduceMinKeepDim
// CHECK: %[[RESULT:.*]] = "tfl.reduce_min"(%arg0, %cst) <{keep_dims = true}> : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<1x128xf32>
// CHECK: return %[[RESULT]] : tensor<1x128xf32>
}

func.func @FoldReduceMaxKeepDim(%arg0: tensor<8x128xf32>) -> tensor<1x128xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1, 128]> : tensor<2xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst) {keep_dims = false} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<128xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<128xf32>, tensor<2xi32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

// CHECK-LABEL: FoldReduceMaxKeepDim
// CHECK: %[[RESULT:.*]] = "tfl.reduce_max"(%arg0, %cst) <{keep_dims = true}> : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<1x128xf32>
// CHECK: return %[[RESULT]] : tensor<1x128xf32>
}

func.func @FoldReduceProdKeepDim(%arg0: tensor<8x128xf32>) -> tensor<1x1xf32> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, 1]> : tensor<2xi32>
  %0 = "tfl.reduce_prod"(%arg0, %cst) {keep_dims = false} : (tensor<8x128xf32>, tensor<2xi32>) -> tensor<f32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<f32>, tensor<2xi32>) -> tensor<1x1xf32>
  func.return %1 : tensor<1x1xf32>

// CHECK-LABEL: FoldReduceProdKeepDim
// CHECK: %[[RESULT:.*]] = "tfl.reduce_prod"(%arg0, %cst) <{keep_dims = true}> : (tensor<8x128xf32>, tensor<2xi32>) -> tensor<1x1xf32>
// CHECK: return %[[RESULT]] : tensor<1x1xf32>
}

func.func @FoldMeanKeepDim(%arg0: tensor<8x128xf32>) -> tensor<1x128xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[1, 128]> : tensor<2xi32>
  %0 = "tfl.mean"(%arg0, %cst) {keep_dims = false} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<128xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<128xf32>, tensor<2xi32>) -> tensor<1x128xf32>
  func.return %1 : tensor<1x128xf32>

// CHECK-LABEL: FoldMeanKeepDim
// CHECK: %[[RESULT:.*]] = "tfl.mean"(%arg0, %cst) <{keep_dims = true}> : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<1x128xf32>
// CHECK: return %[[RESULT]] : tensor<1x128xf32>
}

func.func @SoftMaxWithNormalization(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst) {keep_dims = true} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8x1xf32>
  %1 = "tfl.sub"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<8x128xf32>, tensor<8x1xf32>) -> tensor<8x128xf32>
  %2 = "tfl.exp"(%1) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  %3 = "tfl.sum"(%2, %cst) {keep_dims = true} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8x1xf32>
  %4 = "tfl.div"(%2, %3) {fused_activation_function = "NONE"} : (tensor<8x128xf32>, tensor<8x1xf32>) -> tensor<8x128xf32>
  func.return %4 : tensor<8x128xf32>

// CHECK-LABEL: SoftMaxWithNormalization
// CHECK: %[[RESULT:.*]] = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
// CHECK: return %[[RESULT]] : tensor<8x128xf32>
}

func.func @SoftMaxWithoutNormalization(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.exp"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = true} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8x1xf32>
  %2 = "tfl.div"(%0, %1) {fused_activation_function = "NONE"} : (tensor<8x128xf32>, tensor<8x1xf32>) -> tensor<8x128xf32>
  func.return %2 : tensor<8x128xf32>

// CHECK-LABEL: SoftMaxWithoutNormalization
// CHECK: %[[RESULT:.*]] = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
// CHECK: return %[[RESULT]] : tensor<8x128xf32>
}

func.func @SoftMaxWithoutNormalizationNegAxis(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.exp"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  %1 = "tfl.sum"(%0, %cst) {keep_dims = true} : (tensor<8x128xf32>, tensor<1xi32>) -> tensor<8x1xf32>
  %2 = "tfl.div"(%0, %1) {fused_activation_function = "NONE"} : (tensor<8x128xf32>, tensor<8x1xf32>) -> tensor<8x128xf32>
  func.return %2 : tensor<8x128xf32>

// CHECK-LABEL: SoftMaxWithoutNormalizationNegAxis
// CHECK: %[[RESULT:.*]] = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<8x128xf32>) -> tensor<8x128xf32>
// CHECK: return %[[RESULT]] : tensor<8x128xf32>
}

// CHECK-LABEL: fuseScalarAddIntoConv2d
func.func @fuseScalarAddIntoConv2d(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<256x8x7x16xf32> {
  %cst = arith.constant dense<1.5> : tensor<f32>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x8x7x16xf32>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x16xf32>, tensor<f32>) -> tensor<256x8x7x16xf32>
  func.return %1 : tensor<256x8x7x16xf32>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf32>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: fuseExpanded1DMulIntoConv2d
func.func @fuseExpanded1DMulIntoConv2d(%arg0: tensor<1x8x8x207xf32>) -> tensor<1x8x8x256xf32> {
  %cst_0 = arith.constant dense<1.4> : tensor<256x3x3x207xf32>
  %cst_1 = arith.constant dense<1.5> : tensor<256xf32>
  %cst_2 = arith.constant dense<2.0> : tensor<1x1x1x256xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst_0, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x8x8x207xf32>, tensor<256x3x3x207xf32>, tensor<256xf32>) -> tensor<1x8x8x256xf32>
  %1 = "tfl.mul"(%0, %cst_2) {fused_activation_function = "NONE"} : (tensor<1x8x8x256xf32>, tensor<1x1x1x256xf32>) -> tensor<1x8x8x256xf32>
  func.return %1 : tensor<1x8x8x256xf32>

// CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<2.800000e+00> : tensor<256x3x3x207xf32>
// CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<3.000000e+00> : tensor<256xf32>
// CHECK: "tfl.conv_2d"(%arg0, %[[CST_0]], %[[CST_1]])

}

// CHECK-LABEL: ConvertMul1ToIdentity
func.func @ConvertMul1ToIdentity(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %cst = arith.constant dense<1.0> : tensor<1x2x3x4xf32>
  %0 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  func.return %0 : tensor<1x2x3x4xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: DontConvertMul12ToIdentity
func.func @DontConvertMul12ToIdentity(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
  // CHECK-DAG: %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK: %0 = tfl.mul %arg0, %cst {fused_activation_function = "NONE"} : tensor<2xf32>
  // CHECK: return %0 : tensor<2xf32>
}

// CHECK-LABEL: ConvertMul1WithBroadcastToIdentity
// If the broadcast doesn't change the dimensions (i.e. constant is smaller than input)
func.func @ConvertMul1WithBroadcastToIdentity(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2xf32>
  %0 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: DontConvertMul1WithBroadcastToIdentity
// If the broadcast changes the dimensions (i.e. constant is larger than input)
func.func @DontConvertMul1WithBroadcastToIdentity(%arg0: tensor<2xf32>) -> tensor<2x2xf32> {
  %cst = arith.constant dense<1.0> : tensor<2x2xf32>
  %0 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
  // CHECK-DAG: %cst = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
  // CHECK: %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: ConvertConstSelectToIdentity
func.func @ConvertConstSelectToIdentity(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xi1>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>) {
  %cst_true = arith.constant dense<true> : tensor<1x2x3x4xi1>
  %cst_false = arith.constant dense<false> : tensor<1x2x3x4xi1>
  %0 = "tfl.select"(%cst_true, %arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %1 = "tfl.select_v2"(%cst_true, %arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %2 = "tfl.select"(%cst_false, %arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %3 = "tfl.select_v2"(%cst_false, %arg0, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %4 = "tfl.select"(%arg2, %cst_true, %cst_false) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  %5 = "tfl.select_v2"(%arg2, %cst_true, %cst_false) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  %6 = "tfl.select"(%arg2, %cst_false, %cst_true) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  %7 = "tfl.select_v2"(%arg2, %cst_false, %cst_true) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  func.return %0, %1, %2, %3, %4, %5, %6, %7 : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>, tensor<1x2x3x4xi1>
  // CHECK: %0 = "tfl.logical_not"(%arg2) : (tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  // CHECK: %1 = "tfl.logical_not"(%arg2) : (tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  // CHECK: return %arg0, %arg0, %arg1, %arg1, %arg2, %arg2, %0, %1
}

// CHECK-LABEL: DontConvertConstSelectBroadcast
func.func @DontConvertConstSelectBroadcast(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<false> : tensor<2x3xi1>
  %0 = "tfl.select"(%cst, %arg0, %arg1) : (tensor<2x3xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
  // CHECK: %0 = "tfl.select"(%cst, %arg0, %arg1) : (tensor<2x3xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  // CHECK: return %0
}

// CHECK-LABEL: DontConvertConstSelectMixed
func.func @DontConvertConstSelectMixed(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %cst = arith.constant dense<[false, true]> : tensor<2xi1>
  %0 = "tfl.select"(%cst, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %1 = "tfl.select_v2"(%cst, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0, %1 : tensor<2xf32>, tensor<2xf32>
  // CHECK: %0 = "tfl.select"(%cst, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %1 = "tfl.select_v2"(%cst, %arg0, %arg1) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %0, %1
}

// CHECK-LABEL: CheckSelectNegated
func.func @CheckSelectNegated(%arg0: tensor<1x2x3x4xi1>, %arg1: tensor<1x2x3x4xf32>, %arg2: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) {
  %not = "tfl.logical_not"(%arg0) : (tensor<1x2x3x4xi1>) -> tensor<1x2x3x4xi1>
  %sel_v1 = "tfl.select"(%not, %arg1, %arg2) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %sel_v2 = "tfl.select_v2"(%not, %arg1, %arg2) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  func.return %sel_v1, %sel_v2 : tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>
  // CHECK: %[[SEL_V1:.*]] = "tfl.select"(%arg0, %arg2, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  // CHECK: %[[SEL_V2:.*]] = "tfl.select_v2"(%arg0, %arg2, %arg1) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  // CHECK: return %[[SEL_V1]], %[[SEL_V2]]
}

// CHECK-LABEL: EliminateRedundantLogicalAnd
func.func @EliminateRedundantLogicalAnd(%arg0: tensor<1xi1>, %arg1: tensor<2x3xi1>) -> (tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>) {
  %cst_false0 = arith.constant dense<false> : tensor<1xi1>
  %cst_false1 = arith.constant dense<false> : tensor<2x3xi1>
  %cst_false2 = arith.constant dense<false> : tensor<3xi1>
  %cst_false3 = arith.constant dense<false> : tensor<2x1xi1>
  %cst_true0 = arith.constant dense<true> : tensor<1xi1>
  %cst_true1 = arith.constant dense<true> : tensor<2x3xi1>
  %cst_true2 = arith.constant dense<true> : tensor<3xi1>
  %cst_true3 = arith.constant dense<true> : tensor<2x1xi1>
  %cst_mixed = arith.constant dense<[[false, false, false], [true, true, true]]> : tensor<2x3xi1>

  // False constant => tfl.logical_and is replaced by the constant, if shapes are OK
  %f0 = "tfl.logical_and"(%arg0, %cst_false0): (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>        // match
  %f1 = "tfl.logical_and"(%arg0, %cst_false1): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // match
  %f2 = "tfl.logical_and"(%arg1, %cst_false0): (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>    // no match
  %f3 = "tfl.logical_and"(%arg1, %cst_false1): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // match
  %f4 = "tfl.logical_and"(%arg1, %cst_false2): (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>    // no match
  %f5 = "tfl.logical_and"(%arg1, %cst_false3): (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>  // no match
  // True constant => tfl.logical_and is replaced by the input, if shapes are OK
  %t0 = "tfl.logical_and"(%arg0, %cst_true0): (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>        // match
  %t1 = "tfl.logical_and"(%arg0, %cst_true1): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // no match
  %t2 = "tfl.logical_and"(%arg1, %cst_true0): (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>    // match
  %t3 = "tfl.logical_and"(%arg1, %cst_true1): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // match
  %t4 = "tfl.logical_and"(%arg1, %cst_true2): (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>    // match
  %t5 = "tfl.logical_and"(%arg1, %cst_true3): (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>  // match
  // Mixed constant => tfl.logical_and remains
  %m0 = "tfl.logical_and"(%arg0, %cst_mixed): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // no match
  %m1 = "tfl.logical_and"(%arg1, %cst_mixed): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // no match
  func.return %f0, %f1, %f2, %f3, %f4, %f5, %t0, %t1, %t2, %t3, %t4, %t5, %m0, %m1 : tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>

  // CHECK-DAG: %[[CST_FALSE0:.*]] = arith.constant dense<false> : tensor<1xi1>
  // CHECK-DAG: %[[CST_FALSE1:.*]] = arith.constant dense<false> : tensor<2x3xi1>
  // CHECK-DAG: %[[CST_FALSE2:.*]] = arith.constant dense<false> : tensor<3xi1>
  // CHECK-DAG: %[[CST_FALSE3:.*]] = arith.constant dense<false> : tensor<2x1xi1>
  // CHECK-DAG: %[[CST_TRUE1:.*]] = arith.constant dense<true> : tensor<2x3xi1>
  // CHECK-DAG: %[[CST_MIXED:.*]] = arith.constant dense<{{\[\[}}false, false, false], [true, true, true]]> : tensor<2x3xi1>
  // CHECK: %[[FALSE2:.*]] = tfl.logical_and(%arg1, %[[CST_FALSE0]]) : (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>
  // CHECK: %[[FALSE4:.*]] = tfl.logical_and(%arg1, %[[CST_FALSE2]]) : (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[FALSE5:.*]] = tfl.logical_and(%arg1, %[[CST_FALSE3]]) : (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>
  // CHECK: %[[TRUE1:.*]] = tfl.logical_and(%arg0, %[[CST_TRUE1]]) : (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[MIXED0:.*]] = tfl.logical_and(%arg0, %[[CST_MIXED]]) : (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[MIXED1:.*]] = tfl.logical_and %arg1, %[[CST_MIXED]] : tensor<2x3xi1>
  // CHECK: return %[[CST_FALSE0]], %[[CST_FALSE1]], %[[FALSE2]], %[[CST_FALSE1]], %[[FALSE4]], %[[FALSE5]], %arg0, %[[TRUE1]], %arg1, %arg1, %arg1, %arg1, %[[MIXED0]], %[[MIXED1]] : tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>
}

// CHECK-LABEL: EliminateRedundantLogicalOr
func.func @EliminateRedundantLogicalOr(%arg0: tensor<1xi1>, %arg1: tensor<2x3xi1>) -> (tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>) {
  %cst_false0 = arith.constant dense<false> : tensor<1xi1>
  %cst_false1 = arith.constant dense<false> : tensor<2x3xi1>
  %cst_false2 = arith.constant dense<false> : tensor<3xi1>
  %cst_false3 = arith.constant dense<false> : tensor<2x1xi1>
  %cst_true0 = arith.constant dense<true> : tensor<1xi1>
  %cst_true1 = arith.constant dense<true> : tensor<2x3xi1>
  %cst_true2 = arith.constant dense<true> : tensor<3xi1>
  %cst_true3 = arith.constant dense<true> : tensor<2x1xi1>
  %cst_mixed = arith.constant dense<[[false, false, false], [true, true, true]]> : tensor<2x3xi1>

  // False constant => tfl.logical_or is replaced by the input, if shapes are OK
  %f0 = "tfl.logical_or"(%arg0, %cst_false0): (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>        // match
  %f1 = "tfl.logical_or"(%arg0, %cst_false1): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // no match
  %f2 = "tfl.logical_or"(%arg1, %cst_false0): (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>    // match
  %f3 = "tfl.logical_or"(%arg1, %cst_false1): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // match
  %f4 = "tfl.logical_or"(%arg1, %cst_false2): (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>    // match
  %f5 = "tfl.logical_or"(%arg1, %cst_false3): (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>  // match
  // True constant => tfl.logical_or is replaced by the constant, if shapes are OK
  %t0 = "tfl.logical_or"(%arg0, %cst_true0): (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>        // match
  %t1 = "tfl.logical_or"(%arg0, %cst_true1): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // match
  %t2 = "tfl.logical_or"(%arg1, %cst_true0): (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>    // no match
  %t3 = "tfl.logical_or"(%arg1, %cst_true1): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // match
  %t4 = "tfl.logical_or"(%arg1, %cst_true2): (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>    // no match
  %t5 = "tfl.logical_or"(%arg1, %cst_true3): (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>  // no match
  // Mixed constant => tfl.logical_or remains
  %m0 = "tfl.logical_or"(%arg0, %cst_mixed): (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>    // no match
  %m1 = "tfl.logical_or"(%arg1, %cst_mixed): (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>  // no match
  func.return %f0, %f1, %f2, %f3, %f4, %f5, %t0, %t1, %t2, %t3, %t4, %t5, %m0, %m1 : tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>

  // CHECK-DAG: %[[CST_FALSE0:.*]] = arith.constant dense<false> : tensor<2x3xi1>
  // CHECK-DAG: %[[CST_TRUE0:.*]] = arith.constant dense<true> : tensor<1xi1>
  // CHECK-DAG: %[[CST_TRUE1:.*]] = arith.constant dense<true> : tensor<2x3xi1>
  // CHECK-DAG: %[[CST_TRUE2:.*]] = arith.constant dense<true> : tensor<3xi1>
  // CHECK-DAG: %[[CST_TRUE3:.*]] = arith.constant dense<true> : tensor<2x1xi1>
  // CHECK-DAG: %[[CST_MIXED:.*]] = arith.constant dense<{{\[\[}}false, false, false], [true, true, true]]> : tensor<2x3xi1>
  // CHECK: %[[FALSE2:.*]] = tfl.logical_or(%arg0, %[[CST_FALSE0]]) : (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[TRUE2:.*]] = tfl.logical_or(%arg1, %[[CST_TRUE0]]) : (tensor<2x3xi1>, tensor<1xi1>) -> tensor<2x3xi1>
  // CHECK: %[[TRUE4:.*]] = tfl.logical_or(%arg1, %[[CST_TRUE2]]) : (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[TRUE5:.*]] = tfl.logical_or(%arg1, %[[CST_TRUE3]]) : (tensor<2x3xi1>, tensor<2x1xi1>) -> tensor<2x3xi1>
  // CHECK: %[[MIXED0:.*]] = tfl.logical_or(%arg0, %[[CST_MIXED]]) : (tensor<1xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  // CHECK: %[[MIXED1:.*]] = tfl.logical_or %arg1, %[[CST_MIXED]] : tensor<2x3xi1>
  // CHECK: return %arg0, %[[FALSE2]], %arg1, %arg1, %arg1, %arg1, %[[CST_TRUE0]], %[[CST_TRUE1]], %[[TRUE2]], %[[CST_TRUE1]], %[[TRUE4]], %[[TRUE5]], %[[MIXED0]], %[[MIXED1]] : tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<1xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>, tensor<2x3xi1>
}

// CHECK-LABEL: EliminateReduceOpsBool
func.func @EliminateReduceOpsBool(%arg: tensor<1x2x1x3xi1>, %arg_scalar: tensor<i1>, %arg_unknown: tensor<?xi1>, %axis_unknown: tensor<?xi32>) -> (tensor<i1>, tensor<i1>, tensor<2x1x3xi1>, tensor<1x2x1x3xi1>, tensor<1x1x1x3xi1>, tensor<1x1x3xi1>, tensor<1x2x3xi1>, tensor<1x2x1x3xi1>, tensor<1x2x1xi1>, tensor<1x2x1x1xi1>, tensor<?xi1>) {
  %axis_0 = arith.constant dense<0> : tensor<1xi32>
  %axis_1 = arith.constant dense<1> : tensor<1xi32>
  %axis_2 = arith.constant dense<2> : tensor<1xi32>
  %axis_3 = arith.constant dense<3> : tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg_scalar, %axis_0) { keep_dims = false } : (tensor<i1>, tensor<1xi32>) -> tensor<i1>
  %1 = "tfl.reduce_any"(%arg_scalar, %axis_1) { keep_dims = true } : (tensor<i1>, tensor<1xi32>) -> tensor<i1>
  %2 = "tfl.reduce_any"(%arg, %axis_0) { keep_dims = false } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<2x1x3xi1>
  %3 = "tfl.reduce_any"(%arg, %axis_0) { keep_dims = true } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1x3xi1>
  %4 = "tfl.reduce_any"(%arg, %axis_1) { keep_dims = false } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x1x1x3xi1>
  %5 = "tfl.reduce_all"(%arg, %axis_1) { keep_dims = true } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x1x3xi1>
  %6 = "tfl.reduce_all"(%arg, %axis_2) { keep_dims = false } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x3xi1>
  %7 = "tfl.reduce_all"(%arg, %axis_2) { keep_dims = true } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1x3xi1>
  %8 = "tfl.reduce_all"(%arg, %axis_3) { keep_dims = false } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1xi1>
  %9 = "tfl.reduce_all"(%arg, %axis_3) { keep_dims = true } : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1x1xi1>
  %10 = "tfl.reduce_all"(%arg_unknown, %axis_unknown) { keep_dims = true } : (tensor<?xi1>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : tensor<i1>, tensor<i1>, tensor<2x1x3xi1>, tensor<1x2x1x3xi1>, tensor<1x1x1x3xi1>, tensor<1x1x3xi1>, tensor<1x2x3xi1>, tensor<1x2x1x3xi1>, tensor<1x2x1xi1>, tensor<1x2x1x1xi1>, tensor<?xi1>

  // CHECK-DAG: %[[AXIS_0:.*]] = arith.constant dense<0> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_1:.*]] = arith.constant dense<1> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_2:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_3:.*]] = arith.constant dense<3> : tensor<1xi32>
  // CHECK: %[[RET_0:.*]] = "tfl.reduce_any"(%arg0, %[[AXIS_0]]) <{keep_dims = false}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<2x1x3xi1>
  // CHECK: %[[RET_1:.*]] = "tfl.reduce_any"(%arg0, %[[AXIS_1]]) <{keep_dims = false}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x1x1x3xi1>
  // CHECK: %[[RET_2:.*]] = "tfl.reduce_all"(%arg0, %[[AXIS_1]]) <{keep_dims = true}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x1x3xi1>
  // CHECK: %[[RET_3:.*]] = "tfl.reduce_all"(%arg0, %[[AXIS_2]]) <{keep_dims = false}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x3xi1>
  // CHECK: %[[RET_4:.*]] = "tfl.reduce_all"(%arg0, %[[AXIS_3]]) <{keep_dims = false}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1xi1>
  // CHECK: %[[RET_5:.*]] = "tfl.reduce_all"(%arg0, %[[AXIS_3]]) <{keep_dims = true}> : (tensor<1x2x1x3xi1>, tensor<1xi32>) -> tensor<1x2x1x1xi1>
  // CHECK: %[[RET_6:.*]] = "tfl.reduce_all"(%arg2, %arg3) <{keep_dims = true}> : (tensor<?xi1>, tensor<?xi32>) -> tensor<?xi1>
  // CHECK: return %arg1, %arg1, %[[RET_0]], %arg0, %[[RET_1]], %[[RET_2]], %[[RET_3]], %arg0, %[[RET_4]], %[[RET_5]], %[[RET_6]] : tensor<i1>, tensor<i1>, tensor<2x1x3xi1>, tensor<1x2x1x3xi1>, tensor<1x1x1x3xi1>, tensor<1x1x3xi1>, tensor<1x2x3xi1>, tensor<1x2x1x3xi1>, tensor<1x2x1xi1>, tensor<1x2x1x1xi1>, tensor<?xi1>
}

// CHECK-LABEL: EliminateReduceOpsFloat
func.func @EliminateReduceOpsFloat(%arg: tensor<1x2x1x3xf32>, %arg_scalar: tensor<f32>, %arg_unknown: tensor<?xf32>, %axis_unknown: tensor<?xi32>) -> (tensor<f32>, tensor<f32>, tensor<2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x2x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1xf32>, tensor<1x2x1x1xf32>, tensor<?xf32>) {
  %axis_0 = arith.constant dense<0> : tensor<1xi32>
  %axis_1 = arith.constant dense<1> : tensor<1xi32>
  %axis_2 = arith.constant dense<2> : tensor<1xi32>
  %axis_3 = arith.constant dense<3> : tensor<1xi32>
  %0 = "tfl.mean"(%arg_scalar, %axis_0) { keep_dims = false } : (tensor<f32>, tensor<1xi32>) -> tensor<f32>
  %1 = "tfl.sum"(%arg_scalar, %axis_1) { keep_dims = true } : (tensor<f32>, tensor<1xi32>) -> tensor<f32>
  %2 = "tfl.reduce_min"(%arg, %axis_0) { keep_dims = false } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<2x1x3xf32>
  %3 = "tfl.reduce_max"(%arg, %axis_0) { keep_dims = true } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1x3xf32>
  %4 = "tfl.reduce_prod"(%arg, %axis_1) { keep_dims = false } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x1x1x3xf32>
  %5 = "tfl.mean"(%arg, %axis_1) { keep_dims = true } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x1x3xf32>
  %6 = "tfl.sum"(%arg, %axis_2) { keep_dims = false } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x3xf32>
  %7 = "tfl.reduce_min"(%arg, %axis_2) { keep_dims = true } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1x3xf32>
  %8 = "tfl.reduce_max"(%arg, %axis_3) { keep_dims = false } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1xf32>
  %9 = "tfl.reduce_prod"(%arg, %axis_3) { keep_dims = true } : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1x1xf32>
  %10 = "tfl.sum"(%arg_unknown, %axis_unknown) { keep_dims = true } : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  func.return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10 : tensor<f32>, tensor<f32>, tensor<2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x2x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1xf32>, tensor<1x2x1x1xf32>, tensor<?xf32>

  // CHECK-DAG: %[[AXIS_0:.*]] = arith.constant dense<0> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_1:.*]] = arith.constant dense<1> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_2:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK-DAG: %[[AXIS_3:.*]] = arith.constant dense<3> : tensor<1xi32>
  // CHECK: %[[RET_0:.*]] = "tfl.reduce_min"(%arg0, %[[AXIS_0]]) <{keep_dims = false}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<2x1x3xf32>
  // CHECK: %[[RET_1:.*]] = "tfl.reduce_prod"(%arg0, %[[AXIS_1]]) <{keep_dims = false}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x1x1x3xf32>
  // CHECK: %[[RET_2:.*]] = "tfl.mean"(%arg0, %[[AXIS_1]]) <{keep_dims = true}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x1x3xf32>
  // CHECK: %[[RET_3:.*]] = "tfl.sum"(%arg0, %[[AXIS_2]]) <{keep_dims = false}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x3xf32>
  // CHECK: %[[RET_4:.*]] = "tfl.reduce_max"(%arg0, %[[AXIS_3]]) <{keep_dims = false}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1xf32>
  // CHECK: %[[RET_5:.*]] = "tfl.reduce_prod"(%arg0, %[[AXIS_3]]) <{keep_dims = true}> : (tensor<1x2x1x3xf32>, tensor<1xi32>) -> tensor<1x2x1x1xf32>
  // CHECK: %[[RET_6:.*]] = "tfl.sum"(%arg2, %arg3) <{keep_dims = true}> : (tensor<?xf32>, tensor<?xi32>) -> tensor<?xf32>
  // CHECK: return %arg1, %arg1, %[[RET_0]], %arg0, %[[RET_1]], %[[RET_2]], %[[RET_3]], %arg0, %[[RET_4]], %[[RET_5]], %[[RET_6]] : tensor<f32>, tensor<f32>, tensor<2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x3xf32>, tensor<1x2x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1xf32>, tensor<1x2x1x1xf32>, tensor<?xf32>
}

// CHECK-LABEL: RemoveSoftmaxBeforeArgmax
func.func @RemoveSoftmaxBeforeArgmax(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_max"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MAX]] : tensor<16xi32>
}

// CHECK-LABEL: RemoveSoftmaxBeforeArgmin
func.func @RemoveSoftmaxBeforeArgmin(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_min"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK: %[[ARG_MIN:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MIN]] : tensor<16xi32>
}

// CHECK-LABEL: RemoveLogSoftmaxBeforeArgmax
func.func @RemoveLogSoftmaxBeforeArgmax(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.log_softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_max"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%arg0, %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MAX]] : tensor<16xi32>
}

// CHECK-LABEL: RemoveLogSoftmaxBeforeArgmin
func.func @RemoveLogSoftmaxBeforeArgmin(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.log_softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_min"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK: %[[ARG_MIN:.*]] = "tfl.arg_min"(%arg0, %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MIN]] : tensor<16xi32>
}

// CHECK-LABEL: DontRemoveSoftmaxNegativeBetaBeforeArgmax
func.func @DontRemoveSoftmaxNegativeBetaBeforeArgmax(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<-1> : tensor<1xi32>
  %0 = "tfl.softmax"(%arg0) {beta = -1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_max"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK: %[[SOFTMAX:.*]] = "tfl.softmax"(%arg0) <{beta = -1.000000e+00 : f32}> : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  // CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%[[SOFTMAX]], %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MAX]] : tensor<16xi32>
}

// CHECK-LABEL: DontRemoveSoftmaxNonLastAxisBeforeArgmax
func.func @DontRemoveSoftmaxNonLastAxisBeforeArgmax(%arg0: tensor<16x1024xf32>) -> tensor<16xi32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  %1 = "tfl.arg_max"(%0, %cst) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  func.return %1 : tensor<16xi32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : tensor<1xi32>
  // CHECK: %[[SOFTMAX:.*]] = "tfl.softmax"(%arg0) <{beta = 1.000000e+00 : f32}> : (tensor<16x1024xf32>) -> tensor<16x1024xf32>
  // CHECK: %[[ARG_MAX:.*]] = "tfl.arg_max"(%[[SOFTMAX]], %[[CST]]) : (tensor<16x1024xf32>, tensor<1xi32>) -> tensor<16xi32>
  // CHECK: return %[[ARG_MAX]] : tensor<16xi32>
}

// CHECK-LABEL: @ReorderReshapex2Add
func.func @ReorderReshapex2Add(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x2x3x4xf32>) -> tensor<6x4xf32> {
  %shape = arith.constant dense<[6, 4]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
  %1 = "tfl.reshape"(%arg1, %shape) : (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<6x4xf32>, tensor<6x4xf32>) -> tensor<6x4xf32>
  func.return %2 : tensor<6x4xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[6, 4]> : tensor<2xi32>
  // CHECK: %[[VAL_0:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x2x3x4xf32>
  // CHECK: %[[VAL_1:.*]] = "tfl.reshape"(%[[VAL_0]], %[[CST]]) : (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<6x4xf32>
  // CHECK: return %[[VAL_1]]
}

// CHECK-LABEL: @NotReorderReshapeAndMul_MultipleUsers
func.func @NotReorderReshapeAndMul_MultipleUsers(%arg0: tensor<3x40xf32>, %arg1: tensor<1x3x8x1xf32>) -> (tensor<1x3x8x5xf32>, tensor<1x3x8x5xf32>) {
  %cst_1 = arith.constant dense<[1, 3, 8, 5]> : tensor<4xi32>
  %2 = "tfl.reshape"(%arg0, %cst_1) : (tensor<3x40xf32>, tensor<4xi32>) -> tensor<1x3x8x5xf32>
  %3 = tfl.mul %2, %2 {fused_activation_function = "NONE"} : tensor<1x3x8x5xf32>
  %9 = tfl.mul(%2, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x3x8x5xf32>, tensor<1x3x8x1xf32>) -> tensor<1x3x8x5xf32>
  return %3, %9 : tensor<1x3x8x5xf32>, tensor<1x3x8x5xf32>
  // CHECK:  %cst = arith.constant dense<[1, 3, 8, 5]> : tensor<4xi32>
  // CHECK:  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x40xf32>, tensor<4xi32>) -> tensor<1x3x8x5xf32>
  // CHECK:  %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<1x3x8x5xf32>
  // CHECK:  %2 = tfl.mul(%0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x3x8x5xf32>, tensor<1x3x8x1xf32>) -> tensor<1x3x8x5xf32>
  // CHECK:  return %1, %2 : tensor<1x3x8x5xf32>, tensor<1x3x8x5xf32>
}

// CHECK-LABEL: @DontReorderReshapex2Add
func.func @DontReorderReshapex2Add(%arg0: tensor<1x2x3x4x!quant.uniform<i8:f32, 3.0>>, %arg1: tensor<1x2x3x4x!quant.uniform<i8:f32, 5.0>>) -> tensor<6x4x!quant.uniform<i8:f32, 7.0>> {
  %shape = arith.constant dense<[6, 4]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %shape) : (tensor<1x2x3x4x!quant.uniform<i8:f32, 3.0>>, tensor<2xi32>) -> tensor<6x4x!quant.uniform<i8:f32, 3.0>>
  %1 = "tfl.reshape"(%arg1, %shape) : (tensor<1x2x3x4x!quant.uniform<i8:f32, 5.0>>, tensor<2xi32>) -> tensor<6x4x!quant.uniform<i8:f32, 5.0>>
  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<6x4x!quant.uniform<i8:f32, 3.0>>, tensor<6x4x!quant.uniform<i8:f32, 5.0>>) -> tensor<6x4x!quant.uniform<i8:f32, 7.0>>
  func.return %2 : tensor<6x4x!quant.uniform<i8:f32, 7.0>>

  // CHECK: %[[SHAPE:.*]] = arith.constant dense<[6, 4]> : tensor<2xi32>
  // CHECK-DAG: %[[VAL_0:.*]] = "tfl.reshape"(%arg0, %[[SHAPE]])
  // CHECK-DAG: %[[VAL_1:.*]] = "tfl.reshape"(%arg1, %[[SHAPE]])
  // CHECK: %[[ADD:.*]] = tfl.add(%[[VAL_0]], %[[VAL_1]])
  // CHECK: return %[[ADD]]
}

// CHECK-LABEL: ConvertSliceToIdentityI32
func.func @ConvertSliceToIdentityI32(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %begin = arith.constant dense<0> : tensor<4xi32>
  %shape = arith.constant dense<[2,3,4,5]> : tensor<4xi32>
  %0 = "tfl.slice"(%arg0, %begin, %shape) : (tensor<2x3x4x5xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
  func.return %0 : tensor<2x3x4x5xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: ConvertSliceToIdentityI64
func.func @ConvertSliceToIdentityI64(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %begin = arith.constant dense<0> : tensor<4xi64>
  %shape = arith.constant dense<[2,3,4,5]> : tensor<4xi64>
  %0 = "tfl.slice"(%arg0, %begin, %shape) : (tensor<2x3x4x5xf32>, tensor<4xi64>, tensor<4xi64>) -> tensor<2x3x4x5xf32>
  func.return %0 : tensor<2x3x4x5xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: ConvertSliceToIdentityStaticDimWithShapeWithNeg1
func.func @ConvertSliceToIdentityStaticDimWithShapeWithNeg1(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %begin = arith.constant dense<0> : tensor<4xi32>
  %shape = arith.constant dense<[-1, 3, -1, 5]> : tensor<4xi32>
  %0 = "tfl.slice"(%arg0, %begin, %shape) : (tensor<2x3x4x5xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<2x3x4x5xf32>
  func.return %0 : tensor<2x3x4x5xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: ConvertSliceToIdentityDynamicDimAndShapeWithNeg1
func.func @ConvertSliceToIdentityDynamicDimAndShapeWithNeg1(%arg0: tensor<?x3x?x5xf32>) -> tensor<?x3x?x5xf32> {
  %begin = arith.constant dense<0> : tensor<4xi32>
  %shape = arith.constant dense<[-1, 3, -1, 5]> : tensor<4xi32>
  %0 = "tfl.slice"(%arg0, %begin, %shape) : (tensor<?x3x?x5xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x3x?x5xf32>
  func.return %0 : tensor<?x3x?x5xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: DontConvertSliceToIdentity
func.func @DontConvertSliceToIdentity(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x4xf32>, tensor<1x2x3x4xf32>) {
  %begin0 = arith.constant dense<0> : tensor<4xi64>
  %shape0 = arith.constant dense<[2,3,4,4]> : tensor<4xi64>
  %begin1 = arith.constant dense<1> : tensor<4xi64>
  %shape1 = arith.constant dense<[1,2,3,4]> : tensor<4xi64>
  %0 = "tfl.slice"(%arg0, %begin0, %shape0) : (tensor<2x3x4x5xf32>, tensor<4xi64>, tensor<4xi64>) -> tensor<2x3x4x4xf32>
  %1 = "tfl.slice"(%arg0, %begin1, %shape1) : (tensor<2x3x4x5xf32>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
  func.return %0, %1 : tensor<2x3x4x4xf32>, tensor<1x2x3x4xf32>
  // CHECK-DAG: %[[BEGIN_0:.*]] = arith.constant dense<0> : tensor<4xi64>
  // CHECK-DAG: %[[SHAPE_0:.*]] = arith.constant dense<[2, 3, 4, 4]> : tensor<4xi64>
  // CHECK-DAG: %[[BEGIN_1:.*]] = arith.constant dense<1> : tensor<4xi64>
  // CHECK-DAG: %[[SHAPE_1:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  // CHECK: %[[SLICE_0:.*]] = "tfl.slice"(%arg0, %[[BEGIN_0]], %[[SHAPE_0]]) : (tensor<2x3x4x5xf32>, tensor<4xi64>, tensor<4xi64>) -> tensor<2x3x4x4xf32>
  // CHECK: %[[SLICE_1:.*]] = "tfl.slice"(%arg0, %[[BEGIN_1]], %[[SHAPE_1]]) : (tensor<2x3x4x5xf32>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x2x3x4xf32>
  // CHECK: return %[[SLICE_0]], %[[SLICE_1]] : tensor<2x3x4x4xf32>, tensor<1x2x3x4xf32>
}

// CHECK-LABEL: DontConvertSliceToIdentityNonConstShape
func.func @DontConvertSliceToIdentityNonConstShape(%arg0: tensor<?xf32>, %arg1: tensor<1xi32>) -> tensor<?xf32> {
  %begin = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.slice"(%arg0, %begin, %arg1) : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
  // CHECK-DAG: %[[BEGIN:.*]] = arith.constant dense<0> : tensor<1xi32>
  // CHECK: %[[SLICE:.*]] = "tfl.slice"(%arg0, %[[BEGIN]], %arg1) : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK: return %[[SLICE]] : tensor<?xf32>
}

// CHECK-LABEL: DontConvertSliceToIdentityDynamicDimButEqualShape
func.func @DontConvertSliceToIdentityDynamicDimButEqualShape(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %begin = arith.constant dense<0> : tensor<1xi32>
  %shape = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.slice"(%arg0, %begin, %shape) : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
  // CHECK-DAG: %[[BEGIN:.*]] = arith.constant dense<0> : tensor<1xi32>
  // CHECK-DAG: %[[SHAPE:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK: %[[SLICE:.*]] = "tfl.slice"(%arg0, %[[BEGIN]], %[[SHAPE]]) : (tensor<?xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xf32>
  // CHECK: return %[[SLICE]] : tensor<?xf32>
}

// CHECK-LABEL: @FuseAddWithFullyConnectedWithBias
func.func @FuseAddWithFullyConnectedWithBias(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_add = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.add"(%arg, %cst_add) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // 2.0 * 3.0 * 512 + 5.0 = 3077.0
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<3.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<3.077000e+03> : tensor<1024xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.fully_connected"(%arg0, %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  // CHECK: return %[[RESULT]]
}

// Checks the `FuseAddAndFullyConnected` pattern is not applied if there is quantized type.
// CHECK-LABEL: @FuseAddWithFullyConnectedWithQuantizedWeight
func.func @FuseAddWithFullyConnectedWithQuantizedWeight(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_add = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = "tfl.pseudo_qconst"() {qtype = tensor<3072x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>, value = dense<1> : tensor<1024x512xi8>} : () -> tensor<1024x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.add"(%arg, %cst_add) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK: tfl.add
}

// CHECK-LABEL: @FuseBatchMatMulAndTransposeWithQuantizedWeight
func.func @FuseBatchMatMulAndTransposeWithQuantizedWeight(%arg: tensor<1x2xf32>) -> tensor<1x3xf32> {
  %cst_3 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %79 = "tfl.pseudo_qconst"() {qtype = tensor<3x2x!quant.uniform<i8<-127:127>:f32:0, {2.378620e-03,2.848260e-03,2.545190e-03}>>, value = dense<10> : tensor<3x2xi8>} : () -> tensor<3x2x!quant.uniform<i8<-127:127>:f32:0, {2.378620e-03,2.848260e-03,2.545190e-03}>>
  %80 = "tfl.transpose"(%79, %cst_3) : (tensor<3x2x!quant.uniform<i8<-127:127>:f32:0, {2.378620e-03,2.848260e-03,2.545190e-03}>>, tensor<2xi32>) -> tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {2.378620e-03,2.848260e-03,2.545190e-03}>>
  %81 = "tfl.batch_matmul"(%arg, %80) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x2xf32>, tensor<2x3x!quant.uniform<i8<-127:127>:f32:1, {2.378620e-03,2.848260e-03,2.545190e-03}>>) -> tensor<1x3xf32>
  func.return %81 : tensor<1x3xf32>

  // CHECK: tfl.fully_connected
}

// CHECK-LABEL: @FuseAddWithFullyConnectedNoBias
// Note: Currently not fused.
func.func @FuseAddWithFullyConnectedNoBias(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_add = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = "tfl.no_value"() {value} : () -> none

  %0 = "tfl.add"(%arg, %cst_add) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[ADDEND:.*]] = arith.constant dense<2.000000e+00> : tensor<512xf32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<3.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[VAL_0:.*]] = tfl.add(%arg0, %[[ADDEND]]) <{fused_activation_function = "NONE"}> : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  // CHECK: %[[VAL_1:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>
  // CHECK: return %[[VAL_1]]
}

// CHECK-LABEL: @DontFuseAddWithFullyConnectedMismatchedDimensions
func.func @DontFuseAddWithFullyConnectedMismatchedDimensions(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_add = arith.constant dense<2.0> : tensor<2x512xf32>  // Not 1D
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.add"(%arg, %cst_add) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<2x512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[ADDEND:.*]] = arith.constant dense<2.000000e+00> : tensor<2x512xf32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<3.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<5.000000e+00> : tensor<1024xf32>
  // CHECK: %[[VAL_0:.*]] = tfl.add %arg0, %[[ADDEND]] {fused_activation_function = "NONE"} : tensor<2x512xf32>
  // CHECK: %[[VAL_1:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  // CHECK: return %[[VAL_1]]
}

// CHECK-LABEL: @FuseMulWithFullyConnectedWithBias
func.func @FuseMulWithFullyConnectedWithBias(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_mul = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.mul"(%arg, %cst_mul) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<6.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<5.000000e+00> : tensor<1024xf32>
  // CHECK: %[[RESULT:.*]] = "tfl.fully_connected"(%arg0, %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  // CHECK: return %[[RESULT]]
}

// Checks the `FuseMulAndFullyConnected` pattern is not applied if there is quantized type.
// CHECK-LABEL: @FuseMulWithFullyConnectedWithQuantizedWeight
func.func @FuseMulWithFullyConnectedWithQuantizedWeight(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_mul = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = "tfl.pseudo_qconst"() {qtype = tensor<3072x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>, value = dense<1> : tensor<1024x512xi8>} : () -> tensor<1024x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.mul"(%arg, %cst_mul) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512x!quant.uniform<i8<-127:127>:f32, 0.039600040763616562>>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK: tfl.mul
}

// CHECK-LABEL: @FuseMulWithFullyConnectedNoBias
func.func @FuseMulWithFullyConnectedNoBias(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_mul = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = "tfl.no_value"() {value} : () -> none

  %0 = "tfl.mul"(%arg, %cst_mul) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<6.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[VAL_0:.*]] = "tfl.fully_connected"(%arg0, %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>
  // CHECK: return %[[VAL_0]]

  // NoFusing-LABEL: FuseMulWithFullyConnectedNoBias
  // NoFusing-DAG: %[[MWEIGHTS:.*]] = arith.constant dense<2.000000e+00> : tensor<512xf32>
  // NoFusing-DAG: %[[WEIGHTS:.*]] = arith.constant dense<3.000000e+00> : tensor<1024x512xf32>
  // NoFusing-DAG: %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // NoFusing: %[[MUL:.*]] = tfl.mul(%arg0, %[[MWEIGHTS]]) <{fused_activation_function = "NONE"}> : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  // NoFusing: %[[VAL:.*]] = "tfl.fully_connected"(%[[MUL]], %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>
  // NoFusing: return %[[VAL]]
}

// CHECK-LABEL: @FuseMulWithFullyConnectedNoBiasWithOptionalAttribute
func.func @FuseMulWithFullyConnectedNoBiasWithOptionalAttribute(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_mul = arith.constant dense<2.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = "tfl.no_value"() {value} : () -> none

  %0 = "tfl.mul"(%arg, %cst_mul) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, none) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<6.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[VAL_0:.*]] = "tfl.fully_connected"(%arg0, %[[WEIGHTS]], %[[BIAS]]) <{asymmetric_quantize_inputs = true,
}

// CHECK-LABEL: @DontFuseMulWithFullyConnectedMismatchedDimensions
func.func @DontFuseMulWithFullyConnectedMismatchedDimensions(%arg: tensor<2x512xf32>) -> tensor<2x1024xf32> {
  %cst_mul = arith.constant dense<2.0> : tensor<2x512xf32>  // Not 1D
  %cst_weights = arith.constant dense<3.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<5.0> : tensor<1024xf32>

  %0 = "tfl.mul"(%arg, %cst_mul) {fused_activation_function = "NONE"} : (tensor<2x512xf32>, tensor<2x512xf32>) -> tensor<2x512xf32>
  %1 = "tfl.fully_connected" (%0, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %1 : tensor<2x1024xf32>

  // CHECK-DAG: %[[MULTIPLIER:.*]] = arith.constant dense<2.000000e+00> : tensor<2x512xf32>
  // CHECK-DAG: %[[WEIGHTS:.*]] = arith.constant dense<3.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[BIAS:.*]] = arith.constant dense<5.000000e+00> : tensor<1024xf32>
  // CHECK: %[[VAL_0:.*]] = tfl.mul %arg0, %[[MULTIPLIER]] {fused_activation_function = "NONE"} : tensor<2x512xf32>
  // CHECK: %[[VAL_1:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[WEIGHTS]], %[[BIAS]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  // CHECK: return %[[VAL_1]]
}

// CHECK-LABEL: RemoveReshapeBeforeFullyConnectedExpandDims0
func.func @RemoveReshapeBeforeFullyConnectedExpandDims0(%arg0: tensor<128x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<1x128x64xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x128x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<128x32xf32>
  func.return %1 : tensor<128x32xf32>
  // CHECK: %[[FULLY_CONNECTED:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<128x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<128x32xf32>
  // CHECK: return %[[FULLY_CONNECTED]] : tensor<128x32xf32>
}

// CHECK-LABEL: RemoveReshapeBeforeFullyConnectedReshape
func.func @RemoveReshapeBeforeFullyConnectedReshape(%arg0: tensor<128x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32xf32>) -> tensor<128x32xf32> {
  %cst = arith.constant dense<[4, 32, 64]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<4x32x64xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x32x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<128x32xf32>
  func.return %1 : tensor<128x32xf32>
  // CHECK: %[[FULLY_CONNECTED:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<128x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<128x32xf32>
  // CHECK: return %[[FULLY_CONNECTED]] : tensor<128x32xf32>
}

// CHECK-LABEL: DontRemoveReshapeBeforeFullyConnectedKeepNumDims
func.func @DontRemoveReshapeBeforeFullyConnectedKeepNumDims(%arg0: tensor<128x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32xf32>) -> tensor<1x128x32xf32> {
  %cst = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<1x128x64xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<1x128x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<1x128x32xf32>
  func.return %1 : tensor<1x128x32xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<1x128x64xf32>
  // CHECK: %[[FULLY_CONNECTED:.*]] = "tfl.fully_connected"(%[[RESHAPE]], %arg1, %arg2) <{fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<1x128x64xf32>, tensor<32x64xf32>, tensor<32xf32>) -> tensor<1x128x32xf32>
  // CHECK: return %[[FULLY_CONNECTED]] : tensor<1x128x32xf32>
}

// CHECK-LABEL: DontRemoveReshapeBeforeFullyConnectedChangeLastDim
func.func @DontRemoveReshapeBeforeFullyConnectedChangeLastDim(%arg0: tensor<128x64xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32xf32>) -> tensor<256x32xf32> {
  %cst = arith.constant dense<[1, 256, 32]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<1x256x32xf32>
  %1 = "tfl.fully_connected"(%0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x256x32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
  func.return %1 : tensor<256x32xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1, 256, 32]> : tensor<3xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<128x64xf32>, tensor<3xi32>) -> tensor<1x256x32xf32>
  // CHECK: %[[FULLY_CONNECTED:.*]] = "tfl.fully_connected"(%[[RESHAPE]], %arg1, %arg2) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<1x256x32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<256x32xf32>
  // CHECK: return %[[FULLY_CONNECTED]] : tensor<256x32xf32>
}

// CHECK-LABEL: DontFuseAddWithConvActivationFunc
func.func @DontFuseAddWithConvActivationFunc(%arg0: tensor<1x3x1x1xf32>) -> tensor<1x2x1x3xf32> {
  %cst = arith.constant dense<1.5> : tensor<1xf32>
  %cst_1 = arith.constant dense<0.0> : tensor<3xf32>
  %cst_2 = arith.constant dense<1.1> : tensor<3x2x1x1xf32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "RELU6"} : (tensor<1x3x1x1xf32>, tensor<1xf32>) -> tensor<1x3x1x1xf32>
  %1 = "tfl.conv_2d"(%0, %cst_2, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x1x1xf32>, tensor<3x2x1x1xf32>, tensor<3xf32>) -> tensor<1x2x1x3xf32>
  func.return %1 : tensor<1x2x1x3xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.500000e+00> : tensor<1xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<1.100000e+00> : tensor<3x2x1x1xf32>
  // CHECK: %[[ADD:.*]] = tfl.add(%arg0, %[[CST]]) <{fused_activation_function = "RELU6"}> : (tensor<1x3x1x1xf32>, tensor<1xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: %[[CONV:.*]] = "tfl.conv_2d"(%[[ADD]], %[[CST_2]], %[[CST_1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x3x1x1xf32>, tensor<3x2x1x1xf32>, tensor<3xf32>) -> tensor<1x2x1x3xf32>
  // CHECK: return %[[CONV]]
}

// CHECK-LABEL: fuseUnpackAndConcatToReshape
func.func @fuseUnpackAndConcatToReshape(%arg0: tensor<1x3x2xf32>) -> tensor<1x6xf32> {
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<1x3x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>)
  %1 = "tfl.concatenation"(%0#0, %0#1, %0#2) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x6xf32>
  func.return %1 : tensor<1x6xf32>
  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"(){{.*}}dense<[1, 6]> : tensor<2xi32>
  // CHECK: %[[RES:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<1x3x2xf32>, tensor<2xi32>) -> tensor<1x6xf32>
  // CHECK: return %[[RES]]
}

// CHECK-LABEL: replaceReshapeEqualWithOneHotSingleDim
func.func @replaceReshapeEqualWithOneHotSingleDim(%arg: tensor<1xi32>) -> tensor<3xi1> {
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %result = "tfl.equal"(%arg, %cst) : (tensor<1xi32>, tensor<3xi32>) -> tensor<3xi1>
  func.return %result : tensor<3xi1>

  // CHECK-NOT: tfl.one_hot
}

// CHECK-LABEL: replaceReshapeEqualWithOneHot
func.func @replaceReshapeEqualWithOneHot(%arg: tensor<2x1xi32>) -> tensor<2x3xi1> {
  // Good match: Replace with one_hot
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %result = "tfl.equal"(%arg, %cst) : (tensor<2x1xi32>, tensor<3xi32>) -> tensor<2x3xi1>
  func.return %result : tensor<2x3xi1>

  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<true> : tensor<i1>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST4]]) : (tensor<2x1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK: %[[RES:.*]] = "tfl.one_hot"(%[[RESHAPE]], %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<i1>, tensor<i1>) -> tensor<2x3xi1>
}

// CHECK-LABEL: ReplaceReshapeEqualWithOneHotWithBatchingDim
func.func @ReplaceReshapeEqualWithOneHotWithBatchingDim(%arg: tensor<2x2x1xi32>) -> tensor<2x2x3xi1> {
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %result = "tfl.equal"(%arg, %cst) : (tensor<2x2x1xi32>, tensor<3xi32>) -> tensor<2x2x3xi1>
  func.return %result : tensor<2x2x3xi1>

  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<true> : tensor<i1>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<false> : tensor<i1>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<2> : tensor<2xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST4]]) : (tensor<2x2x1xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  // CHECK: %[[RES:.*]] = "tfl.one_hot"(%[[RESHAPE]], %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2x2xi32>, tensor<i32>, tensor<i1>, tensor<i1>) -> tensor<2x2x3xi1>
}

// CHECK-LABEL: noReplaceReshapeEqualWithOneHotBadShape
func.func @noReplaceReshapeEqualWithOneHotBadShape(%arg: tensor<6xi32>) -> tensor<2x3xi1> {
  // Do not replace: shape's last dimension isn't 1
  %shape = arith.constant dense<[2, 3]> : tensor<2xi32>
  %cst = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %tmp = "tfl.reshape"(%arg, %shape) : (tensor<6xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  %result = "tfl.equal"(%tmp, %cst) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi1>
  func.return %result : tensor<2x3xi1>

  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<[2, 3]> : tensor<2xi32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK: %[[TMP:.*]] = "tfl.reshape"(%arg0, %[[CST1]]) : (tensor<6xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  // CHECK: %[[RES:.*]] = "tfl.equal"(%[[TMP]], %[[CST2]]) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi1>
}

// CHECK-LABEL: noReplaceReshapeEqualWithOneHotBadIndex
func.func @noReplaceReshapeEqualWithOneHotBadIndex(%arg: tensor<2xi32>) -> tensor<2x3xi1> {
  // Do not replace: the constant is not a series from 0 to N-1
  %shape = arith.constant dense<[2, 1]> : tensor<2xi32>
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %tmp = "tfl.reshape"(%arg, %shape) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x1xi32>
  %result = "tfl.equal"(%tmp, %cst) : (tensor<2x1xi32>, tensor<3xi32>) -> tensor<2x3xi1>
  func.return %result : tensor<2x3xi1>

  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<[2, 1]> : tensor<2xi32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  // CHECK: %[[TMP:.*]] = "tfl.reshape"(%arg0, %[[CST1]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x1xi32>
  // CHECK: %[[RES:.*]] = "tfl.equal"(%[[TMP]], %[[CST2]]) : (tensor<2x1xi32>, tensor<3xi32>) -> tensor<2x3xi1>
}

// CHECK-LABEL: ReplaceReshapeEqualOneHotDynamicBatch
func.func @ReplaceReshapeEqualOneHotDynamicBatch(%arg0: tensor<?xi32>) -> (tensor<?x10xf32>) {
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %1 = "tfl.equal"(%0, %cst_0) : (tensor<?x1xi32>, tensor<10xi32>) -> tensor<?x10xi1>
  %2 = "tfl.cast"(%1) : (tensor<?x10xi1>) -> tensor<?x10xf32>
  func.return %2 : tensor<?x10xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<-1> : tensor<1xi32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<10> : tensor<i32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<-1> : tensor<i32>
  // CHECK: %[[EXPAND_DIMS:.*]] = "tfl.expand_dims"(%arg0, %[[CST_3]]) : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%0, %[[CST]]) : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32>
  // CHECK: %[[ONE_HOT:.*]] = "tfl.one_hot"(%1, %[[CST_0]], %[[CST_1]], %[[CST_2]]) <{axis = -1 : i32}> : (tensor<?xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<?x10xf32>
  // CHECK-NEXT: return %[[ONE_HOT]]
}

// CHECK-LABEL: noReplaceReshapeEqualWithOneHotDynamicNonBatch
func.func @noReplaceReshapeEqualWithOneHotDynamicNonBatch(%arg0: tensor<1x?xi32>) -> tensor<1x?x10xf32> {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %1 = "tfl.equal"(%arg0, %cst) : (tensor<1x?xi32>, tensor<10xi32>) -> tensor<1x?x10xi1>
  %2 = "tfl.cast"(%1) : (tensor<1x?x10xi1>) -> tensor<1x?x10xf32>
  func.return %2 : tensor<1x?x10xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  // CHECK: %[[EQUAL:.*]] = "tfl.equal"(%arg0, %[[CST]]) : (tensor<1x?xi32>, tensor<10xi32>) -> tensor<1x?x10xi1>
  // CHECK: %[[CAST:.*]] = "tfl.cast"(%[[EQUAL]]) : (tensor<1x?x10xi1>) -> tensor<1x?x10xf32>
  // CHECK-NEXT: return %[[CAST]]
}

// CHECK-LABEL: noReplaceReshapeEqualWithOneHotUnranked
func.func @noReplaceReshapeEqualWithOneHotUnranked(%arg0: tensor<*xi1>) -> tensor<*xi1> {
  %cst = arith.constant dense<true> : tensor<i1>
  %1 = "tfl.equal"(%arg0, %cst) : (tensor<*xi1>, tensor<i1>) -> tensor<*xi1>
  func.return %1 : tensor<*xi1>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<true> : tensor<i1>
  // CHECK: %[[EQUAL:.*]] = "tfl.equal"(%arg0, %cst) : (tensor<*xi1>, tensor<i1>) -> tensor<*xi1>
  // CHECK-NEXT: return %[[EQUAL]]
}

// CHECK-LABEL: noReplaceReshapeEqualWithOneHotDynamicNonBatchRank1
func.func @noReplaceReshapeEqualWithOneHotDynamicNonBatchRank1(%arg0: tensor<?xi32>) -> tensor<?x10xf32> {
  %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  %1 = "tfl.equal"(%arg0, %cst) : (tensor<?xi32>, tensor<10xi32>) -> tensor<?x10xi1>
  %2 = "tfl.cast"(%1) : (tensor<?x10xi1>) -> tensor<?x10xf32>
  func.return %2 : tensor<?x10xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  // CHECK: %[[EQUAL:.*]] = "tfl.equal"(%arg0, %[[CST]]) : (tensor<?xi32>, tensor<10xi32>) -> tensor<?x10xi1>
  // CHECK: %[[CAST:.*]] = "tfl.cast"(%[[EQUAL]]) : (tensor<?x10xi1>) -> tensor<?x10xf32>
  // CHECK-NEXT: return %[[CAST]]
}

// CHECK-LABEL: fuseOneHotCast
func.func @fuseOneHotCast(%arg: tensor<2xi32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %depth = arith.constant dense<3> : tensor<i32>
  %bool_on = arith.constant dense<true> : tensor<i1>
  %bool_off = arith.constant dense<false> : tensor<i1>
  %int_on = arith.constant dense<5> : tensor<i32>
  %int_off = arith.constant dense<7> : tensor<i32>

  %tmp_bool = "tfl.one_hot"(%arg, %depth, %bool_on, %bool_off) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<i1>, tensor<i1>) -> tensor<2x3xi1>
  %result_bool = "tfl.cast"(%tmp_bool) : (tensor<2x3xi1>) -> tensor<2x3xf32>

  %tmp_int = "tfl.one_hot"(%arg, %depth, %int_on, %int_off) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2x3xi1>
  %result_int = "tfl.cast"(%tmp_int) : (tensor<2x3xi1>) -> tensor<2x3xf32>

  func.return %result_bool, %result_int : tensor<2x3xf32>, tensor<2x3xf32>

  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<5.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST5:.*]] = arith.constant dense<7.000000e+00> : tensor<f32>
  // CHECK: %[[RES1:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES2:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST4]], %[[CST5]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
}

// CHECK-LABEL: replaceOneHotFullyConnectedWithLookup
func.func @replaceOneHotFullyConnectedWithLookup(%arg: tensor<2xi32>) -> tensor<2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<[[7.0, 11.0, 13.0], [17.0, 19.0, 23.0], [29.0, 31.0, 37.0], [41.0, 43.0, 47.0], [53.0, 59.0, 61.0]]> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}7.000000e+00, 1.700000e+01, 2.900000e+01, 4.100000e+01, 5.300000e+01], [1.100000e+01, 1.900000e+01, 3.100000e+01, 4.300000e+01, 5.900000e+01], [1.300000e+01, 2.300000e+01, 3.700000e+01, 4.700000e+01, 6.100000e+01]]> : tensor<3x5xf32>
  // CHECK: %[[RES:.*]] = "tfl.embedding_lookup"(%arg0, %[[CST]]) : (tensor<2xi32>, tensor<3x5xf32>) -> tensor<2x5xf32>
  // CHECK: return %[[RES]] : tensor<2x5xf32>
}

// CHECK-LABEL: dontReplaceOneHotFullyConnectedWithLookupBadIndexType
func.func @dontReplaceOneHotFullyConnectedWithLookupBadIndexType(%arg: tensor<2xi64>) -> tensor<2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<2xi64>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK-NOT: "tfl.embedding_lookup"
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<7.000000e+00> : tensor<5x3xf32>
  // CHECK-DAG: %[[CST5:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[TMP:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi64>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = "tfl.fully_connected"(%[[TMP]], %[[CST4]], %[[CST5]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>
  // CHECK: return %[[RES]] : tensor<2x5xf32>
}

// CHECK-LABEL: dontReplaceOneHotFullyConnectedWithLookupBadIndexTypeWithOptionalAttribute
func.func @dontReplaceOneHotFullyConnectedWithLookupBadIndexTypeWithOptionalAttribute(%arg: tensor<2xi64>) -> tensor<2x5xf32> {
  // Test whether ReplaceOneHotFullyConnectedWithLookup in optimize_patterns.td works as expected.
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<2xi64>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK-NOT: "tfl.embedding_lookup"
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<7.000000e+00> : tensor<5x3xf32>
  // CHECK-DAG: %[[CST5:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[TMP:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi64>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = "tfl.fully_connected"(%[[TMP]], %[[CST4]], %[[CST5]]) <{asymmetric_quantize_inputs = true,
}

// CHECK-LABEL: ReplaceOneHotFullyConnectedWithLookup2DRank
func.func @ReplaceOneHotFullyConnectedWithLookup2DRank(%arg: tensor<11x2xi32>) -> tensor<11x2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<11x2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<11x2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<11x2x3xf32>, tensor<5x3xf32>, none) -> tensor<11x2x5xf32>

  func.return %result : tensor<11x2x5xf32>

  // CHECK-DAG: %[[CST0:.*]] = "tfl.pseudo_const"(){{.*}}dense<22> : tensor<1xi32>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<7.000000e+00> : tensor<3x5xf32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<[11, 2, 5]> : tensor<3xi32>
  // CHECK: %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST0]]) : (tensor<11x2xi32>, tensor<1xi32>) -> tensor<22xi32>
  // CHECK: %[[TMP:.*]] = "tfl.embedding_lookup"(%[[RESHAPE]], %[[CST1]]) : (tensor<22xi32>, tensor<3x5xf32>) -> tensor<22x5xf32>
  // CHECK: %[[RES:.*]] = "tfl.reshape"(%[[TMP]], %[[CST2]]) : (tensor<22x5xf32>, tensor<3xi32>) -> tensor<11x2x5xf32>
  // CHECK: return %[[RES]] : tensor<11x2x5xf32>
}

// CHECK-LABEL: dontReplaceOneHotFullyConnectedWithLookupBadOn
func.func @dontReplaceOneHotFullyConnectedWithLookupBadOn(%arg: tensor<2xi32>) -> tensor<2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on_badvalue = arith.constant dense<2.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on_badvalue, %off) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK-NOT: "tfl.embedding_lookup"
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<7.000000e+00> : tensor<5x3xf32>
  // CHECK-DAG: %[[CST5:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[TMP:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = "tfl.fully_connected"(%[[TMP]], %[[CST4]], %[[CST5]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>
  // CHECK: return %[[RES]] : tensor<2x5xf32>
}

// CHECK-LABEL: dontReplaceOneHotFullyConnectedWithLookupBadOff
func.func @dontReplaceOneHotFullyConnectedWithLookupBadOff(%arg: tensor<2xi32>) -> tensor<2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off_badvalue = arith.constant dense<-1.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off_badvalue) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK-NOT: "tfl.embedding_lookup"
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<-1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<7.000000e+00> : tensor<5x3xf32>
  // CHECK-DAG: %[[CST5:.*]] = "tfl.no_value"() <{value}> : () -> none
  // CHECK: %[[TMP:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = "tfl.fully_connected"(%[[TMP]], %[[CST4]], %[[CST5]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x3xf32>, tensor<5x3xf32>, none) -> tensor<2x5xf32>
  // CHECK: return %[[RES]] : tensor<2x5xf32>
}

// CHECK-LABEL: dontReplaceOneHotFullyConnectedWithLookupBadBias
func.func @dontReplaceOneHotFullyConnectedWithLookupBadBias(%arg: tensor<2xi32>) -> tensor<2x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<7.0> : tensor<5x3xf32>
  %bias_badvalue = arith.constant dense<11.0> : tensor<f32>

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias_badvalue) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x3xf32>, tensor<5x3xf32>, tensor<f32>) -> tensor<2x5xf32>

  func.return %result : tensor<2x5xf32>

  // CHECK-NOT: "tfl.embedding_lookup"
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK-DAG: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST3:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST4:.*]] = arith.constant dense<7.000000e+00> : tensor<5x3xf32>
  // CHECK-DAG: %[[CST5:.*]] = arith.constant dense<1.100000e+01> : tensor<f32>
  // CHECK: %[[TMP:.*]] = "tfl.one_hot"(%arg0, %[[CST1]], %[[CST2]], %[[CST3]]) <{axis = -1 : i32}> : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: %[[RES:.*]] = "tfl.fully_connected"(%[[TMP]], %[[CST4]], %[[CST5]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x3xf32>, tensor<5x3xf32>, tensor<f32>) -> tensor<2x5xf32>
  // CHECK: return %[[RES]] : tensor<2x5xf32>
}

// CHECK-LABEL: replaceOneHotFullyConnectedWithLookupWithDynamicIndexShapeRank1
func.func @replaceOneHotFullyConnectedWithLookupWithDynamicIndexShapeRank1(%arg: tensor<?xi32>) -> tensor<?x5xf32> {
  %depth = arith.constant dense<3> : tensor<i32>
  %on = arith.constant dense<1.0> : tensor<f32>
  %off = arith.constant dense<0.0> : tensor<f32>
  %filter = arith.constant dense<[[7.0, 11.0, 13.0], [17.0, 19.0, 23.0], [29.0, 31.0, 37.0], [41.0, 43.0, 47.0], [53.0, 59.0, 61.0]]> : tensor<5x3xf32>
  %bias = "tfl.no_value"() {value} : () -> none

  %tmp = "tfl.one_hot"(%arg, %depth, %on, %off) {axis = -1 : i32} : (tensor<?xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<?x3xf32>
  %result = "tfl.fully_connected"(%tmp, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<?x3xf32>, tensor<5x3xf32>, none) -> tensor<?x5xf32>

  func.return %result : tensor<?x5xf32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}7.000000e+00, 1.700000e+01, 2.900000e+01, 4.100000e+01, 5.300000e+01], [1.100000e+01, 1.900000e+01, 3.100000e+01, 4.300000e+01, 5.900000e+01], [1.300000e+01, 2.300000e+01, 3.700000e+01, 4.700000e+01, 6.100000e+01]]> : tensor<3x5xf32>
  // CHECK: %[[RES:.*]] = "tfl.embedding_lookup"(%arg0, %[[CST]]) : (tensor<?xi32>, tensor<3x5xf32>) -> tensor<?x5xf32>
  // CHECK: return %[[RES]] : tensor<?x5xf32>
}

// CHECK-LABEL:   func @optimizeTopK(
// CHECK-SAME:                        %[[ARG:.*]]: tensor<3x10xf32>) -> (tensor<3x5xf32>, tensor<3x5xi32>) {
// CHECK:           %[[K:.*]] = "tfl.pseudo_const"(){{.*}}dense<5> : tensor<i32>
// CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = "tfl.topk_v2"(%[[ARG]], %[[K]]) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x5xf32>, tensor<3x5xi32>)
// CHECK:           return %[[VALUES]], %[[INDICES]] : tensor<3x5xf32>, tensor<3x5xi32>
// CHECK:         }
func.func @optimizeTopK(%arg: tensor<3x10xf32>) -> (tensor<3x5xf32>, tensor<3x5xi32>) {
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<0> : tensor<2xi32>
  %cst_1 = arith.constant dense<[3, 5]> : tensor<2xi32>
  %0 = "tfl.slice"(%values, %cst_0, %cst_1) : (tensor<3x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xf32>
  %1 = "tfl.slice"(%indices, %cst_0, %cst_1) : (tensor<3x10xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xi32>
  func.return %0, %1 : tensor<3x5xf32>, tensor<3x5xi32>
}

// CHECK-LABEL:   func @optimizeTopKOnlyValues(
// CHECK-SAME:                                 %[[ARG:.*]]: tensor<3x10xf32>) -> tensor<3x5xf32> {
// CHECK:           %[[K:.*]] = "tfl.pseudo_const"(){{.*}}dense<5> : tensor<i32>
// CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = "tfl.topk_v2"(%[[ARG]], %[[K]]) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x5xf32>, tensor<3x5xi32>)
// CHECK:           return %[[VALUES]] : tensor<3x5xf32>
// CHECK:         }
func.func @optimizeTopKOnlyValues(%arg: tensor<3x10xf32>) -> tensor<3x5xf32>{
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<0> : tensor<2xi32>
  %cst_1 = arith.constant dense<[3, 5]> : tensor<2xi32>
  %0 = "tfl.slice"(%values, %cst_0, %cst_1) : (tensor<3x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xf32>
  func.return %0 : tensor<3x5xf32>
}

// CHECK-LABEL:   func @optimizeTopKOnlyIndices(
// CHECK-SAME:                                  %[[ARG:.*]]: tensor<3x10xf32>) -> tensor<3x5xi32> {
// CHECK:           %[[K:.*]] = "tfl.pseudo_const"(){{.*}}dense<5> : tensor<i32>
// CHECK:           %[[VALUES:.*]], %[[INDICES:.*]] = "tfl.topk_v2"(%[[ARG]], %[[K]]) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x5xf32>, tensor<3x5xi32>)
// CHECK:           return %[[INDICES]] : tensor<3x5xi32>
// CHECK:         }
func.func @optimizeTopKOnlyIndices(%arg: tensor<3x10xf32>) -> tensor<3x5xi32>{
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<0> : tensor<2xi32>
  %cst_1 = arith.constant dense<[3, 5]> : tensor<2xi32>
  %0 = "tfl.slice"(%indices, %cst_0, %cst_1) : (tensor<3x10xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xi32>
  func.return %0 : tensor<3x5xi32>
}

// CHECK-LABEL:   func @nonZeroBeginDontOptimizeTopK
func.func @nonZeroBeginDontOptimizeTopK(%arg: tensor<3x10xf32>) -> (tensor<3x5xf32>, tensor<3x5xi32>) {
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<[0, 1]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[3, 5]> : tensor<2xi32>
  // CHECK: "tfl.slice"
  %0 = "tfl.slice"(%values, %cst_0, %cst_1) : (tensor<3x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xf32>
  // CHECK: "tfl.slice"
  %1 = "tfl.slice"(%indices, %cst_0, %cst_1) : (tensor<3x10xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xi32>
  func.return %0, %1 : tensor<3x5xf32>, tensor<3x5xi32>
}

// CHECK-LABEL:   func @invalidSliceSizeDontOptimizeTopK
func.func @invalidSliceSizeDontOptimizeTopK(%arg: tensor<3x10xf32>) -> (tensor<2x5xf32>, tensor<2x5xi32>) {
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<[0, 0]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[2, 5]> : tensor<2xi32>
  // CHECK: "tfl.slice"
  %0 = "tfl.slice"(%values, %cst_0, %cst_1) : (tensor<3x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x5xf32>
  // CHECK: "tfl.slice"
  %1 = "tfl.slice"(%indices, %cst_0, %cst_1) : (tensor<3x10xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x5xi32>
  func.return %0, %1 : tensor<2x5xf32>, tensor<2x5xi32>
}

// CHECK-LABEL:   func @multiUsesDontOptimizeTopK
func.func @multiUsesDontOptimizeTopK(%arg: tensor<3x10xf32>) -> (tensor<3x5xf32>, tensor<3x5xi32>, tensor<3x10xf32>) {
  %K = arith.constant dense<10> : tensor<i32>
  %values, %indices = "tfl.topk_v2"(%arg, %K) : (tensor<3x10xf32>, tensor<i32>) -> (tensor<3x10xf32>, tensor<3x10xi32>)
  %cst_0 = arith.constant dense<0> : tensor<2xi32>
  %cst_1 = arith.constant dense<[3, 5]> : tensor<2xi32>
  // CHECK: "tfl.slice"
  %0 = "tfl.slice"(%values, %cst_0, %cst_1) : (tensor<3x10xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xf32>
  // CHECK: "tfl.slice"
  %1 = "tfl.slice"(%indices, %cst_0, %cst_1) : (tensor<3x10xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x5xi32>
  %2 = "tfl.add"(%values, %values) {fused_activation_function = "NONE"} : (tensor<3x10xf32>, tensor<3x10xf32>) -> tensor<3x10xf32>
  func.return %0, %1, %2 : tensor<3x5xf32>, tensor<3x5xi32>, tensor<3x10xf32>
}

// CHECK-LABEL:   func @eliminateCumSumCheckIndices
func.func @eliminateCumSumCheckIndices(%arg: tensor<1x2x1x3xf32>) -> (tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>) {
  %axis_m4 = arith.constant dense<-4> : tensor<i32>
  %axis_m3 = arith.constant dense<-3> : tensor<i32>
  %axis_m2 = arith.constant dense<-2> : tensor<i32>
  %axis_m1 = arith.constant dense<-1> : tensor<i32>
  %axis_00 = arith.constant dense<0> : tensor<i32>
  %axis_p1 = arith.constant dense<1> : tensor<i32>
  %axis_p2 = arith.constant dense<2> : tensor<i32>
  %axis_p3 = arith.constant dense<3> : tensor<i32>
  %res_m4 = "tfl.cumsum"(%arg, %axis_m4) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_m3 = "tfl.cumsum"(%arg, %axis_m3) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  %res_m2 = "tfl.cumsum"(%arg, %axis_m2) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_m1 = "tfl.cumsum"(%arg, %axis_m1) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  %res_00 = "tfl.cumsum"(%arg, %axis_00) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_p1 = "tfl.cumsum"(%arg, %axis_p1) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  %res_p2 = "tfl.cumsum"(%arg, %axis_p2) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_p3 = "tfl.cumsum"(%arg, %axis_p3) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  func.return %res_m4, %res_m3, %res_m2, %res_m1, %res_00, %res_p1, %res_p2, %res_p3 : tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>

  // CHECK-DAG: %[[AXIS_M3:.*]] = arith.constant dense<-3> : tensor<i32>
  // CHECK-DAG: %[[AXIS_M1:.*]] = arith.constant dense<-1> : tensor<i32>
  // CHECK-DAG: %[[AXIS_P1:.*]] = arith.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[AXIS_P3:.*]] = arith.constant dense<3> : tensor<i32>
  // CHECK: %[[RES_M3:.*]] = "tfl.cumsum"(%arg0, %[[AXIS_M3]]) <{exclusive = false, reverse = false}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: %[[RES_M1:.*]] = "tfl.cumsum"(%arg0, %[[AXIS_M1]]) <{exclusive = false, reverse = false}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: %[[RES_P1:.*]] = "tfl.cumsum"(%arg0, %[[AXIS_P1]]) <{exclusive = false, reverse = false}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: %[[RES_P3:.*]] = "tfl.cumsum"(%arg0, %[[AXIS_P3]]) <{exclusive = false, reverse = false}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: return %arg0, %[[RES_M3]], %arg0, %[[RES_M1]], %arg0, %[[RES_P1]], %arg0, %[[RES_P3]] : tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>
}

// CHECK-LABEL:   func @eliminateCumSumCheckAttributes
func.func @eliminateCumSumCheckAttributes(%arg: tensor<1x2x1x3xf32>) -> (tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>) {
  %axis = arith.constant dense<2> : tensor<i32>
  %res_ff = "tfl.cumsum"(%arg, %axis) {exclusive = false, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_ft = "tfl.cumsum"(%arg, %axis) {exclusive = false, reverse =  true} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>  // Eliminated
  %res_tf = "tfl.cumsum"(%arg, %axis) {exclusive =  true, reverse = false} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  %res_tt = "tfl.cumsum"(%arg, %axis) {exclusive =  true, reverse =  true} : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  func.return %res_ff, %res_ft, %res_tf, %res_tt: tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>

  // CHECK: %[[AXIS:.*]] = arith.constant dense<2> : tensor<i32>
  // CHECK: %[[RES_TF:.*]] = "tfl.cumsum"(%arg0, %[[AXIS]]) <{exclusive = true, reverse = false}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: %[[RES_TT:.*]] = "tfl.cumsum"(%arg0, %[[AXIS]]) <{exclusive = true, reverse = true}> : (tensor<1x2x1x3xf32>, tensor<i32>) -> tensor<1x2x1x3xf32>
  // CHECK: return %arg0, %arg0, %[[RES_TF]], %[[RES_TT]] : tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>
}

func.func @gelu(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.707106769> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tf.Erf"(%1) : (tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.add"(%2, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %4 : tensor<3xf32>

// CHECK-LABEL:gelu
// CHECK: "tfl.gelu"(%arg0) <{approximate = false}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_erfc(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.707106769> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %2 = "tfl.neg"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tf.Erfc"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.mul"(%arg0, %cst_0) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%5, %4) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %6 : tensor<3xf32>

// CHECK-LABEL:gelu_erfc
// CHECK: "tfl.gelu"(%arg0) <{approximate = false}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_no_match(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.707106769> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_not_1 = arith.constant dense<3.000000e+00> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tf.Erf"(%1) : (tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.add"(%2, %cst_not_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %4 : tensor<3xf32>

// CHECK-LABEL:gelu_no_match
// CHECK: "tf.Erf"
}

func.func @gelu_no_match_not_oneuse(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.707106769> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tf.Erf"(%1) : (tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.add"(%2, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>

  %5 = "tfl.add"(%4, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %5 : tensor<3xf32>

// CHECK-LABEL:gelu_no_match_not_oneuse
// CHECK: "tf.Erf"
}

func.func @gelu_approximate(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst_2) {device = ""} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%6, %5) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_approximate_with_mul(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %99 = "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %0 = "tfl.mul"(%99, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%6, %5) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_approximate_with_mul2(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %99 = "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %0 = "tfl.mul"(%arg0, %99) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%6, %5) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_approximate1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst_2) {device = ""} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%5, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%arg0, %6) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_approximate1_with_mul(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %99 = "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %0 = "tfl.mul"(%99, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%5, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%arg0, %6) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}


func.func @gelu_approximate1_with_mul1(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %99 = "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %0 = "tfl.mul"(%arg0, %99) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%5, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%arg0, %6) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.gelu"(%arg0) <{approximate = true}> : (tensor<3xf32>) -> tensor<3xf32>
}

func.func @gelu_approximate_no_match(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_not_1 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst_2) {device = ""} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3)  {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst)  {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_not_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%arg0, %cst_not_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%6, %5) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %7 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate_no_match
// CHECK: "tfl.tanh"
}

func.func @gelu_approximate_not_oneuse(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = arith.constant dense<0.797884583> : tensor<f32>
  %cst_0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_2 = arith.constant dense<3.000000e+00> : tensor<f32>
  %cst_3 = arith.constant dense<4.471500e-02> : tensor<f32>
  %0 = "tfl.pow"(%arg0, %cst_2) {device = ""} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tfl.mul"(%0, %cst_3) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = "tfl.add"(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %3 = "tfl.mul"(%2, %cst) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %4 = "tfl.tanh"(%3) : (tensor<3xf32>) -> tensor<3xf32>
  %5 = "tfl.add"(%4, %cst_1) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %6 = "tfl.mul"(%arg0, %cst_0) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
  %7 = "tfl.mul"(%6, %5) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>

  %8 = "tfl.add"(%7, %2) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %8 : tensor<3xf32>

// CHECK-LABEL:gelu_approximate
// CHECK: "tfl.tanh"
}

// CHECK-LABEL:   func @eliminateExtraSelectLhs
func.func @eliminateExtraSelectLhs(%arg0: tensor<4x2x1xf32>, %arg1: tensor<4x2x1xi1>) -> (tensor<4x2x1xf32>) {
  %cst0 = arith.constant dense<1.0> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst_zeros = arith.constant dense<0.0> : tensor<4x2x1xf32>

  %0 = "tfl.select_v2"(%arg1, %cst_zeros, %arg0) : (tensor<4x2x1xi1>, tensor<4x2x1xf32>, tensor<4x2x1xf32>) -> tensor<4x2x1xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<4x2x1xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2x1xf32>
  %2 = "tfl.select_v2"(%arg1, %cst_zeros, %1) : (tensor<4x2x1xi1>, tensor<4x2x1xf32>, tensor<4x2x1xf32>) -> tensor<4x2x1xf32>

  func.return %2 : tensor<4x2x1xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<2.000000e+00> : tensor<2xf32>
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[CST]], %[[CST_1]]) <{asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x2x1xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2x1xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = "tfl.select_v2"
  // CHECK-NEXT: return %[[SELECT]]
}

// CHECK-LABEL:   func @eliminateExtraSelectRhs
func.func @eliminateExtraSelectRhs(%arg0: tensor<4x2x1xf32>, %arg1: tensor<4x2x1xi1>) -> (tensor<4x2x1xf32>) {
  %cst0 = arith.constant dense<1.0> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst_zeros = arith.constant dense<0.0> : tensor<4x2x1xf32>

  %0 = "tfl.select_v2"(%arg1, %arg0, %cst_zeros) : (tensor<4x2x1xi1>, tensor<4x2x1xf32>, tensor<4x2x1xf32>) -> tensor<4x2x1xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<4x2x1xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2x1xf32>
  %2 = "tfl.select_v2"(%arg1, %1, %cst_zeros) : (tensor<4x2x1xi1>, tensor<4x2x1xf32>, tensor<4x2x1xf32>) -> tensor<4x2x1xf32>

  func.return %2 : tensor<4x2x1xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<2.000000e+00> : tensor<2xf32>
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[CST]], %[[CST_1]]) <{asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x2x1xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2x1xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = "tfl.select_v2"
  // CHECK-NEXT: return %[[SELECT]]
}

// CHECK-LABEL:   func @DontEliminateExtraSelect
func.func @DontEliminateExtraSelect(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi1>) -> (tensor<4x2xf32>) {
  %cst0 = arith.constant dense<1.0> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>
  %cst_zeros = arith.constant dense<0.0> : tensor<4x2xf32>

  // Select's last dimension isn't 1
  %0 = "tfl.select_v2"(%arg1, %arg0, %cst_zeros) : (tensor<4x2xi1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %1 = "tfl.fully_connected"(%0, %cst0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  %2 = "tfl.select_v2"(%arg1, %1, %cst_zeros) : (tensor<4x2xi1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>

  func.return %2 : tensor<4x2xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<2.000000e+00> : tensor<2xf32>
  // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
  // CHECK: %[[SELECT:.*]] = "tfl.select_v2"(%arg1, %arg0, %[[CST_2]]) : (tensor<4x2xi1>, tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%[[SELECT]], %[[CST]], %[[CST_1]]) <{asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: %[[SELECT_1:.*]] = "tfl.select_v2"
  // CHECK-NEXT: return %[[SELECT_1]]
}

// CHECK-LABEL:   func @fuseReluToMin1_StaticShapeWithBroadcastedCst_Float1
func.func @fuseReluToMin1_StaticShapeWithBroadcastedCst_Float1(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
  %cst0 = arith.constant dense<0.0> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %cst0) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %cst1 = arith.constant dense<1.0> : tensor<f32>
  %1 = "tfl.minimum"(%0, %cst1) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>

  func.return %1 : tensor<2x2xf32>
  // CHECK-NOT: "tfl.relu"
  // CHECK-NOT: "tfl.minimum"
  // CHECK-NOT: "tfl.pseudo_const"
  // CHECK: "tfl.relu_0_to_1"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
}

// CHECK-LABEL:   func @fuseReluToMin1_StaticShapeWithBroadcastedCst_Float2
func.func @fuseReluToMin1_StaticShapeWithBroadcastedCst_Float2(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
  %cst0 = arith.constant dense<1.0> : tensor<f32>
  %0 = "tfl.minimum"(%arg0, %cst0) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  %cst1 = arith.constant dense<0.0> : tensor<f32>
  %1 = "tfl.maximum"(%0, %cst1) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>

  func.return %1 : tensor<2x2xf32>
  // CHECK-NOT: "tfl.relu"
  // CHECK-NOT: "tfl.minimum"
  // CHECK-NOT: "tfl.pseudo_const"
  // CHECK: "tfl.relu_0_to_1"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
}

// CHECK-LABEL:   func @fuseReluToMin1_StaticShapeWithSameShapeCst_Float
func.func @fuseReluToMin1_StaticShapeWithSameShapeCst_Float2(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>) {
  %cst0 = arith.constant dense<1.0> : tensor<2x2xf32>
  %0 = "tfl.minimum"(%arg0, %cst0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %cst1 = arith.constant dense<0.0> : tensor<2x2xf32>
  %1 = "tfl.maximum"(%0, %cst1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

  func.return %1 : tensor<2x2xf32>
  // CHECK-NOT: "tfl.relu"
  // CHECK-NOT: "tfl.minimum"
  // CHECK-NOT: "tfl.pseudo_const"
  // CHECK: "tfl.relu_0_to_1"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
}



// CHECK-LABEL:   func @fuseAddAndStridedSlice
func.func @fuseAddAndStridedSlice(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[cst:.*]] = arith.constant dense<1> : tensor<1xi32>
  // CHECK-DAG:  %[[c0:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK:  %1 = "tfl.strided_slice"(%arg0, %arg1, %[[cst]], %[[c0]]) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = true, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst_0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tfl.add"(%arg1, %cst_0) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @fuseSubAndStridedSlice
func.func @fuseSubAndStridedSlice(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[cst:.*]] = arith.constant dense<1> : tensor<1xi32>
  // CHECK-DAG:  %[[c0:.*]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK:  %1 = "tfl.strided_slice"(%arg0, %arg1, %[[cst]], %[[c0]]) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = true, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst_0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tfl.sub"(%arg1, %cst_0) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceNonConstantStride
func.func @dontFuseAddAndStridedSliceNonConstantStrides(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %0 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK:  %1 = tfl.add(%arg1, %0) <{fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:  %2 = "tfl.strided_slice"(%arg0, %arg1, %1, %arg2) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.add"(%arg1, %cst) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %arg2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceOffset
func.func @dontFuseAddAndStridedSliceOffset(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %0 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK:  %1 = tfl.add(%arg2, %0) <{fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:  %2 = "tfl.strided_slice"(%arg0, %arg1, %1, %arg3) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.add"(%arg2, %cst) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceNonConstantOffset
func.func @dontFuseAddAndStridedSliceNonConstantOffset(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK:  %0 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<1xi32>
  // CHECK: "tfl.strided_slice"(%arg0, %arg1, %0, %arg2) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %0 = "tfl.add"(%arg1, %arg1) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %arg2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceBeginMask
func.func @dontFuseAddAndStridedSliceBeginMask(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %0 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-DAG:  %1 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK:  %2 = tfl.add(%arg1, %0) <{fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:  %3 = "tfl.strided_slice"(%arg0, %arg1, %2, %1) <{begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst_0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tfl.add"(%arg1, %cst_0) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %cst_1) {begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceEndMask
func.func @dontFuseAddAndStridedSliceEndMask(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %0 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-DAG:  %1 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK:  %2 = tfl.add(%arg1, %0) <{fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:  %3 = "tfl.strided_slice"(%arg0, %arg1, %2, %1) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst_0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tfl.add"(%arg1, %cst_0) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @dontFuseAddAndStridedSliceEllipsisMask
func.func @dontFuseAddAndStridedSliceEllipsisMask(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %0 = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-DAG:  %1 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK:  %2 = tfl.add(%arg1, %0) <{fused_activation_function = "NONE"}> : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  // CHECK:  %3 = "tfl.strided_slice"(%arg0, %arg1, %2, %1) <{begin_mask = 0 : i32, ellipsis_mask = 1 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>

  %cst_0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tfl.add"(%arg1, %cst_0) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %1 = "tfl.strided_slice"(%arg0, %arg1, %0, %cst_1) {begin_mask = 0 : i32, ellipsis_mask = 1 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL:   func @fuseSigmoid
func.func @fuseSigmoid(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: "tfl.logistic"
  %cst = arith.constant dense<1.000000e+00> : tensor<10xf32>
  %0 = "tfl.neg"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
  %1 = "tfl.exp"(%0) : (tensor<10xf32>) -> tensor<10xf32>
  %2 = tfl.add %1, %cst {fused_activation_function = "NONE"} : tensor<10xf32>
  %3 = tfl.div %cst, %2 {fused_activation_function = "NONE"} : tensor<10xf32>
  return %3 : tensor<10xf32>
}

// CHECK-LABEL:   func @fuseElu
func.func @fuseElu(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_tf_0", outputs = "Identity_1"}} {
  // CHECK: "tfl.elu"
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<10xf32>
  %0 = tfl.greater(%arg0, %cst_0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %1 = "tfl.select"(%0, %cst_0, %arg0) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %2 = "tfl.exp"(%1) : (tensor<10xf32>) -> tensor<10xf32>
  %3 = tfl.sub(%2, %cst) {fused_activation_function = "NONE"} : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
  %4 = "tfl.select"(%0, %arg0, %3) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %4 : tensor<10xf32>
}

// CHECK-LABEL:   func @fuseHardSwishJAX
func.func @fuseHardSwishJAX(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_tf_0", outputs = "Identity_1"}} {
  // CHECK: "tfl.hard_swish"
  %cst = arith.constant dense<3.000000e+00> : tensor<10xf32>
  %cst_0 = arith.constant dense<6.000000e+00> : tensor<10xf32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<10xf32>
  %1 = "tfl.relu"(%0) : (tensor<10xf32>) -> tensor<10xf32>
  %2 = "tfl.minimum"(%1, %cst_0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  %3 = tfl.div %2, %cst_0 {fused_activation_function = "NONE"} : tensor<10xf32>
  %4 = tfl.mul %arg0, %3 {fused_activation_function = "NONE"} : tensor<10xf32>
  return %4 : tensor<10xf32>
}

// CHECK-LABEL:   func @fuseLeakyRelu
func.func @fuseLeakyRelu(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_tf_0", outputs = "Identity_1"}} {
  // CHECK: "tfl.leaky_relu"
  %cst = arith.constant dense<0.000000e+00> : tensor<10xf32>
  %cst_0 = arith.constant dense<[0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977]> : tensor<10xf32>
  %0 = tfl.greater_equal(%arg0, %cst) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %1 = tfl.mul %arg0, %cst_0 {fused_activation_function = "NONE"} : tensor<10xf32>
  %2 = "tfl.select"(%0, %arg0, %1) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL:   func @fuseLeakyReluNotSplat
func.func @fuseLeakyReluNotSplat(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_tf_0", outputs = "Identity_1"}} {
  // CHECK-NOT: "tfl.leaky_relu"
  %cst = arith.constant dense<0.000000e+00> : tensor<10xf32>
  %cst_0 = arith.constant dense<[0.000000e+00, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977, 0.00999999977]> : tensor<10xf32>
  %0 = tfl.greater_equal(%arg0, %cst) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  %1 = tfl.mul %arg0, %cst_0 {fused_activation_function = "NONE"} : tensor<10xf32>
  %2 = "tfl.select"(%0, %arg0, %1) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL:   func @fuse_2d_upscaling
func.func @fuse_2d_upscaling(%arg0: tensor<8x1x8x1280xf32>) -> tensor<1x16x16x1280xf32> {
  // CHECK: "tfl.resize_nearest_neighbor"
  %cst = "tfl.pseudo_const"() {value = dense<[[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6], [7], [7]]> : tensor<16x1xi32>} : () -> tensor<16x1xi32>
  %gather_nd_first = "tfl.gather_nd"(%arg0, %cst) : (tensor<8x1x8x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x8x1280xf32>
  %transpose_first_perm = "tfl.pseudo_const"() {value = dense<[2, 1, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_first = "tfl.transpose"(%gather_nd_first, %transpose_first_perm) : (tensor<16x1x8x1280xf32>, tensor<4xi32>) -> tensor<8x1x16x1280xf32>
  %gather_nd_second = "tfl.gather_nd"(%transpose_first, %cst) : (tensor<8x1x16x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x16x1280xf32>
  %transpose_second_perm = "tfl.pseudo_const"() {value = dense<[1, 2, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_second = "tfl.transpose"(%gather_nd_second, %transpose_second_perm) : (tensor<16x1x16x1280xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
  return %transpose_second : tensor<1x16x16x1280xf32>
}

// CHECK-LABEL:   func @dont_fuse_non_2d_upscaling_wrong_indices
func.func @dont_fuse_non_2d_upscaling_wrong_indices(%arg0: tensor<8x1x8x1280xf32>) -> tensor<1x16x16x1280xf32> {
  // CHECK-NOT: "tfl.resize_nearest_neighbor"
  %cst = "tfl.pseudo_const"() {value = dense<[[1], [1], [1], [1], [2], [2], [2], [2], [4], [4], [4], [4], [5], [5], [5], [5]]> : tensor<16x1xi32>} : () -> tensor<16x1xi32>
  %gather_nd_first = "tfl.gather_nd"(%arg0, %cst) : (tensor<8x1x8x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x8x1280xf32>
  %transpose_first_perm = "tfl.pseudo_const"() {value = dense<[2, 1, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_first = "tfl.transpose"(%gather_nd_first, %transpose_first_perm) : (tensor<16x1x8x1280xf32>, tensor<4xi32>) -> tensor<8x1x16x1280xf32>
  %gather_nd_second = "tfl.gather_nd"(%transpose_first, %cst) : (tensor<8x1x16x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x16x1280xf32>
  %transpose_second_perm = "tfl.pseudo_const"() {value = dense<[1, 2, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_second = "tfl.transpose"(%gather_nd_second, %transpose_second_perm) : (tensor<16x1x16x1280xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
  return %transpose_second : tensor<1x16x16x1280xf32>
}

// CHECK-LABEL:   func @dont_fuse_non_2d_upscaling_wrong_perm
func.func @dont_fuse_non_2d_upscaling_wrong_perm(%arg0: tensor<8x1x8x1280xf32>) -> tensor<1x16x16x1280xf32> {
  // CHECK-NOT: "tfl.resize_nearest_neighbor"
  %cst = "tfl.pseudo_const"() {value = dense<[[0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6], [7], [7]]> : tensor<16x1xi32>} : () -> tensor<16x1xi32>
  %gather_nd_first = "tfl.gather_nd"(%arg0, %cst) : (tensor<8x1x8x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x8x1280xf32>
  %transpose_first_perm = "tfl.pseudo_const"() {value = dense<[2, 1, 0, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_first = "tfl.transpose"(%gather_nd_first, %transpose_first_perm) : (tensor<16x1x8x1280xf32>, tensor<4xi32>) -> tensor<8x1x16x1280xf32>
  %gather_nd_second = "tfl.gather_nd"(%transpose_first, %cst) : (tensor<8x1x16x1280xf32>, tensor<16x1xi32>) -> tensor<16x1x16x1280xf32>
  %transpose_second_perm = "tfl.pseudo_const"() {value = dense<[1, 0, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %transpose_second = "tfl.transpose"(%gather_nd_second, %transpose_second_perm) : (tensor<16x1x16x1280xf32>, tensor<4xi32>) -> tensor<1x16x16x1280xf32>
  return %transpose_second : tensor<1x16x16x1280xf32>
}

// CHECK-LABEL: FuseReshapeAndTransposeAroundBatchMatmul
func.func @FuseReshapeAndTransposeAroundBatchMatmul(%arg0: tensor<1x128x1024xf32>, %arg1: tensor<1024x16xf32>) -> tensor<1x128x16xf32> {
  %cst_0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[16, 1, 128]> : tensor<3xi32>
  %cst_2 = arith.constant dense<[1024, 128]> : tensor<2xi32>
  %cst_3 = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst_3) : (tensor<1x128x1024xf32>, tensor<3xi32>) -> tensor<1024x1x128xf32>
  %1 = "tfl.reshape"(%0, %cst_2) : (tensor<1024x1x128xf32>, tensor<2xi32>) -> tensor<1024x128xf32>
  // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%arg0, %arg1) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}>
  // CHECK-NOT: tfl.reshape
  // CHECK-NOT: tfl.transpose
  %2 = "tfl.batch_matmul"(%arg1, %1) {adj_x = true, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1024x16xf32>, tensor<1024x128xf32>) -> tensor<16x128xf32>
  %3 = "tfl.reshape"(%2, %cst_1) : (tensor<16x128xf32>, tensor<3xi32>) -> tensor<16x1x128xf32>
  %4 = "tfl.transpose"(%3, %cst_0) : (tensor<16x1x128xf32>, tensor<3xi32>) -> tensor<1x128x16xf32>
  func.return %4 : tensor<1x128x16xf32>
  // CHECK: return %[[BMM]] : tensor<1x128x16xf32>
}

// CHECK-LABEL: FuseReshapeAndTransposeAroundBatchMatmulWithLargerThan3Rank
func.func @FuseReshapeAndTransposeAroundBatchMatmulWithLargerThan3Rank(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<16x1024xf32>) -> tensor<1x128x16xf32> {
  %cst_0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[16, 1, 128]> : tensor<3xi32>
  %cst_2 = arith.constant dense<[1024, 128]> : tensor<2xi32>
  %cst_3 = arith.constant dense<[2, 3, 0, 1]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst_3) : (tensor<1x128x4x256xf32>, tensor<4xi32>) -> tensor<4x256x1x128xf32>
  %1 = "tfl.reshape"(%0, %cst_2) : (tensor<4x256x1x128xf32>, tensor<2xi32>) -> tensor<1024x128xf32>
  // CHECK: %[[RESHAE_ARG0:.*]] = "tfl.reshape"(%arg0, %[[CST:.*]]) : (tensor<1x128x4x256xf32>, tensor<3xi32>) -> tensor<1x128x1024xf32>
  // CHECK: %[[BMM:.*]] = "tfl.batch_matmul"(%[[RESHAE_ARG0]], %arg1) <{adj_x = false, adj_y = true, asymmetric_quantize_inputs = false}>
  // CHECK-NOT: tfl.reshape
  // CHECK-NOT: tfl.transpose
  %2 = "tfl.batch_matmul"(%arg1, %1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<16x1024xf32>, tensor<1024x128xf32>) -> tensor<16x128xf32>
  %3 = "tfl.reshape"(%2, %cst_1) : (tensor<16x128xf32>, tensor<3xi32>) -> tensor<16x1x128xf32>
  %4 = "tfl.transpose"(%3, %cst_0) : (tensor<16x1x128xf32>, tensor<3xi32>) -> tensor<1x128x16xf32>
  func.return %4 : tensor<1x128x16xf32>
  // CHECK: return %[[BMM]] : tensor<1x128x16xf32>
}

// CHECK-LABEL: NotFuseTransposeFCRhsToBatchMatmul
func.func @NotFuseTransposeFCRhsToBatchMatmul(%arg0: tensor<16x1024xf32>, %arg1: tensor<1024x128x!quant.uniform<u8:f32, 0.038859266393324911:129>>, %arg2: none) -> tensor<16x128xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %dequant = "tfl.dequantize"(%arg1) : (tensor<1024x128x!quant.uniform<u8:f32, 0.038859266393324911:129>>) -> tensor<1024x128xf32>
  %0 = "tfl.transpose"(%dequant, %cst) : (tensor<1024x128xf32>, tensor<2xi32>) -> tensor<128x1024xf32>
  // CHECK: tfl.fully_connected
  // CHECK-NOT: tfl.batch_matmul
  %1 = "tfl.fully_connected"(%arg0, %0, %arg2) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<16x1024xf32>, tensor<128x1024xf32>, none) -> tensor<16x128xf32>
  func.return %1 : tensor<16x128xf32>
}

// CHECK-LABEL: NotFuseTransposeFCRhsFromSplitToBatchMatmul
func.func @NotFuseTransposeFCRhsFromSplitToBatchMatmul(%arg0: tensor<16x1024xf32>, %arg1: tensor<1024x256x!quant.uniform<u8:f32, 0.038859266393324911:129>>, %arg2: none) -> tensor<16x128xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_1 = arith.constant dense<1> : tensor<i32>
  %dequant = "tfl.dequantize"(%arg1) : (tensor<1024x256x!quant.uniform<u8:f32, 0.038859266393324911:129>>) -> tensor<1024x256xf32>
  %split:2 = "tfl.split"(%cst_1, %dequant) {num_splits = 2 : i32} : (tensor<i32>, tensor<1024x256xf32>) -> (tensor<1024x128xf32>, tensor<1024x128xf32>)
  %0 = "tfl.transpose"(%split#0, %cst) : (tensor<1024x128xf32>, tensor<2xi32>) -> tensor<128x1024xf32>
  // CHECK: tfl.fully_connected
  // CHECK-NOT: tfl.batch_matmul
  %1 = "tfl.fully_connected"(%arg0, %0, %arg2) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<16x1024xf32>, tensor<128x1024xf32>, none) -> tensor<16x128xf32>
  func.return %1 : tensor<16x128xf32>
}

// CHECK-LABEL: FuseTransposeReshapeIntoBatchMatmul
func.func @FuseTransposeReshapeIntoBatchMatmul(%arg0: tensor<4x1024xf32>, %arg1: tensor<8x4x256xf32>, %arg2: none) -> tensor<4x8xf32> {
  %cst_0 = arith.constant dense<[1024, 8]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg1, %cst_1) : (tensor<8x4x256xf32>, tensor<3xi32>) -> tensor<4x256x8xf32>
  %1 = "tfl.reshape"(%0, %cst_0) : (tensor<4x256x8xf32>, tensor<2xi32>) -> tensor<1024x8xf32>
  // CHECK: %[[RES0:.*]] = "tfl.reshape"(%arg1, %[[CST:.*]]) : (tensor<8x4x256xf32>, tensor<2xi32>) -> tensor<8x1024xf32>
  // CHECK: %[[RES1:.*]] = "tfl.batch_matmul"(%arg0, %[[RES0]]) <{adj_x = false, adj_y = true, asymmetric_quantize_inputs = false}> : (tensor<4x1024xf32>, tensor<8x1024xf32>) -> tensor<4x8xf32>
  %2 = "tfl.batch_matmul"(%arg0, %1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<4x1024xf32>, tensor<1024x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
  // CHECK: return %[[RES1]] : tensor<4x8xf32>
}

// CHECK-LABEL: FuseTransposeAfterBatchMatmul
func.func @FuseTransposeAfterBatchMatmul(%arg0: tensor<4x1024xf32>, %arg1: tensor<8x1024xf32>, %arg2: none) -> tensor<8x4xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %[[RES0:.*]] = "tfl.batch_matmul"(%arg1, %arg0) <{adj_x = false, adj_y = true, asymmetric_quantize_inputs = false}> : (tensor<8x1024xf32>, tensor<4x1024xf32>) -> tensor<8x4xf32>
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = true, asymmetric_quantize_inputs = false} : (tensor<4x1024xf32>, tensor<8x1024xf32>) -> tensor<4x8xf32>
  %1 = "tfl.transpose"(%0, %cst) : (tensor<4x8xf32>, tensor<2xi32>) -> tensor<8x4xf32>
  func.return %1 : tensor<8x4xf32>
  // CHECK: return %[[RES0]] : tensor<8x4xf32>
}

// CHECK-LABEL: fuseLogSoftmax
func.func @fuseLogSoftmax(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = arith.constant dense<1> : tensor<1xi32>
  %1 = "tfl.reduce_max"(%arg0, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %2 = tfl.sub(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  %3 = "tfl.exp"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %4 = "tfl.sum"(%3, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %5 = "tfl.log"(%4) : (tensor<10x1xf32>) -> tensor<10x1xf32>
  %6 = tfl.sub(%2, %5) {fused_activation_function = "NONE"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %6 : tensor<10x10xf32>
 // CHECK: "tfl.log_softmax"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// CHECK-LABEL: fuseLogSoftmaxAxisNegativeOne
func.func @fuseLogSoftmaxAxisNegativeOne(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = arith.constant dense<-1> : tensor<1xi32>
  %1 = "tfl.reduce_max"(%arg0, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %2 = tfl.sub(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  %3 = "tfl.exp"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %4 = "tfl.sum"(%3, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %5 = "tfl.log"(%4) : (tensor<10x1xf32>) -> tensor<10x1xf32>
  %6 = tfl.sub(%2, %5) {fused_activation_function = "NONE"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %6 : tensor<10x10xf32>
 // CHECK: "tfl.log_softmax"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// CHECK-LABEL: fuseLogSoftmaxFusedActivationFunction
func.func @fuseLogSoftmaxFusedActivationFunction(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = arith.constant dense<1> : tensor<1xi32>
  %1 = "tfl.reduce_max"(%arg0, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %2 = tfl.sub(%arg0, %1) {fused_activation_function = "RELU"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  %3 = "tfl.exp"(%2) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %4 = "tfl.sum"(%3, %0) {keep_dims = true} : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<10x1xf32>
  %5 = "tfl.log"(%4) : (tensor<10x1xf32>) -> tensor<10x1xf32>
  %6 = tfl.sub(%2, %5) {fused_activation_function = "RELU"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<10x10xf32>
  return %6 : tensor<10x10xf32>
 // CHECK-NOT: "tfl.log_softmax"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// CHECK-LABEL: fuseLogSoftmax1D
func.func @fuseLogSoftmax1D(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = arith.constant dense<0> : tensor<1xi32>
  %1 = "tfl.reduce_max"(%arg0, %0) {keep_dims = true} : (tensor<10xf32>, tensor<1xi32>) -> tensor<1xf32>
  %2 = tfl.sub(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  %3 = "tfl.exp"(%2) : (tensor<10xf32>) -> tensor<10xf32>
  %4 = "tfl.sum"(%3, %0) {keep_dims = true} : (tensor<10xf32>, tensor<1xi32>) -> tensor<1xf32>
  %5 = "tfl.log"(%4) : (tensor<1xf32>) -> tensor<1xf32>
  %6 = tfl.sub(%2, %5) {fused_activation_function = "NONE"} : (tensor<10xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %6 : tensor<10xf32>
 // CHECK: "tfl.log_softmax"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
}

// CHECK-LABEL: fuseLogSoftmaxNotLastAxis
func.func @fuseLogSoftmaxNotLastAxis(%arg0: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  %0 = arith.constant dense<1> : tensor<1xi32>
  %1 = "tfl.reduce_max"(%arg0, %0) {keep_dims = true} : (tensor<10x10x10xf32>, tensor<1xi32>) -> tensor<10x1x10xf32>
  %2 = tfl.sub(%arg0, %1) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x1x10xf32>) -> tensor<10x10x10xf32>
  %3 = "tfl.exp"(%2) : (tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
  %4 = "tfl.sum"(%3, %0) {keep_dims = true} : (tensor<10x10x10xf32>, tensor<1xi32>) -> tensor<10x1x10xf32>
  %5 = "tfl.log"(%4) : (tensor<10x1x10xf32>) -> tensor<10x1x10xf32>
  %6 = tfl.sub(%2, %5) {fused_activation_function = "NONE"} : (tensor<10x10x10xf32>, tensor<10x1x10xf32>) -> tensor<10x10x10xf32>
  return %6 : tensor<10x10x10xf32>
 // CHECK-NOT: "tfl.log_softmax"(%arg0) : (tensor<10x10x10f32>) -> tensor<10x10x10xf32>
}

// CHECK-LABEL: @ReorderNCHWTransposeAddForConv
func.func @ReorderNCHWTransposeAddForConv(%arg0: tensor<1x40x40x1xf32>, %filter: tensor<3x3x3x3xf32>) -> tensor<1x3x40x40xf32> {
  %no_bias = "tfl.no_value"() {value} : () -> none
  %perm = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %bias = arith.constant dense<1.5> : tensor<1x3x1x1xf32>
  %0 = "tfl.conv_2d"(%arg0, %filter, %no_bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x40x40x1xf32>, tensor<3x3x3x3xf32>, none) -> tensor<1x40x40x3xf32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<1x40x40x3xf32>, tensor<4xi32>) -> tensor<1x3x40x40xf32>
  %2 = "tfl.add"(%1, %bias) {fused_activation_function = "NONE"} : (tensor<1x3x40x40xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x40x40xf32>
  func.return %2 : tensor<1x3x40x40xf32>

  // CHECK: %[[perm:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  // CHECK: %[[conv:.*]] = "tfl.conv_2d"
  // CHECK: %[[add:.*]] = tfl.add(%[[conv]]
  // CHECK: %[[transpose:.*]] = "tfl.transpose"(%[[add]], %[[perm]])
  // CHECK: return %[[transpose]]
}

// CHECK-LABEL: @NoReorderNCHWTransposeAddNotConv
func.func @NoReorderNCHWTransposeAddNotConv(%arg0: tensor<1x40x40x3xf32>, %filter: tensor<3x3x3x3xf32>) -> tensor<1x3x40x40xf32> {
  %no_bias = "tfl.no_value"() {value} : () -> none
  %perm = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %bias = arith.constant dense<1.5> : tensor<1x3x1x1xf32>
  %1 = "tfl.transpose"(%arg0, %perm) : (tensor<1x40x40x3xf32>, tensor<4xi32>) -> tensor<1x3x40x40xf32>
  %2 = "tfl.add"(%1, %bias) {fused_activation_function = "NONE"} : (tensor<1x3x40x40xf32>, tensor<1x3x1x1xf32>) -> tensor<1x3x40x40xf32>
  func.return %2 : tensor<1x3x40x40xf32>

  // CHECK: %[[transpose:.*]] = "tfl.transpose"
  // CHECK: %[[add:.*]] = tfl.add(%[[transpose]],
  // CHECK: return %[[add]]
}

// CHECK-LABEL: @NoReorderNCHWTransposeAddNotNCHW
func.func @NoReorderNCHWTransposeAddNotNCHW(%arg0: tensor<1x40x40x1xf32>, %filter: tensor<3x3x3x3xf32>) -> tensor<1x40x3x40xf32> {
  %no_bias = "tfl.no_value"() {value} : () -> none
  %perm = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  %bias = arith.constant dense<1.5> : tensor<1x1x3x1xf32>
  %0 = "tfl.conv_2d"(%arg0, %filter, %no_bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x40x40x1xf32>, tensor<3x3x3x3xf32>, none) -> tensor<1x40x40x3xf32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<1x40x40x3xf32>, tensor<4xi32>) -> tensor<1x40x3x40xf32>
  %2 = "tfl.add"(%1, %bias) {fused_activation_function = "NONE"} : (tensor<1x40x3x40xf32>, tensor<1x1x3x1xf32>) -> tensor<1x40x3x40xf32>
  func.return %2 : tensor<1x40x3x40xf32>

  // CHECK: %[[transpose:.*]] = "tfl.transpose"
  // CHECK: %[[add:.*]] = tfl.add(%[[transpose]],
  // CHECK: return %[[add]]
}


// CHECK-LABEL: @NoReorderNCHWTransposeAddNotBias
func.func @NoReorderNCHWTransposeAddNotBias(%arg0: tensor<1x40x40x1xf32>, %filter: tensor<3x3x3x3xf32>) -> tensor<1x3x40x40xf32> {
  %no_bias = "tfl.no_value"() {value} : () -> none
  %perm = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %bias = arith.constant dense<1.5> : tensor<1x3x40x40xf32>
  %0 = "tfl.conv_2d"(%arg0, %filter, %no_bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x40x40x1xf32>, tensor<3x3x3x3xf32>, none) -> tensor<1x40x40x3xf32>
  %1 = "tfl.transpose"(%0, %perm) : (tensor<1x40x40x3xf32>, tensor<4xi32>) -> tensor<1x3x40x40xf32>
  %2 = "tfl.add"(%1, %bias) {fused_activation_function = "NONE"} : (tensor<1x3x40x40xf32>, tensor<1x3x40x40xf32>) -> tensor<1x3x40x40xf32>
  func.return %2 : tensor<1x3x40x40xf32>

  // CHECK: %[[transpose:.*]] = "tfl.transpose"
  // CHECK: %[[add:.*]] = tfl.add %[[transpose]],
  // CHECK: return %[[add]]
}

// CHECK-LABEL: @ConvertStridedSliceToSlice
func.func @ConvertStridedSliceToSlice(%arg0: tensor<2x3872x1x128xf32>) -> tensor<1x3872x1x128xf32> {
  %44 = arith.constant dense<0> : tensor<4xi32>
  %45 = arith.constant dense<[1, 3872, 1, 128]> : tensor<4xi32>
  %46 = arith.constant dense<1> : tensor<4xi32>
  %47 = "tfl.strided_slice"(%arg0, %44, %45, %46) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<2x3872x1x128xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x3872x1x128xf32>
  func.return %47 : tensor<1x3872x1x128xf32>

  // CHECK: %[[slice:.*]] = "tfl.slice"
  // CHECK: return %[[slice]]
}

// CHECK-LABEL: @FuseExcessBroadcastingOnReshapes
func.func @FuseExcessBroadcastingOnReshapes(%arg0: tensor<1x8xf32>) -> tensor<1x1x1x128xf32> {
    %cst = arith.constant dense<[1, 1, 1, 8, 1, 1]> : tensor<6xi32>
    %cst_0 = arith.constant dense<[1, 1, 1, 8, 16, 1]> : tensor<6xi32>
    %cst_1 = arith.constant dense<[1, 1, 1, 128]> : tensor<4xi32>
    %0 = "tfl.reshape"(%arg0, %cst) : (tensor<1x8xf32>, tensor<6xi32>) -> tensor<1x1x1x8x1x1xf32>
    %1 = "tfl.broadcast_to"(%0, %cst_0) : (tensor<1x1x1x8x1x1xf32>, tensor<6xi32>) -> tensor<1x1x1x8x16x1xf32>
    %2 = "tfl.reshape"(%1, %cst_1) : (tensor<1x1x1x8x16x1xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
    return %2 : tensor<1x1x1x128xf32>
    // CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
    // CHECK: %cst_0 = arith.constant dense<[1, 1, 1, 128]> : tensor<4xi32>
    // CHECK: %cst_1 = arith.constant dense<[8, 1]> : tensor<2xi32>
    // CHECK: %0 = "tfl.reshape"(%arg0, %cst_1) : (tensor<1x8xf32>, tensor<2xi32>) -> tensor<8x1xf32>
    // CHECK: %1 = tfl.mul(%0, %cst) <{fused_activation_function = "NONE"}> : (tensor<8x1xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    // CHECK: %2 = "tfl.reshape"(%1, %cst_0) : (tensor<8x16xf32>, tensor<4xi32>) -> tensor<1x1x1x128xf32>
    // CHECK: return %2 : tensor<1x1x1x128xf32>
}

// CHECK-LABEL: @FuseExcessBroadcastingOnReshapesDynamicShapes
func.func @FuseExcessBroadcastingOnReshapesDynamicShapes(%arg0: tensor<?x10x1xf32>, %arg1: tensor<6xi32>, %arg2: tensor<6xi32>, %arg3: tensor<2xi32>) -> tensor<?x50xf32> {
    %1196 = "tfl.reshape"(%arg0, %arg1) : (tensor<?x10x1xf32>, tensor<6xi32>) -> tensor<1x?x1x10x1x1xf32>
    %1197 = "tfl.broadcast_to"(%1196, %arg2) : (tensor<1x?x1x10x1x1xf32>, tensor<6xi32>) -> tensor<1x?x1x10x5x1xf32>
    %1198 = "tfl.reshape"(%1197, %arg3) : (tensor<1x?x1x10x5x1xf32>, tensor<2xi32>) -> tensor<?x50xf32>
    return %1198 : tensor<?x50xf32>

    // CHECK: %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<?x10x1xf32>, tensor<6xi32>) -> tensor<1x?x1x10x1x1xf32>
    // CHECK: %1 = "tfl.broadcast_to"(%0, %arg2) : (tensor<1x?x1x10x1x1xf32>, tensor<6xi32>) -> tensor<1x?x1x10x5x1xf32>
    // CHECK: %2 = "tfl.reshape"(%1, %arg3) : (tensor<1x?x1x10x5x1xf32>, tensor<2xi32>) -> tensor<?x50xf32>
    // CHECK: return %2 : tensor<?x50xf32>
}

// CHECK-LABEL: @broadcast_to_f32_low_dim
func.func @broadcast_to_f32_low_dim(%arg0: tensor<3xf32>, %arg1: tensor<2xi32>) -> tensor<3x3xf32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
  // CHECK:  %cst = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK:  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @broadcast_to_i32_low_dim
func.func @broadcast_to_i32_low_dim(%arg0: tensor<3xi32>, %arg1: tensor<2xi32>) -> tensor<3x3xi32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32>) -> tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
  // CHECK:  %cst = arith.constant dense<1> : tensor<3x3xi32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
  // CHECK:  return %0 : tensor<3x3xi32>
}

// CHECK-LABEL: @broadcast_to_low_dim_with_unknown_shape
func.func @broadcast_to_low_dim_with_unknown_shape(%arg0: tensor<3xf32>, %arg1: tensor<*xi32>) -> tensor<3x3xf32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xf32>, tensor<*xi32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
  // CHECK:  %cst = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK:  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @broadcast_to_i16_low_dim
func.func @broadcast_to_i16_low_dim(%arg0: tensor<3xi16>, %arg1: tensor<2xi32>) -> tensor<3x3xi16> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xi16>, tensor<2xi32>) -> tensor<3x3xi16>
  return %0 : tensor<3x3xi16>
  // CHECK:  %cst = arith.constant dense<1> : tensor<3x3xi16>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xi16>, tensor<3x3xi16>) -> tensor<3x3xi16>
  // CHECK:  return %0 : tensor<3x3xi16>
}

// CHECK-LABEL: @broadcast_to_i32_low_dim_with_unknown_output
func.func @broadcast_to_i32_low_dim_with_unknown_output(%arg0: tensor<3xi32>, %arg1: tensor<2xi32>) -> tensor<*xi32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
  // CHECK:  %cst = arith.constant dense<1> : tensor<i32>
  // CHECK:  %0 = "tfl.fill"(%arg1, %cst) : (tensor<2xi32>, tensor<i32>) -> tensor<*xi32>
  // CHECK:  %1 = tfl.mul(%arg0, %0) <{fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<*xi32>) -> tensor<*xi32>
  // CHECK:  return %1 : tensor<*xi32>
}

// CHECK-LABEL: @broadcast_to_ui32
func.func @broadcast_to_ui32(%arg0: tensor<ui32>, %arg1: tensor<1xi64>) -> tensor<10xui32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<ui32>, tensor<1xi64>) -> tensor<10xui32>
  return %0 : tensor<10xui32>
  // CHECK:  %cst = arith.constant dense<1> : tensor<10xui32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<ui32>, tensor<10xui32>) -> tensor<10xui32>
  // CHECK:  return %0 : tensor<10xui32>
}

// CHECK-LABEL: @broadcast_to_f32
func.func @broadcast_to_f32(%arg0: tensor<3xf32>, %arg1: tensor<2xi32>) -> tensor<3x3xf32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
  // CHECK:  %cst = arith.constant dense<1.000000e+00> : tensor<3x3xf32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK:  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @broadcast_to_i32
func.func @broadcast_to_i32(%arg0: tensor<3xi32>, %arg1: tensor<2xi32>) -> tensor<3x3xi32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32>) -> tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
  // CHECK:  %cst = arith.constant dense<1> : tensor<3x3xi32>
  // CHECK:  %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
  // CHECK:  return %0 : tensor<3x3xi32>
}

// CHECK-LABEL: @broadcast_to_i32_with_dynamic_shape_and_output
func.func @broadcast_to_i32_with_dynamic_shape_and_output(%arg0: tensor<3xi32>, %arg1: tensor<2xi32>) -> tensor<3x?xi32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32>) -> tensor<3x?xi32>
  return %0 : tensor<3x?xi32>
  // CHECK:  %cst = arith.constant dense<1> : tensor<i32>
  // CHECK:  %0 = "tfl.fill"(%arg1, %cst) : (tensor<2xi32>, tensor<i32>) -> tensor<3x?xi32>
  // CHECK:  %1 = tfl.mul(%arg0, %0) <{fused_activation_function = "NONE"}> : (tensor<3xi32>, tensor<3x?xi32>) -> tensor<3x?xi32>
  // CHECK:  return %1 : tensor<3x?xi32>
}

// CHECK-LABEL: @broadcast_to_ui32_with_dynamic_output
func.func @broadcast_to_ui32_with_dynamic_output(%arg0: tensor<1xi32>) -> tensor<?xui32> {
  %cst = arith.constant dense<0> : tensor<1xui32>
  %0 = "tfl.broadcast_to"(%cst, %arg0) : (tensor<1xui32>, tensor<1xi32>) -> tensor<?xui32>
  return %0 : tensor<?xui32>

  // CHECK:  %cst = arith.constant dense<0> : tensor<1xui32>
  // CHECK:  %0 = "tfl.broadcast_to"(%cst, %arg0) : (tensor<1xui32>, tensor<1xi32>) -> tensor<?xui32>
  // CHECK:  return %0 : tensor<?xui32>
}


// CHECK-LABEL: @ConvertStridedSliceToSliceNeg
func.func @ConvertStridedSliceToSliceNeg(%arg0: tensor<5x5x5x5xf32>) -> tensor<*xf32> {
  %44 = arith.constant dense<[5, 5, 5, 5]> : tensor<4xi32>
  %45 = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi32>
  %46 = arith.constant dense<1> : tensor<4xi32>
  %47 = "tfl.strided_slice"(%arg0, %44, %45, %46) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<5x5x5x5xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<*xf32>
  func.return %47 : tensor<*xf32>

  // CHECK-NOT: %[[slice:.*]] = "tfl.slice"
}

// CHECK-LABEL: @StridedSliceToSliceBeginNeg
func.func @StridedSliceToSliceBeginNeg(%arg0: tensor<5x5x5x5xf32>) -> tensor<*xf32> {
  %44 = arith.constant dense<[-5, 0, 0, 0]> : tensor<4xi32>
  %45 = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi32>
  %46 = arith.constant dense<1> : tensor<4xi32>
  %47 = "tfl.strided_slice"(%arg0, %44, %45, %46) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32} : (tensor<5x5x5x5xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<*xf32>
  func.return %47 : tensor<*xf32>

  // CHECK-NOT: %[[slice:.*]] = "tfl.slice"
}

// CHECK-LABEL: conv3d_external_padding
func.func @conv3d_external_padding(%arg0: tensor<1x7x7x7x128xf32>, %arg1: tensor<3x3x3x128x256xf32>) -> tensor<1x7x7x7x256xf32> {
  %cst = arith.constant dense<[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x7x7x7x128xf32>, tensor<5x2xi64>) -> tensor<1x9x9x9x128xf32>
  %1 = "tfl.conv_3d"(%0, %arg1, %cst_0) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x9x9x9x128xf32>, tensor<3x3x3x128x256xf32>, tensor<256xf32>) -> tensor<1x7x7x7x256xf32>
  return %1 : tensor<1x7x7x7x256xf32>
}

// CHECK: %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x7x7x7x128xf32>, tensor<3x3x3x128x256xf32>, tensor<256xf32>) -> tensor<1x7x7x7x256xf32>

// CHECK-LABEL: conv3d_external_padding_strided
func.func @conv3d_external_padding_strided(%arg0: tensor<1x8x56x56x128xf32>, %arg1: tensor<3x3x3x128x256xf32>) -> tensor<1x4x28x28x256xf32> {
  %cst = arith.constant dense<[[0, 0], [0, 1], [0, 1], [0, 1], [0, 0]]> : tensor<5x2xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x8x56x56x128xf32>, tensor<5x2xi64>) -> tensor<1x9x57x57x128xf32>
  %1 = "tfl.conv_3d"(%0, %arg1, %cst_0) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x9x57x57x128xf32>, tensor<3x3x3x128x256xf32>, tensor<256xf32>) -> tensor<1x4x28x28x256xf32>
  return %1 : tensor<1x4x28x28x256xf32>
}

// CHECK: %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x8x56x56x128xf32>, tensor<3x3x3x128x256xf32>, tensor<256xf32>) -> tensor<1x4x28x28x256xf32>

// CHECK-LABEL: conv2d_external_padding
func.func @conv2d_external_padding(%arg0: tensor<1x7x7x128xf32>, %arg1: tensor<256x3x3x128xf32>) -> tensor<1x7x7x256xf32> {
  %cst = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x7x7x128xf32>, tensor<4x2xi64>) -> tensor<1x9x9x128xf32>
  %1 = "tfl.conv_2d"(%0, %arg1, %cst_0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x9x9x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x7x7x256xf32>
  return %1 : tensor<1x7x7x256xf32>
}

// CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x7x7x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x7x7x256xf32>

// CHECK-LABEL: conv2d_external_padding_strided
func.func @conv2d_external_padding_strided(%arg0: tensor<1x8x8x128xf32>, %arg1: tensor<256x3x3x128xf32>) -> tensor<1x4x4x256xf32> {
  %cst = arith.constant dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x8x8x128xf32>, tensor<4x2xi64>) -> tensor<1x9x9x128xf32>
  %1 = "tfl.conv_2d"(%0, %arg1, %cst_0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x9x9x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x4x4x256xf32>
  return %1 : tensor<1x4x4x256xf32>
}

// CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x8x8x128xf32>, tensor<256x3x3x128xf32>, tensor<256xf32>) -> tensor<1x4x4x256xf32>

// CHECK-LABEL: depthwise_conv_external_same_padding
func.func @depthwise_conv_external_same_padding(%arg0: tensor<1x8x8x64xf32>, %arg1: tensor<1x3x3x64xf32>) -> tensor<1x8x8x64xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %cst_0 = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
  %0 = "tfl.pad"(%arg0, %cst_0) : (tensor<1x8x8x64xf32>, tensor<4x2xi64>) -> tensor<1x10x10x64xf32>
  %1 = "tfl.depthwise_conv_2d"(%0, %arg1, %cst) <{
    depth_multiplier = 1 : i32,
    dilation_h_factor = 1 : i32,
    dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE",
    padding = "VALID",
    stride_h = 1 : i32,
    stride_w = 1 : i32
  }> : (tensor<1x10x10x64xf32>, tensor<1x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
  return %1 : tensor<1x8x8x64xf32>
}

// CHECK: %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %cst) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x8x8x64xf32>, tensor<1x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>

// CHECK-LABEL: reorder_gather_cast
func.func @reorder_gather_cast(%arg0: tensor<2x3x5xi8>, %arg1: tensor<2x7xi32>) -> tensor<2x7x5xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<2x3x5xi8>) -> tensor<2x3x5xf32>
  %1 = "tfl.gather"(%0, %arg1) {axis = 1 : i32, batch_dims = 1 : i32}: (tensor<2x3x5xf32>, tensor<2x7xi32>) -> tensor<2x7x5xf32>
  func.return %1 : tensor<2x7x5xf32>
}

// CHECK: %0 = "tfl.gather"(%arg0, %arg1) <{axis = 1 : i32, batch_dims = 1 : i32}> : (tensor<2x3x5xi8>, tensor<2x7xi32>) -> tensor<2x7x5xi8>
// CHECK: %1 = "tfl.cast"(%0) : (tensor<2x7x5xi8>) -> tensor<2x7x5xf32>

// CHECK-LABEL: @RealDivWithConstDivisor
func.func @RealDivWithConstDivisor(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<5.000000e+00> : tensor<f32>
  %1 = tfl.div(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  func.return %1 : tensor<2x3xf32>
  // CHECK: %cst = arith.constant dense<2.000000e-01> : tensor<f32>
  // CHECK: %0 = tfl.mul(%arg0, %cst) <{fused_activation_function = "NONE"}> : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  // CHECK: return %0 : tensor<2x3xf32>
}

//CHECK-LABEL: @PushTransposeThroughSqueezeNoDims
func.func @PushTransposeThroughSqueezeNoDims(%arg0: tensor<1x1x2x3xf32>) -> (tensor<3x2xf32>) {
  %cst = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x2x3xf32>, tensor<4xi32>) -> tensor<1x3x1x2xf32>
  %1 = "tfl.squeeze"(%0): (tensor<1x3x1x2xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>

  // CHECK: %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %cst_0 = arith.constant dense<[2, 3]> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<1x1x2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  // CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
}

//CHECK-LABEL: @PushTransposeThroughSqueeze1
func.func @PushTransposeThroughSqueeze1(%arg0: tensor<1x1x2x3xf32>) -> (tensor<3x2xf32>) {
  %cst = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x2x3xf32>, tensor<4xi32>) -> tensor<1x3x1x2xf32>
  %1 = "tfl.squeeze"(%0) {squeeze_dims = [0, 2]}: (tensor<1x3x1x2xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>

  // CHECK: %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %cst_0 = arith.constant dense<[2, 3]> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<1x1x2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  // CHECK: %1 = "tfl.transpose"(%0, %cst) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK: return
}

//CHECK-LABEL: @PushTransposeThroughSqueeze2
func.func @PushTransposeThroughSqueeze2(%arg0: tensor<1x1x2x3xf32>) -> (tensor<2x3xf32>) {
  %cst = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x2x3xf32>, tensor<4xi32>) -> tensor<1x2x1x3xf32>
  %1 = "tfl.squeeze"(%0) {squeeze_dims = [0, 2]}: (tensor<1x2x1x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

  // CHECK: %cst = arith.constant dense<[2, 3]> : tensor<2xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %cst) : (tensor<1x1x2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  // CHECK: return
}

//CHECK-LABEL: @EliminateBooleanCastCompare
func.func @EliminateBooleanCastCompare(%arg0: tensor<*xi1>) -> (tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>) {
  %zero = arith.constant dense<0> : tensor<i32>
  %cast = "tfl.cast"(%arg0) : (tensor<*xi1>) -> tensor<*xi32>

  %1 = "tfl.equal"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %2 = "tfl.less_equal"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %3 = "tfl.greater_equal"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %4 = "tfl.not_equal"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %5 = "tfl.greater"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %6 = "tfl.less"(%cast, %zero) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>

  %7 = "tfl.equal"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
  %8 = "tfl.less_equal"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
  %9 = "tfl.greater_equal"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
  %10 = "tfl.not_equal"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
  %11 = "tfl.greater"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
  %12 = "tfl.less"(%zero, %cast) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>

  return %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12 : tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>

  // CHECK: %0 = "tfl.logical_not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %1 = "tfl.logical_not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %2 = "tfl.zeros_like"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %3 = "tfl.logical_not"(%2) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %4 = "tfl.zeros_like"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %5 = "tfl.logical_not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %6 = "tfl.zeros_like"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %7 = "tfl.logical_not"(%6) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %8 = "tfl.logical_not"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: %9 = "tfl.zeros_like"(%arg0) : (tensor<*xi1>) -> tensor<*xi1>
  // CHECK: return %0, %1, %3, %arg0, %arg0, %4, %5, %7, %8, %arg0, %9, %arg0 : tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>, tensor<*xi1>
}

// CHECK-LABEL: @ReorderTransposeReshapeTranspose
func.func @ReorderTransposeReshapeTranspose(%arg0: tensor<282x2048xf32>) -> tensor<2x1x282x1024xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[2, 1024, 1, 282]> : tensor<4xi32>
  %cst_2 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<282x2048xf32>, tensor<2xi32>) -> tensor<2048x282xf32>
  %1 = "tfl.reshape"(%0, %cst_1) : (tensor<2048x282xf32>, tensor<4xi32>) -> tensor<2x1024x1x282xf32>
  %2 = "tfl.transpose"(%1, %cst_2) : (tensor<2x1024x1x282xf32>, tensor<4xi32>) -> tensor<2x1x282x1024xf32>
  return %2: tensor<2x1x282x1024xf32>

  // CHECK:      %cst = arith.constant dense<[1, 3, 0, 2]> : tensor<4xi32>
  // CHECK-NEXT: %cst_0 = arith.constant dense<[282, 2, 1024, 1]> : tensor<4xi32>
  // CHECK-NEXT: %0 = "tfl.reshape"(%arg0, %cst_0) : (tensor<282x2048xf32>, tensor<4xi32>) -> tensor<282x2x1024x1xf32>
  // CHECK-NEXT: %1 = "tfl.transpose"(%0, %cst) : (tensor<282x2x1024x1xf32>, tensor<4xi32>) -> tensor<2x1x282x1024xf32>
  // CHECK-NEXT: return %1 : tensor<2x1x282x1024xf32>
}

// CHECK-LABEL: @FullyConnectedSwapOperandsWhenLHSIsConst
func.func @FullyConnectedSwapOperandsWhenLHSIsConst(%arg0: tensor<4x2xf32>, %arg1: none) -> tensor<2x4xf32> {
  %cst = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %0 = "tfl.fully_connected"(%cst, %arg0, %arg1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x2xf32>, tensor<4x2xf32>, none) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>

  // CHECK:      %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK-NEXT: %cst_0 = arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  // CHECK-NEXT: %0 = "tfl.fully_connected"(%arg0, %cst_0, %arg1) <{asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  // CHECK-NEXT: %1 = "tfl.transpose"(%0, %cst) : (tensor<4x2xf32>, tensor<2xi32>) -> tensor<2x4xf32>
  // CHECK-NEXT: return %1 : tensor<2x4xf32>
}

// CHECK-LABEL: @FullyConnectedSwapOperandsWhenLHSIsConstBias
func.func @FullyConnectedSwapOperandsWhenLHSIsConstBias(%arg0: tensor<4x2xf32>) -> tensor<2x4xf32> {
  %cst = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst_1 = arith.constant dense<2.0> : tensor<2xf32>
  %0 = "tfl.fully_connected"(%cst, %arg0, %cst_1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x2xf32>, tensor<4x2xf32>, tensor<2xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>

  // CHECK:      [[cst:%.*]] = arith.constant
  // CHECK-NEXT: [[cst_1:%.*]] = arith.constant
  // CHECK-NOT:  %0 = "tfl.fully_connected"(%arg0, [[cst]], [[cst_1]])
}

// CHECK-LABEL: @FullyConnectedSwapOperandsWhenLHSIsConstKeepNumDimsTrue
func.func @FullyConnectedSwapOperandsWhenLHSIsConstKeepNumDimsTrue(%arg0: tensor<4x2xf32>, %arg1: none) -> tensor<2x4xf32> {
  %cst = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %0 = "tfl.fully_connected"(%cst, %arg0, %arg1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"} : (tensor<2x2xf32>, tensor<4x2xf32>, none) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>

  // CHECK:      [[cst:%.*]] = arith.constant
  // CHECK-NOT:  %0 = "tfl.fully_connected"(%arg0, [[cst]], %arg1)
}

// CHECK-LABEL: @FullyConnectedSwapOperandsWhenLHSIsConstFusedActivationFunction
func.func @FullyConnectedSwapOperandsWhenLHSIsConstFusedActivationFunction(%arg0: tensor<4x2xf32>, %arg1: none) -> tensor<2x4xf32> {
  %cst = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %0 = "tfl.fully_connected"(%cst, %arg0, %arg1) {asymmetric_quantize_inputs = true, fused_activation_function = "RELU", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x2xf32>, tensor<4x2xf32>, none) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>

  // CHECK:      [[cst:%.*]] = arith.constant
  // CHECK-NOT:  %0 = "tfl.fully_connected"(%arg0, [[cst]], %arg1)
}

