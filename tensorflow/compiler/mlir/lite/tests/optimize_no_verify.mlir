// Run optimize pass only and check the results.
// Tests in this file for optimization patterns that doesn't match
// TFLite runtime restrictions.
// RUN: tf-opt %s -tfl-optimize | FileCheck %s

// CHECK-LABEL: fuseScalarAddIntoConv2dHalf
func @fuseScalarAddIntoConv2dHalf(%arg0: tensor<256x32x32x3xf16>, %arg1: tensor<16x3x3x3xf16>) -> tensor<256x8x7x16xf16> {
  %cst = arith.constant dense<1.5> : tensor<f16>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xf16>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xf16>, tensor<16x3x3x3xf16>, tensor<16xf16>) -> tensor<256x8x7x16xf16>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x16xf16>, tensor<f16>) -> tensor<256x8x7x16xf16>
  return %1 : tensor<256x8x7x16xf16>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xf16>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: @fuseBroadcastMulIntoFullyConnected
func @fuseBroadcastMulIntoFullyConnected(%arg0: tensor<1x10368xbf16>) -> tensor<32x1x256xbf16> {
  %cst_0 = arith.constant dense<2.0> : tensor<256x10368xbf16>
  %cst_1 = constant unit
  %cst_2 = arith.constant dense<3.0> : tensor<32x1x256xbf16>
  %0 = "tfl.fully_connected"(%arg0, %cst_0, %cst_1) {
    fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"
  } : (tensor<1x10368xbf16>, tensor<256x10368xbf16>, none) -> tensor<1x256xbf16>
  %1 = "tfl.mul"(%0, %cst_2) {fused_activation_function = "NONE"} : (tensor<1x256xbf16>, tensor<32x1x256xbf16>) -> tensor<32x1x256xbf16>
  return %1 : tensor<32x1x256xbf16>

// CHECK:  %[[V0:.*]] = "tfl.fully_connected"(%arg0, {{.*}}) {{{.*}}} : (tensor<1x10368xbf16>, tensor<256x10368xbf16>, none) -> tensor<1x256xbf16>
// CHECK:  %[[V1:.*]] = tfl.mul(%[[V0]], {{.*}}) {{{.*}}} : (tensor<1x256xbf16>, tensor<32x1x256xbf16>) -> tensor<32x1x256xbf16>
// CHECK:  return %[[V1]] : tensor<32x1x256xbf16>
}

// CHECK-LABEL: Relu_bf16
func @Relu_bf16(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %cst = arith.constant dense<0.0> : tensor<2x3xbf16>
  %0 = "tfl.maximum"(%arg0, %cst) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %0 : tensor<2x3xbf16>

  // CHECK: %[[RESULT:.*]] = "tfl.relu"(%arg0)
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: fuseScalarAddIntoConv2dBf16
func @fuseScalarAddIntoConv2dBf16(%arg0: tensor<256x32x32x3xbf16>, %arg1: tensor<16x3x3x3xbf16>) -> tensor<256x8x7x16xbf16> {
  %cst = arith.constant dense<1.5> : tensor<bf16>
  %cst_0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<16xbf16>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst_0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<256x32x32x3xbf16>, tensor<16x3x3x3xbf16>, tensor<16xbf16>) -> tensor<256x8x7x16xbf16>
  %1 = "tfl.add"(%0, %cst) {fused_activation_function = "NONE"} : (tensor<256x8x7x16xbf16>, tensor<bf16>) -> tensor<256x8x7x16xbf16>
  return %1 : tensor<256x8x7x16xbf16>

  // CHECK-DAG: %cst = arith.constant dense<[2.500000e+00, 3.500000e+00, 4.500000e+00, 5.500000e+00, 6.500000e+00, 7.500000e+00, 8.500000e+00, 9.500000e+00, 1.050000e+01, 1.150000e+01, 1.250000e+01, 1.350000e+01, 1.450000e+01, 1.550000e+01, 1.650000e+01, 1.750000e+01]> : tensor<16xbf16>
  // CHECK: %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)
}

// CHECK-LABEL: RemoveReshapeAfterFullyConnected
func @RemoveReshapeAfterFullyConnected(%arg0: tensor<4x1024x1024xbf16>) -> tensor<4x1024x4096xbf16> {
  %cst_0 = arith.constant dense<1.0> : tensor<4096x1024xbf16>
  %cst_1 = constant unit
  %cst_2 = arith.constant dense<[4, 1024, 4096]> : tensor<3xi32>
  %0 = "tfl.fully_connected"(%arg0, %cst_0, %cst_1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x1024x1024xbf16>, tensor<4096x1024xbf16>, none) -> tensor<4096x4096xbf16>
  %1 = "tfl.reshape"(%0, %cst_2) : (tensor<4096x4096xbf16>, tensor<3xi32>) -> tensor<4x1024x4096xbf16>
  return %1 : tensor<4x1024x4096xbf16>
  // CHECK: %[[V0:.*]] = "tfl.fully_connected"(%arg0, {{.*}}) {{.*}}keep_num_dims = true{{.*}} -> tensor<4x1024x4096xbf16>
  // CHECK: return %[[V0]]
}

// CHECK-LABEL: RemoveReshapeAfterFullyConnectedAdd
func @RemoveReshapeAfterFullyConnectedAdd(%arg0: tensor<4x1024x1024xbf16>) -> tensor<4x1024x4096xbf16> {
  %cst_0 = arith.constant dense<1.0> : tensor<4096x1024xbf16>
  %cst_1 = constant unit
  %cst_2 = arith.constant dense<[4, 1024, 4096]> : tensor<3xi32>
  %0 = "tfl.fully_connected"(%arg0, %cst_0, %cst_1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x1024x1024xbf16>, tensor<4096x1024xbf16>, none) -> tensor<4096x4096xbf16>
  %1 = "tfl.reshape"(%0, %cst_2) : (tensor<4096x4096xbf16>, tensor<3xi32>) -> tensor<4x1024x4096xbf16>
  %2 = "tfl.mul"(%1, %1) {fused_activation_function = "NONE"} : (tensor<4x1024x4096xbf16>, tensor<4x1024x4096xbf16>) -> tensor<4x1024x4096xbf16>
  return %2 : tensor<4x1024x4096xbf16>
  // CHECK: %[[V0:.*]] = "tfl.fully_connected"(%arg0, {{.*}}) {{.*}}keep_num_dims = true{{.*}} -> tensor<4x1024x4096xbf16>
  // CHECK: %[[V1:.*]] = tfl.mul %[[V0]], %[[V0]] {{.*}} : tensor<4x1024x4096xbf16
  // CHECK: return %[[V1]]
}
