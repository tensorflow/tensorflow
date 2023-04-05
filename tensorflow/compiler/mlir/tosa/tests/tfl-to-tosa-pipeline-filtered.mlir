// RUN: tf-opt --pass-pipeline='builtin.module(func.func(tosa-legalize-tfl{disable-patterns=TFLConv2D,TFLSoftmax, enable-patterns=TFLFullyConnected,TFLTranspose}))' %s | FileCheck %s

// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAR0:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[VAR1:.*]] = "tfl.conv_2d"(%arg0, %arg1, %[[VAR0]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}
func.func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @test_softmax(
// CHECK-SAME:%[[VAR0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK: %[[VAR1:.*]] = "tfl.softmax"(%[[VAR0]]) {beta = 1.000000e+00 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAR1]] : tensor<13x21x3xf32>
func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: %[[VAR0:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<28xf32>}
// CHECK: %[[VAR2:.*]] = "tosa.transpose"(%arg1, %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.fully_connected"(%arg0, %[[VAR2]], %[[VAR1]])
func.func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<19x28xf32>, tensor<2xi32>) -> tensor<*xf32>
  %1 = "tfl.fully_connected"(%arg0, %0, %cst_0)  {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}  : (tensor<14x19xf32>, tensor<*xf32>, none) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----
