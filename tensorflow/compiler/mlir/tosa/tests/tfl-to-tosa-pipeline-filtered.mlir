// RUN: tf-opt --pass-pipeline='builtin.module(func.func(tosa-legalize-tfl{disable-patterns=TFLConv2D,TFLSoftmax, enable-patterns=TFLFullyConnected,TFLTranspose}))' %s | FileCheck %s
// REQUIRES: tf_tosa

// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAR0:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK: %[[VAR1:.*]] = "tfl.conv_2d"(%arg0, %arg1, %[[VAR0]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
func.func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @test_softmax(
// CHECK-SAME:%[[VAR0:.*]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK: %[[VAR1:.*]] = "tfl.softmax"(%[[VAR0]]) <{beta = 1.000000e+00 : f32}> : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAR1]] : tensor<13x21x3xf32>
func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape {values = dense<[14, 1, 1, 19]> : tensor<4xindex>}
// CHECK-DAG: %[[CONST1:.*]] = tosa.const_shape {values = dense<[28, 1, 1, 19]> : tensor<4xindex>}
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape {values = dense<[14, 28]> : tensor<2xindex>}
// CHECK-DAG: %[[CONST3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<28xf32>}>
// CHECK: %[[VAR1:.*]] = tosa.transpose %arg1 {perms = array<i32: 1, 0>}
// CHECK-DAG: %[[VAR2:.*]] = tosa.reshape %arg0, %[[CONST0]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR1]], %[[CONST1]]
// CHECK-DAG: %[[VAR4:.*]] = tosa.conv2d %[[VAR2]], %[[VAR3]], %[[VAR0]], %[[CONST3]], %[[CONST3]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR5:.*]] = tosa.reshape %[[VAR4]], %[[CONST2]]
func.func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<19x28xf32>, tensor<2xi32>) -> tensor<*xf32>
  %1 = "tfl.fully_connected"(%arg0, %0, %cst_0)  {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}  : (tensor<14x19xf32>, tensor<*xf32>, none) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----
