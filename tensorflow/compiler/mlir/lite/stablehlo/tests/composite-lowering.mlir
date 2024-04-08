// RUN: odml-to-stablehlo-opt -composite-lowering -verify-diagnostics %s | FileCheck %s

func.func @hardswish(%arg0: tensor<2xf32>) -> (tensor<*xf32>) {
  %0 = mhlo.composite "aten.hardswish.default" %arg0 {decomposition = @XlaCallModule_aten.hardswish.default.impl_0} : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}
func.func private @XlaCallModule_aten.hardswish.default.impl_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = mhlo.constant dense<6.000000e+00> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %2 = mhlo.constant dense<3.40282347E+38> : tensor<f32>
  %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %4 = mhlo.constant dense<3.000000e+00> : tensor<f32>
  %5 = "mhlo.broadcast_in_dim"(%4) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %6 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %7 = "mhlo.broadcast_in_dim"(%6) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %8 = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  %9 = "mhlo.broadcast_in_dim"(%8) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %10 = mhlo.add %arg0, %5 : tensor<2xf32>
  %11 = mhlo.clamp %7, %10, %3 : tensor<2xf32>
  %12 = mhlo.clamp %9, %11, %1 : tensor<2xf32>
  %13 = mhlo.multiply %arg0, %12 : tensor<2xf32>
  %14 = mhlo.divide %13, %1 : tensor<2xf32>
  return %14 : tensor<2xf32>
}

// CHECK-LABEL:   func.func @hardswish(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[VAL_1:.*]] = "tfl.hard_swish"(%[[VAL_0]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           %[[VAL_2:.*]] = "tf.Identity"(%[[VAL_1]]) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_3:.*]] = "tf.Identity"(%[[VAL_2]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_3]] : tensor<*xf32>
// CHECK:         }


func.func @avg_pool2d_1(%arg0: tensor<1x3x6x6xf32>) -> (tensor<*xf32>) {
  %0 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = false, count_include_pad = true, divisor_override = "py_None", kernel_size = dense<3> : tensor<2xi64>, padding = dense<0> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_0} : (tensor<1x3x6x6xf32>) -> tensor<1x3x4x4xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_0(%arg0: tensor<1x3x6x6xf32>) -> tensor<1x3x4x4xf32> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<6x6xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = "mhlo.reduce_window"(%arg0, %2) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<1x3x6x6xf32>, tensor<f32>) -> tensor<1x3x4x4xf32>
  %4 = "mhlo.reduce_window"(%1, %2) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {window_dimensions = dense<3> : tensor<2xi64>} : (tensor<6x6xf32>, tensor<f32>) -> tensor<4x4xf32>
  %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<1x3x4x4xf32>
  %6 = mhlo.divide %3, %5 : tensor<1x3x4x4xf32>
  return %6 : tensor<1x3x4x4xf32>
}

// CHECK-LABEL:   func.func @avg_pool2d_1(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<1x3x6x6xf32>) -> tensor<*xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = "tfl.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<1x3x6x6xf32>, tensor<4xi32>) -> tensor<1x6x6x3xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0> : tensor<4x2xi32>
// CHECK:           %[[VAL_4:.*]] = "tfl.pad"(%[[VAL_2]], %[[VAL_3]]) : (tensor<1x6x6x3xf32>, tensor<4x2xi32>) -> tensor<1x6x6x3xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.average_pool_2d"(%[[VAL_4]]) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x3xf32>) -> tensor<1x4x4x3xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:           %[[VAL_7:.*]] = "tfl.transpose"(%[[VAL_5]], %[[VAL_6]]) : (tensor<1x4x4x3xf32>, tensor<4xi32>) -> tensor<1x3x4x4xf32>
// CHECK:           %[[VAL_8:.*]] = "tf.Identity"(%[[VAL_7]]) {device = ""} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_9:.*]] = "tf.Identity"(%[[VAL_8]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_9]] : tensor<*xf32>
// CHECK:         }

func.func @avg_pool2d_2(%arg0: tensor<1x3x6x6xf32>) -> (tensor<*xf32>) {
  %0 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = false, count_include_pad = false, divisor_override = "py_None", kernel_size = dense<3> : tensor<2xi64>, padding = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_1} : (tensor<1x3x6x6xf32>) -> tensor<1x3x6x6xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<1x3x6x6xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_1(%arg0: tensor<1x3x6x6xf32>) -> tensor<1x3x6x6xf32> {
  %0 = mhlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<8x8xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.pad"(%arg0, %1) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x3x6x6xf32>, tensor<f32>) -> tensor<1x3x8x8xf32>
  %3 = "mhlo.reduce_window"(%2, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>} : (tensor<1x3x8x8xf32>, tensor<f32>) -> tensor<1x3x6x6xf32>
  %4 = "mhlo.reduce_window"(%0, %1) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = mhlo.add %arg1, %arg2 : tensor<f32>
    mhlo.return %7 : tensor<f32>
  }) {window_dimensions = dense<3> : tensor<2xi64>} : (tensor<8x8xf32>, tensor<f32>) -> tensor<6x6xf32>
  %5 = "mhlo.broadcast_in_dim"(%4) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<6x6xf32>) -> tensor<1x3x6x6xf32>
  %6 = mhlo.divide %3, %5 : tensor<1x3x6x6xf32>
  return %6 : tensor<1x3x6x6xf32>
}

// CHECK-LABEL:   func.func @avg_pool2d_2(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x3x6x6xf32>) -> tensor<*xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = "tfl.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<1x3x6x6xf32>, tensor<4xi32>) -> tensor<1x6x6x3xf32>
// CHECK:           %[[VAL_3:.*]] = "tfl.average_pool_2d"(%[[VAL_2]]) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x3xf32>) -> tensor<1x6x6x3xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:           %[[VAL_5:.*]] = "tfl.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x6x6x3xf32>, tensor<4xi32>) -> tensor<1x3x6x6xf32>
// CHECK:           %[[VAL_6:.*]] = "tf.Identity"(%[[VAL_5]]) {device = ""} : (tensor<1x3x6x6xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_7:.*]] = "tf.Identity"(%[[VAL_6]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_7]] : tensor<*xf32>
// CHECK:         }
