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
// CHECK:           %[[VAL_5:.*]] = "tfl.average_pool_2d"(%[[VAL_4]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x3xf32>) -> tensor<1x4x4x3xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:           %[[VAL_7:.*]] = "tfl.transpose"(%[[VAL_5]], %[[VAL_6]]) : (tensor<1x4x4x3xf32>, tensor<4xi32>) -> tensor<1x3x4x4xf32>
// CHECK:           %[[VAL_8:.*]] = "tf.Identity"(%[[VAL_7]]) {device = ""} : (tensor<1x3x4x4xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_9:.*]] = "tf.Identity"(%[[VAL_8]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_9]] : tensor<*xf32>

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
// CHECK:           %[[VAL_3:.*]] = "tfl.average_pool_2d"(%[[VAL_2]]) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x3xf32>) -> tensor<1x6x6x3xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:           %[[VAL_5:.*]] = "tfl.transpose"(%[[VAL_3]], %[[VAL_4]]) : (tensor<1x6x6x3xf32>, tensor<4xi32>) -> tensor<1x3x6x6xf32>
// CHECK:           %[[VAL_6:.*]] = "tf.Identity"(%[[VAL_5]]) {device = ""} : (tensor<1x3x6x6xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_7:.*]] = "tf.Identity"(%[[VAL_6]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_7]] : tensor<*xf32>

func.func @avg_pool2d_3(%arg0: tensor<1x1x1x8xf32>) -> (tensor<1x1x1x4xf32>) {
  %2 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = false, count_include_pad = true, divisor_override = "py_None", kernel_size = dense<[1, 3]> : tensor<2xi64>, padding = dense<[0, 1]> : tensor<2xi64>, stride = dense<[1, 2]> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_2} : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x4xf32>
  return %2 : tensor<1x1x1x4xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_2(%arg0: tensor<1x1x1x8xf32>) -> tensor<1x1x1x4xf32>

// CHECK-LABEL: avg_pool2d_3
// CHECK: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x1x8xf32>, tensor<4xi32>) -> tensor<1x1x8x1xf32>
// CHECK{LITERAL}: %cst_0 = arith.constant dense<[[0, 0], [0, 0], [1, 1], [0, 0]]> : tensor<4x2xi32>
// CHECK: %1 = "tfl.pad"(%0, %cst_0) : (tensor<1x1x8x1xf32>, tensor<4x2xi32>) -> tensor<1x1x10x1xf32>
// CHECK: %2 = "tfl.average_pool_2d"(%1) <{filter_height = 1 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 2 : i32}> : (tensor<1x1x10x1xf32>) -> tensor<1x1x4x1xf32>
// CHECK: %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: %3 = "tfl.transpose"(%2, %cst_1) : (tensor<1x1x4x1xf32>, tensor<4xi32>) -> tensor<1x1x1x4xf32>
// CHECK: return %3 : tensor<1x1x1x4xf32>

func.func @avg_pool2d_4(%arg0: tensor<1x1x1x9xf32>) -> (tensor<1x1x1x4xf32>) {
  %2 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = false, count_include_pad = false, divisor_override = "py_None", kernel_size = dense<[1, 3]> : tensor<2xi64>, padding = dense<[0, 0]> : tensor<2xi64>, stride = dense<[1, 2]> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_3} : (tensor<1x1x1x9xf32>) -> tensor<1x1x1x4xf32>
  return %2 : tensor<1x1x1x4xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_3(%arg0: tensor<1x1x1x9xf32>) -> tensor<1x1x1x4xf32>

// CHECK-LABEL: avg_pool2d_4
// CHECK: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x1x9xf32>, tensor<4xi32>) -> tensor<1x1x9x1xf32>
// CHECK: %1 = "tfl.average_pool_2d"(%0) <{filter_height = 1 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 2 : i32}> : (tensor<1x1x9x1xf32>) -> tensor<1x1x4x1xf32>
// CHECK: %cst_0 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: %2 = "tfl.transpose"(%1, %cst_0) : (tensor<1x1x4x1xf32>, tensor<4xi32>) -> tensor<1x1x1x4xf32>
// CHECK: return %2 : tensor<1x1x1x4xf32>


func.func @avg_pool2d_5(%arg0: tensor<1x1x3x3xf32>) -> (tensor<1x1x2x2xf32>) {
  %0 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = true, count_include_pad = true, divisor_override = "py_None", kernel_size = dense<[2, 2]> : tensor<2xi64>, padding = dense<[0, 0]> : tensor<2xi64>, stride = dense<[2, 2]> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_4} : (tensor<1x1x3x3xf32>) -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_4(%arg0: tensor<1x1x3x3xf32>) -> tensor<1x1x2x2xf32>

// CHECK-LABEL: avg_pool2d_5
// CHECK: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x3x3xf32>, tensor<4xi32>) -> tensor<1x3x3x1xf32>
// CHECK{LITERAL}: %cst_0 = arith.constant dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>
// CHECK: %1 = "tfl.pad"(%0, %cst_0) : (tensor<1x3x3x1xf32>, tensor<4x2xi32>) -> tensor<1x4x4x1xf32>
// CHECK: %2 = "tfl.average_pool_2d"(%1) <{filter_height = 2 : i32, filter_width = 2 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32>
// CHECK{LITERAL}: %cst_1 = arith.constant dense<[[[[1.000000e+00], [2.000000e+00]], [[2.000000e+00], [4.000000e+00]]]]> : tensor<1x2x2x1xf32>
// CHECK: %3 = tfl.mul %2, %cst_1 {fused_activation_function = "NONE"} : tensor<1x2x2x1xf32>
// CHECK: %cst_2 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: %4 = "tfl.transpose"(%3, %cst_2) : (tensor<1x2x2x1xf32>, tensor<4xi32>) -> tensor<1x1x2x2xf32>
// CHECK: return %4 : tensor<1x1x2x2xf32>

func.func @avg_pool2d_6(%arg0: tensor<1x1x1x7xf32>) -> (tensor<1x1x1x2xf32>) {
  %0 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = true, count_include_pad = true, divisor_override = "py_None", kernel_size = dense<[1, 5]> : tensor<2xi64>, padding = dense<[0, 0]> : tensor<2xi64>, stride = dense<[1, 3]> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_5} : (tensor<1x1x1x7xf32>) -> tensor<1x1x1x2xf32>
  return %0 : tensor<1x1x1x2xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_5(%arg0: tensor<1x1x1x7xf32>) -> tensor<1x1x1x2xf32>

// CHECK-LABEL: avg_pool2d_6
// CHECK: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x1x7xf32>, tensor<4xi32>) -> tensor<1x1x7x1xf32>
// CHECK{LITERAL}: %cst_0 = arith.constant dense<[[0, 0], [0, 0], [0, 1], [0, 0]]> : tensor<4x2xi32>
// CHECK: %1 = "tfl.pad"(%0, %cst_0) : (tensor<1x1x7x1xf32>, tensor<4x2xi32>) -> tensor<1x1x8x1xf32>
// CHECK: %2 = "tfl.average_pool_2d"(%1) <{filter_height = 1 : i32, filter_width = 5 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 3 : i32}> : (tensor<1x1x8x1xf32>) -> tensor<1x1x2x1xf32>
// CHECK{LITERAL}: %cst_1 = arith.constant dense<[[[[1.000000e+00], [1.250000e+00]]]]> : tensor<1x1x2x1xf32>
// CHECK: %3 = tfl.mul %2, %cst_1 {fused_activation_function = "NONE"} : tensor<1x1x2x1xf32>
// CHECK: %cst_2 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: %4 = "tfl.transpose"(%3, %cst_2) : (tensor<1x1x2x1xf32>, tensor<4xi32>) -> tensor<1x1x1x2xf32>

func.func @avg_pool2d_7(%arg0: tensor<1x1x1x8xf32>) -> (tensor<1x1x1x5xf32>) {
  %0 = mhlo.composite "aten.avg_pool2d.default" %arg0 {composite_attributes = {ceil_mode = true, count_include_pad = true, divisor_override = "py_None", kernel_size = dense<[1, 3]> : tensor<2xi64>, padding = dense<[0, 1]> : tensor<2xi64>, stride = dense<[1, 2]> : tensor<2xi64>}, decomposition = @XlaCallModule_aten.avg_pool2d.default.impl_6} : (tensor<1x1x1x8xf32>) -> tensor<1x1x1x5xf32>
  return %0 : tensor<1x1x1x5xf32>
}
func.func private @XlaCallModule_aten.avg_pool2d.default.impl_6(%arg0: tensor<1x1x1x8xf32>) -> tensor<1x1x1x5xf32>

// CHECK-LABEL: avg_pool2d_7
// CHECK: %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK{LITERAL}: %0 = "tfl.transpose"(%arg0, %cst) : (tensor<1x1x1x8xf32>, tensor<4xi32>) -> tensor<1x1x8x1xf32>
// CHECK{LITERAL}: %cst_0 = arith.constant dense<[[0, 0], [0, 0], [1, 2], [0, 0]]> : tensor<4x2xi32>
// CHECK: %1 = "tfl.pad"(%0, %cst_0) : (tensor<1x1x8x1xf32>, tensor<4x2xi32>) -> tensor<1x1x11x1xf32>
// CHECK: %2 = "tfl.average_pool_2d"(%1) <{filter_height = 1 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 2 : i32}> : (tensor<1x1x11x1xf32>) -> tensor<1x1x5x1xf32>
// CHECK{LITERAL}: %cst_1 = arith.constant dense<[[[[1.000000e+00], [1.000000e+00], [1.000000e+00], [1.000000e+00], [1.500000e+00]]]]> : tensor<1x1x5x1xf32>
// CHECK: %3 = tfl.mul %2, %cst_1 {fused_activation_function = "NONE"} : tensor<1x1x5x1xf32>
// CHECK: %cst_2 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK: %4 = "tfl.transpose"(%3, %cst_2) : (tensor<1x1x5x1xf32>, tensor<4xi32>) -> tensor<1x1x1x5xf32>

func.func @upsample_bilinear2d(%arg0: tensor<1x64x16x16xf32>) -> (tensor<1x64x32x32xf32>) {
  %0 = mhlo.composite "odml.upsample_bilinear2d" %arg0 {composite_attributes = {is_nchw_op = true, align_corners = false, output = dense<32> : tensor<2xi64>}, decomposition = @XlaCallModule_odml.upsample_bilinear2d.impl_21_0} : (tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
func.func private @XlaCallModule_odml.upsample_bilinear2d.impl_21_0(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %0 = mhlo.constant dense<[[0.000000e+00], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01], [7.500000e-01], [2.500000e-01]]> : tensor<32x1xf32>
  %1 = mhlo.constant dense<[0.000000e+00, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01, 7.500000e-01, 2.500000e-01]> : tensor<32xf32>
  %2 = mhlo.constant dense<[1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15]> : tensor<32xi64>
  %3 = mhlo.constant dense<16> : tensor<i64>
  %4 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i64>) -> tensor<32x32xi64>
  %5 = mhlo.constant dense<0> : tensor<i64>
  %6 = "mhlo.broadcast_in_dim"(%5) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i64>) -> tensor<32x32xi64>
  %7 = mhlo.constant dense<[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15]> : tensor<32xi64>
  %8 = "mhlo.broadcast_in_dim"(%7) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<32xi64>) -> tensor<32x32xi64>
  %9 = mhlo.compare  LT, %8, %6 : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
  %10 = mhlo.add %8, %4 : tensor<32x32xi64>
  %11 = mhlo.select %9, %10, %8 : tensor<32x32xi1>, tensor<32x32xi64>
  %12 = mhlo.reshape %11 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
  %13 = "mhlo.broadcast_in_dim"(%7) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<32xi64>) -> tensor<32x32xi64>
  %14 = mhlo.compare  LT, %13, %6 : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
  %15 = mhlo.add %13, %4 : tensor<32x32xi64>
  %16 = mhlo.select %14, %15, %13 : tensor<32x32xi1>, tensor<32x32xi64>
  %17 = mhlo.reshape %16 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
  %18 = "mhlo.concatenate"(%12, %17) <{dimension = 2 : i64}> : (tensor<32x32x1xi64>, tensor<32x32x1xi64>) -> tensor<32x32x2xi64>
  %19 = "mhlo.gather"(%arg0, %18) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>, slice_sizes = dense<[1, 64, 1, 1]> : tensor<4xi64>}> : (tensor<1x64x16x16xf32>, tensor<32x32x2xi64>) -> tensor<1x64x32x32xf32>
  %20 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<32xi64>) -> tensor<32x32xi64>
  %21 = mhlo.compare  LT, %20, %6 : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
  %22 = mhlo.add %20, %4 : tensor<32x32xi64>
  %23 = mhlo.select %21, %22, %20 : tensor<32x32xi1>, tensor<32x32xi64>
  %24 = mhlo.reshape %23 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
  %25 = "mhlo.concatenate"(%12, %24) <{dimension = 2 : i64}> : (tensor<32x32x1xi64>, tensor<32x32x1xi64>) -> tensor<32x32x2xi64>
  %26 = "mhlo.gather"(%arg0, %25) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>, slice_sizes = dense<[1, 64, 1, 1]> : tensor<4xi64>}> : (tensor<1x64x16x16xf32>, tensor<32x32x2xi64>) -> tensor<1x64x32x32xf32>
  %27 = mhlo.subtract %26, %19 : tensor<1x64x32x32xf32>
  %28 = "mhlo.broadcast_in_dim"(%1) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<32xf32>) -> tensor<1x64x32x32xf32>
  %29 = mhlo.multiply %27, %28 : tensor<1x64x32x32xf32>
  %30 = mhlo.add %19, %29 : tensor<1x64x32x32xf32>
  %31 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<32xi64>) -> tensor<32x32xi64>
  %32 = mhlo.compare  LT, %31, %6 : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
  %33 = mhlo.add %31, %4 : tensor<32x32xi64>
  %34 = mhlo.select %32, %33, %31 : tensor<32x32xi1>, tensor<32x32xi64>
  %35 = mhlo.reshape %34 : (tensor<32x32xi64>) -> tensor<32x32x1xi64>
  %36 = "mhlo.concatenate"(%35, %17) <{dimension = 2 : i64}> : (tensor<32x32x1xi64>, tensor<32x32x1xi64>) -> tensor<32x32x2xi64>
  %37 = "mhlo.gather"(%arg0, %36) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>, slice_sizes = dense<[1, 64, 1, 1]> : tensor<4xi64>}> : (tensor<1x64x16x16xf32>, tensor<32x32x2xi64>) -> tensor<1x64x32x32xf32>
  %38 = "mhlo.concatenate"(%35, %24) <{dimension = 2 : i64}> : (tensor<32x32x1xi64>, tensor<32x32x1xi64>) -> tensor<32x32x2xi64>
  %39 = "mhlo.gather"(%arg0, %38) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1], collapsed_slice_dims = [2, 3], start_index_map = [2, 3], index_vector_dim = 2>, slice_sizes = dense<[1, 64, 1, 1]> : tensor<4xi64>}> : (tensor<1x64x16x16xf32>, tensor<32x32x2xi64>) -> tensor<1x64x32x32xf32>
  %40 = mhlo.subtract %39, %37 : tensor<1x64x32x32xf32>
  %41 = mhlo.multiply %40, %28 : tensor<1x64x32x32xf32>
  %42 = mhlo.add %37, %41 : tensor<1x64x32x32xf32>
  %43 = mhlo.subtract %42, %30 : tensor<1x64x32x32xf32>
  %44 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>}> : (tensor<32x1xf32>) -> tensor<1x64x32x1xf32>
  %45 = mhlo.reshape %44 : (tensor<1x64x32x1xf32>) -> tensor<1x64x32xf32>
  %46 = "mhlo.broadcast_in_dim"(%45) <{broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}> : (tensor<1x64x32xf32>) -> tensor<1x64x32x32xf32>
  %47 = mhlo.multiply %43, %46 : tensor<1x64x32x32xf32>
  %48 = mhlo.add %30, %47 : tensor<1x64x32x32xf32>
  return %48 : tensor<1x64x32x32xf32>
}

// CHECK-LABEL:   func.func @upsample_bilinear2d(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_2:.*]] = "tfl.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<1x64x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x64xf32>
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant dense<32> : tensor<2xi32>
// CHECK-DAG:           %[[VAL_4:.*]] = "tfl.resize_bilinear"(%[[VAL_2]], %[[VAL_3]]) <{align_corners = false, half_pixel_centers = true}> : (tensor<1x16x16x64xf32>, tensor<2xi32>) -> tensor<1x32x32x64xf32>
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_6:.*]] = "tfl.transpose"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x32x32x64xf32>, tensor<4xi32>) -> tensor<1x64x32x32xf32>
// CHECK:           return %[[VAL_6]] : tensor<1x64x32x32xf32>
// CHECK:         }

func.func private @gelu_decomp(%arg0: tensor<2xf32>) -> tensor<2xf32>
func.func @gelu(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = mhlo.composite "odml.internal.gelu" %arg0 {composite_attributes = {approx = false}, decomposition = @gelu_decomp} : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: gelu
// CHECK: %0 = "tfl.gelu"(%arg0) <{approximate = false}> : (tensor<2xf32>) -> tensor<2xf32>

// CHECK-LABEL  func.func @jax_image_resize_nearest
func.func @jax_image_resize_nearest(%arg0: tensor<1x2x2x10xf32>) -> (tensor<1x4x4x10xf32>) {
  %1 = mhlo.composite "tfl.resize_nearest_neighbor" %arg0 {composite_attributes = {is_nchw_op = false, align_corners = false, size = dense<4> : tensor<2xi64>}, decomposition = @XlaCallModule_tfl.resize_nearest_neighbor.impl_0} : (tensor<1x2x2x10xf32>) -> tensor<1x4x4x10xf32>
  return %1 : tensor<1x4x4x10xf32>
}
func.func private @XlaCallModule_tfl.resize_nearest_neighbor.impl_0(%arg0: tensor<1x2x2x10xf32>) -> tensor<1x4x4x10xf32> {
  %0 = call @XlaCallModule__resize_0(%arg0) : (tensor<1x2x2x10xf32>) -> tensor<1x4x4x10xf32>
  return %0 : tensor<1x4x4x10xf32>
}
func.func private @XlaCallModule__resize_0(%arg0: tensor<1x2x2x10xf32>) -> (tensor<1x4x4x10xf32>) {
  %0 = mhlo.constant dense<2> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  %2 = mhlo.constant dense<4.000000e+00> : tensor<f32>
  %3 = mhlo.constant dense<2.000000e+00> : tensor<f32>
  %4 = mhlo.constant dense<5.000000e-01> : tensor<f32>
  %5 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xf32>
  %6 = "mhlo.broadcast_in_dim"(%4) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %7 = mhlo.add %5, %6 : tensor<4xf32>
  %8 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %9 = mhlo.multiply %7, %8 : tensor<4xf32>
  %10 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %11 = mhlo.divide %9, %10 : tensor<4xf32>
  %12 = mhlo.floor %11 : tensor<4xf32>
  %13 = mhlo.convert %12 : (tensor<4xf32>) -> tensor<4xi32>
  %14 = "mhlo.broadcast_in_dim"(%1) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<4xi32>
  %15 = mhlo.compare  LT, %13, %14,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %16 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<4xi32>
  %17 = mhlo.add %13, %16 : tensor<4xi32>
  %18 = mhlo.select %15, %17, %13 : tensor<4xi1>, tensor<4xi32>
  %19 = "mhlo.broadcast_in_dim"(%18) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<4x1xi32>
  %20 = "mhlo.gather"(%arg0, %19) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, slice_sizes = dense<[1, 1, 2, 10]> : tensor<4xi64>}> : (tensor<1x2x2x10xf32>, tensor<4x1xi32>) -> tensor<1x4x2x10xf32>
  %21 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xf32>
  %22 = "mhlo.broadcast_in_dim"(%4) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %23 = mhlo.add %21, %22 : tensor<4xf32>
  %24 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %25 = mhlo.multiply %23, %24 : tensor<4xf32>
  %26 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4xf32>
  %27 = mhlo.divide %25, %26 : tensor<4xf32>
  %28 = mhlo.floor %27 : tensor<4xf32>
  %29 = mhlo.convert %28 : (tensor<4xf32>) -> tensor<4xi32>
  %30 = "mhlo.broadcast_in_dim"(%1) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<4xi32>
  %31 = mhlo.compare  LT, %29, %30,  SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %32 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<4xi32>
  %33 = mhlo.add %29, %32 : tensor<4xi32>
  %34 = mhlo.select %31, %33, %29 : tensor<4xi1>, tensor<4xi32>
  %35 = "mhlo.broadcast_in_dim"(%34) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<4x1xi32>
  %36 = "mhlo.gather"(%20, %35) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1, 3], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, slice_sizes = dense<[1, 4, 1, 10]> : tensor<4xi64>}> : (tensor<1x4x2x10xf32>, tensor<4x1xi32>) -> tensor<1x4x4x10xf32>
  return %36 : tensor<1x4x4x10xf32>
}

// CHECK:  %cst = arith.constant dense<4> : tensor<2xi32>
// CHECK:  %0 = "tfl.resize_nearest_neighbor"(%arg0, %cst) <{align_corners = false, half_pixel_centers = true}> : (tensor<1x2x2x10xf32>, tensor<2xi32>) -> tensor<1x4x4x10xf32>
// CHECK:  return %0 : tensor<1x4x4x10xf32>

// CHECK-LABEL  func.func @jax_image_resize_nearest_nchw
func.func @jax_image_resize_nearest_nchw(%arg0: tensor<4x8x32x32xf32>) -> (tensor<4x8x64x64xf32>) {
  %0 = call @XlaCallModule_tfl.resize_nearest_neighbor.impl_1(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x64x64xf32>
  %1 = mhlo.composite "tfl.resize_nearest_neighbor" %arg0 {composite_attributes = {is_nchw_op = true, align_corners = false, size = dense<64> : tensor<2xi64>}, decomposition = @XlaCallModule_tfl.resize_nearest_neighbor.impl_1} : (tensor<4x8x32x32xf32>) -> tensor<4x8x64x64xf32>
  return %1 : tensor<4x8x64x64xf32>
}
func.func private @XlaCallModule_tfl.resize_nearest_neighbor.impl_1(%arg0: tensor<4x8x32x32xf32>) -> tensor<4x8x64x64xf32> {
  %0 = call @XlaCallModule__resize_1(%arg0) : (tensor<4x8x32x32xf32>) -> tensor<4x8x64x64xf32>
  return %0 : tensor<4x8x64x64xf32>
}
func.func private @XlaCallModule__resize_1(%arg0: tensor<4x8x32x32xf32>) -> (tensor<4x8x64x64xf32>) {
  %0 = mhlo.constant dense<32> : tensor<64xi32>
  %1 = mhlo.constant dense<0> : tensor<64xi32>
  %2 = mhlo.constant dense<6.400000e+01> : tensor<64xf32>
  %3 = mhlo.constant dense<3.200000e+01> : tensor<64xf32>
  %4 = mhlo.constant dense<5.000000e-01> : tensor<64xf32>
  %5 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<64xf32>
  %6 = mhlo.add %5, %4 : tensor<64xf32>
  %7 = mhlo.multiply %6, %3 : tensor<64xf32>
  %8 = mhlo.divide %7, %2 : tensor<64xf32>
  %9 = mhlo.floor %8 : tensor<64xf32>
  %10 = mhlo.convert %9 : (tensor<64xf32>) -> tensor<64xi32>
  %11 = mhlo.compare  LT, %10, %1,  SIGNED : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi1>
  %12 = mhlo.add %10, %0 : tensor<64xi32>
  %13 = mhlo.select %11, %12, %10 : tensor<64xi1>, tensor<64xi32>
  %14 = mhlo.reshape %13 : (tensor<64xi32>) -> tensor<64x1xi32>
  %15 = "mhlo.gather"(%arg0, %14) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1, 3], collapsed_slice_dims = [2], start_index_map = [2], index_vector_dim = 1>, slice_sizes = dense<[4, 8, 1, 32]> : tensor<4xi64>}> : (tensor<4x8x32x32xf32>, tensor<64x1xi32>) -> tensor<4x8x64x32xf32>
  %16 = "mhlo.gather"(%15, %14) <{dimension_numbers = #mhlo.gather<offset_dims = [0, 1, 2], collapsed_slice_dims = [3], start_index_map = [3], index_vector_dim = 1>, slice_sizes = dense<[4, 8, 64, 1]> : tensor<4xi64>}> : (tensor<4x8x64x32xf32>, tensor<64x1xi32>) -> tensor<4x8x64x64xf32>
  return %16 : tensor<4x8x64x64xf32>
}
// CHECK:  %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK:  %1 = "tfl.transpose"(%arg0, %cst) : (tensor<4x8x32x32xf32>, tensor<4xi32>) -> tensor<4x32x32x8xf32>
// CHECK:  %cst_0 = arith.constant dense<64> : tensor<2xi32>
// CHECK:  %2 = "tfl.resize_nearest_neighbor"(%1, %cst_0) <{align_corners = false, half_pixel_centers = true}> : (tensor<4x32x32x8xf32>, tensor<2xi32>) -> tensor<4x64x64x8xf32>
// CHECK:  %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:  %3 = "tfl.transpose"(%2, %cst_1) : (tensor<4x64x64x8xf32>, tensor<4xi32>) -> tensor<4x8x64x64xf32>
// CHECK:  return %3 : tensor<4x8x64x64xf32>

