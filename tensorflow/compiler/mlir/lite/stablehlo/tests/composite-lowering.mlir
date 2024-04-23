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

func.func @upsample_bilinear2d(%arg0: tensor<1x64x16x16xf32>) -> (tensor<1x64x32x32xf32>) {
  %0 = mhlo.composite "odml.upsample_bilinear2d" %arg0 {composite_attributes = {align_corners = false, output = dense<32> : tensor<2xi64>}, decomposition = @XlaCallModule_odml.upsample_bilinear2d.impl_21_0} : (tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32>
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
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
// CHECK:           %[[VAL_2:.*]] = "tfl.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<1x64x16x16xf32>, tensor<4xi32>) -> tensor<1x16x16x64xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<32> : tensor<2xi32>
// CHECK:           %[[VAL_4:.*]] = "tfl.resize_bilinear"(%[[VAL_2]], %[[VAL_3]]) {align_corners = false, half_pixel_centers = true} : (tensor<1x16x16x64xf32>, tensor<2xi32>) -> tensor<1x32x32x64xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
// CHECK:           %[[VAL_6:.*]] = "tfl.transpose"(%[[VAL_4]], %[[VAL_5]]) : (tensor<1x32x32x64xf32>, tensor<4xi32>) -> tensor<1x64x32x32xf32>
// CHECK:           return %[[VAL_6]] : tensor<1x64x32x32xf32>
// CHECK:         }
