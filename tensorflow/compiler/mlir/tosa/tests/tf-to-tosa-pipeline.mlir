// RUN: tf-opt --tf-to-tosa-pipeline  --verify-each %s | FileCheck %s
// RUN: tf-opt --tf-tfl-to-tosa-pipeline  --verify-each %s | FileCheck %s

// Operations for testing tf-to-tosa-pipeline
// TODO: These tests are fairly minimal. Expand the checks to be more robust.

// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%arg1, %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.conv2d"(%arg0, %[[VAR2]], %[[VAR0]]) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
func.func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x16xf32>) -> tensor<1x32x32x16xf32> {
  %3 = "tf.Conv2D"(%arg0, %arg1)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : (tensor<1x32x32x8xf32>, tensor<2x2x8x16xf32>) -> tensor<1x32x32x16xf32>
  func.return %3 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.depthwise_conv2d"(%arg0, %arg1, %0) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
func.func @test_depthwise_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<2x2x8x2xf32>) -> tensor<1x32x32x16xf32> {
  %5 = "tf.DepthwiseConv2dNative"(%arg0, %arg1)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>, tensor<2x2x8x2xf32>) -> tensor<1x32x32x16xf32>
  %6 = "tf.Identity"(%5)   : (tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
  func.return %6 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: @test_transpose_conv2d
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x32x32x8xf32>, %[[ARG1:.*]]: tensor<1x1x16x8xf32>
// CHECK:         %[[CONST:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK:         %[[RESHAPE:.*]] = "tosa.reshape"(%[[ARG1]]) {new_shape = array<i64: 16, 1, 1, 8>}
// CHECK:         %[[TRANSPOSE:.*]] = "tosa.transpose_conv2d"(%[[ARG0]], %[[RESHAPE]], %[[CONST]]) {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>}
// CHECK:         return %[[TRANSPOSE]]
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x16x8xf32>) -> tensor<1x32x32x16xf32> {
  %3 = "tf.Const"()  {value = dense<[1, 32, 32, 16]> : tensor<4xi32>}  : () -> tensor<4xi32>
  %4 = "tf.Conv2DBackpropInput"(%3, %arg1, %arg0)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : (tensor<4xi32>, tensor<1x1x16x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x16xf32>
  func.return %4 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv3d
// CHECK-SAME: %[[VAL_0:.*]]: tensor<2x4x128x128x8xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x2x4xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>}
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]> : tensor<5xi32>}
// CHECK: %[[VAL_4:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.conv3d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_2]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 0, 1, 0, 1>, stride = array<i64: 1, 2, 2>}
func.func @test_conv3d(%arg0: tensor<2x4x128x128x8xf32>, %arg1: tensor<2x3x3x2x4xf32>) -> tensor<2x4x64x64x4xf32> {
  %0 = "tf.Conv3D"(%arg0, %arg1) {data_format = "NDHWC", device = "", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 2, 1]} : (tensor<2x4x128x128x8xf32>, tensor<2x3x3x2x4xf32>) -> tensor<2x4x64x64x4xf32>
  return %0 : tensor<2x4x64x64x4xf32>
}

// -----

// CHECK-LABEL: test_conv3d_bias
// CHECK-SAME: %[[VAL_0:.*]]: tensor<3x32x16x16x5xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x5x10xf32>
// CHECK-SAME: %[[VAL_2:.*]]: tensor<10xf32>) -> tensor<3x32x16x16x10xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]> : tensor<5xi32>}
// CHECK: %[[VAL_4:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.conv3d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_2]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d_bias(%arg0: tensor<3x32x16x16x5xf32>, %arg1: tensor<2x3x3x5x10xf32>, %bias: tensor<10xf32>) -> tensor<3x32x16x16x10xf32> {
  %0 = "tf.Conv3D"(%arg0, %arg1) {data_format = "NDHWC", device = "", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<3x32x16x16x5xf32>, tensor<2x3x3x5x10xf32>) -> tensor<3x32x16x16x10xf32>
  %1 = "tf.BiasAdd"(%0, %bias) {data_format = "NHWC", device = ""} : (tensor<3x32x16x16x10xf32>, tensor<10xf32>) -> tensor<3x32x16x16x10xf32>
  return %1 : tensor<3x32x16x16x10xf32>
}

// -----

// CHECK-LABEL: test_add
// CHECK: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %2 = "tf.Add"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: %[[VAR0:.*]] = "tosa.sub"(%arg0, %arg1)
func.func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Sub"(%arg0, %arg1)   : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32}
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Mul"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_real_div
// CHECK: %[[VAR0:.*]] = "tosa.div"(%arg0, %arg1)
func.func @test_real_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.RealDiv"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  func.return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_floor_div
// CHECK: %[[VAR0:.*]] = "tosa.div"(%arg0, %arg1)
func.func @test_floor_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.FloorDiv"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  func.return %2 : tensor<13x21x3xi32>
}


// -----

// CHECK-LABEL: test_exp
// CHECK: %[[VAR0:.*]] = "tosa.exp"(%arg0)
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Exp"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK: %[[VAR0:.*]] = "tosa.reciprocal"(%arg0)
func.func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Reciprocal"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu
// CHECK: %[[VAR0:.*]] = "tosa.clamp"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Relu"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: %[[VAR0:.*]] = "tosa.clamp"(%arg0) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Relu6"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
func.func @test_leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "tf.LeakyRelu"(%arg0) {alpha = 0.5 : f32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: test_concat
// CHECK: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64}
func.func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ConcatV2"(%arg0, %arg1, %2)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<i32>) -> tensor<26x21x3xf32>
  func.return %3 : tensor<26x21x3xf32>
}

// -----

// CHECK-LABEL: test_bitwise_and
// CHECK: %[[VAR0:.*]] = "tosa.bitwise_and"(%arg0, %arg1)
func.func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseAnd"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  func.return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_bitwise_or
// CHECK: %[[VAR0:.*]] = "tosa.bitwise_or"(%arg0, %arg1)
func.func @test_bitwise_or(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseOr"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  func.return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_bitwise_not
// CHECK: %[[VAR0:.*]] = "tosa.bitwise_not"(%arg0)
func.func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %2 = "tf.Invert"(%arg0)   : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  func.return %2 : tensor<13x21x1xi32>
}

// -----

// CHECK-LABEL: test_bitwise_xor
// CHECK: %[[VAR0:.*]] = "tosa.bitwise_xor"(%arg0, %arg1)
func.func @test_bitwise_xor(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseXor"(%arg0, %arg1)   : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  func.return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: %[[VAR0:.*]] = "tosa.logical_and"(%arg0, %arg1)
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %2 = "tf.LogicalAnd"(%arg0, %arg1)   : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: %[[VAR0:.*]] = "tosa.logical_or"(%arg0, %arg1)
func.func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %2 = "tf.LogicalOr"(%arg0, %arg1)   : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: %[[VAR0:.*]] = "tosa.logical_not"(%arg0)
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %2 = "tf.LogicalNot"(%arg0)   : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  func.return %2 : tensor<1x21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_any"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Any"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  func.return %3 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_all
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_all"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.All"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  func.return %3 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_min"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Min"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_max"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Max"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Sum"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum_nonzero_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10x20x30x40x50xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<10x20x30x40x50xf32>, tensor<5xi32>) -> tensor<10x20x30x50x40xf32>
// CHECK: %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_2]]) {new_shape = array<i64: 300000, 40>} : (tensor<10x20x30x50x40xf32>) -> tensor<300000x40xf32>
// CHECK: %[[VAL_4:.*]] = "tosa.reduce_sum"(%[[VAL_3]]) {axis = 1 : i64} : (tensor<300000x40xf32>) -> tensor<300000x1xf32>
// CHECK: %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_4]]) {new_shape = array<i64: 10, 20, 30, 50>} : (tensor<300000x1xf32>) -> tensor<10x20x30x50xf32>
// CHECK: return %[[VAL_5]] : tensor<10x20x30x50xf32>
func.func @test_reduce_sum_nonzero_axis(%arg0: tensor<10x20x30x40x50xf32> {tf._user_specified_name = "inp_list"}) -> tensor<10x20x30x50xf32> {
  %cst = "tf.Const"() {device = "", value = dense<3> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Sum"(%arg0, %cst) {device = "", keep_dims = false} : (tensor<10x20x30x40x50xf32>, tensor<i32>) -> tensor<10x20x30x50xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<10x20x30x50xf32>) -> tensor<10x20x30x50xf32>
  func.return %1 : tensor<10x20x30x50xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.0769230798> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = array<i64: 21, 3>}
// CHECK: %[[VAR4:.*]] = "tosa.mul"(%[[VAR2]], %[[VAR0]]) {shift = 0 : i32}
func.func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Mean"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_prod"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Prod"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: %[[VAR0:.*]] = "tosa.minimum"(%arg0, %arg1)
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Minimum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: %[[VAR0:.*]] = "tosa.maximum"(%arg0, %arg1)
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Maximum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: %[[VAR0:.*]] = "tosa.pow"(%arg0, %arg1)
func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Pow"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: %[[VAR0:.*]] = "tosa.abs"(%arg0)
func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Abs"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: %[[VAR0:.*]] = "tosa.ceil"(%arg0)
func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Ceil"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: %[[VAR0:.*]] = "tosa.floor"(%arg0)
func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Floor"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: %[[VAR0:.*]] = "tosa.log"(%arg0)
func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Log"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_negate
// CHECK: %[[VAR0:.*]] = "tosa.negate"(%arg0)
func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Neg"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK: %[[VAR0:.*]] = "tosa.rsqrt"(%arg0)
func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Rsqrt"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sin
// CHECK-SAME: -> tensor<10xf32>
func.func @test_sin(%arg0: tensor<10xf32>) -> tensor<*xf32> {
  // CHECK-DAG: %[[ONE:.+]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[TWO:.+]] = "tosa.const"() {value = dense<2.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[RESULT_SCALE:.+]] = "tosa.const"() {value = dense<2.38418579E-7> : tensor<1xf32>}
  // CHECK-DAG: %[[INT_MAX:.+]] = "tosa.const"() {value = dense<3.276700e+04> : tensor<1xf32>}
  // CHECK-DAG: %[[IN_SCALE:.+]] = "tosa.const"() {value = dense<0.159154937> : tensor<1xf32>}
  // CHECK-DAG: %[[TBLVAL:.+]] = "tosa.const"() {value = dense<{{.+}}> : tensor<513xi16>}
  // CHECK-DAG: %[[IN_SCALED:.+]] = "tosa.mul"(%arg0, %[[IN_SCALE]])
  // CHECK-DAG: %[[FLOOR:.+]] = "tosa.floor"(%[[IN_SCALED]])
  // CHECK-DAG: %[[SUB1:.+]] = "tosa.sub"(%[[IN_SCALED]], %[[FLOOR]])
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.mul"(%[[SUB1]], %[[TWO]])
  // CHECK-DAG: %[[SUB2:.+]] = "tosa.sub"(%[[MUL1]], %[[ONE]])
  // CHECK-DAG: %[[MUL2:.+]] = "tosa.mul"(%[[SUB2]], %[[INT_MAX]])
  // CHECK-DAG: %[[TO_INT:.+]] = "tosa.cast"(%[[MUL2]])
  // CHECK-DAG: %[[TABLE:.+]] = "tosa.table"(%[[TO_INT]], %[[TBLVAL]])
  // CHECK-DAG: %[[TABLE_CAST:.+]] = "tosa.cast"(%[[TABLE]])
  // CHECK-DAG: %[[RESULT:.+]] = "tosa.mul"(%[[TABLE_CAST:.+]], %[[RESULT_SCALE]])
  %0 = "tf.Sin"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_cos
// CHECK-SAME: -> tensor<10xf32>
func.func @test_cos(%arg0: tensor<10xf32>) -> tensor<*xf32> {
  // CHECK-DAG: %[[RESULT_SCALE:.+]] = "tosa.const"() {value = dense<2.38418579E-7> : tensor<1xf32>}
  // CHECK-DAG: %[[INT_MAX:.+]] = "tosa.const"() {value = dense<3.276700e+04> : tensor<1xf32>}
  // CHECK-DAG: %[[ONE:.+]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[TWO:.+]] = "tosa.const"() {value = dense<2.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[IN_SCALE:.+]] = "tosa.const"() {value = dense<0.159154937> : tensor<1xf32>}
  // CHECK-DAG: %[[HALF_PI:.+]] = "tosa.const"() {value = dense<1.57079637> : tensor<1xf32>}
  // CHECK-DAG: %[[TBLVAL:.+]] = "tosa.const"() {value = dense<{{.+}}> : tensor<513xi16>}
  // CHECK-DAG: %[[IN_TRANSLATE:.+]] = "tosa.add"(%arg0, %[[HALF_PI]])
  // CHECK-DAG: %[[IN_SCALED:.+]] = "tosa.mul"(%[[IN_TRANSLATE]], %[[IN_SCALE]])
  // CHECK-DAG: %[[FLOOR:.+]] = "tosa.floor"(%[[IN_SCALED]])
  // CHECK-DAG: %[[SUB1:.+]] = "tosa.sub"(%[[IN_SCALED]], %[[FLOOR]])
  // CHECK-DAG: %[[MUL1:.+]] = "tosa.mul"(%[[SUB1]], %[[TWO]])
  // CHECK-DAG: %[[SUB2:.+]] = "tosa.sub"(%[[MUL1]], %[[ONE]])
  // CHECK-DAG: %[[MUL2:.+]] = "tosa.mul"(%[[SUB2]], %[[INT_MAX]])
  // CHECK-DAG: %[[TO_INT:.+]] = "tosa.cast"(%[[MUL2]])
  // CHECK-DAG: %[[TABLE:.+]] = "tosa.table"(%[[TO_INT]], %[[TBLVAL]])
  // CHECK-DAG: %[[TABLE_CAST:.+]] = "tosa.cast"(%[[TABLE]])
  // CHECK-DAG: %[[RESULT:.+]] = "tosa.mul"(%[[TABLE_CAST:.+]], %[[RESULT_SCALE]])
  %0 = "tf.Cos"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_sigmoid
// CHECK: %[[VAR0:.*]] = "tosa.sigmoid"(%arg0)
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Sigmoid"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32}
func.func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Square"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: %[[VAR0:.*]] = "tosa.equal"(%arg0, %arg1)
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Equal"(%arg0, %arg1)  {incompatible_shape_error = true}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: %[[VAR0:.*]] = "tosa.greater_equal"(%arg0, %arg1)
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.GreaterEqual"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: %[[VAR0:.*]] = "tosa.greater"(%arg0, %arg1)
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Greater"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK-DAG: %[[VAR0:.*]] = "tosa.greater_equal"(%arg0, %arg1)
// CHECK: %[[VAR1:.*]] = "tosa.logical_not"(%[[VAR0]])
func.func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Less"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK-DAG: %[[VAR0:.*]] = "tosa.greater"(%arg0, %arg1)
// CHECK: %[[VAR1:.*]] = "tosa.logical_not"(%[[VAR0]])
func.func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.LessEqual"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xi1>
  func.return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_argmax
// CHECK: %[[VAR0:.*]] = "tosa.argmax"(%arg0) {axis = 0 : i64}
func.func @test_argmax(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xi32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ArgMax"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<21x3xi32>
  func.return %3 : tensor<21x3xi32>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %2 = "tf.AvgPool"(%arg0)  {data_format = "NHWC", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  func.return %2 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %2 = "tf.MaxPool"(%arg0)  {data_format = "NHWC", explicit_paddings = [], ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  func.return %2 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 819>}
func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %0 = "tf.Const"()  {value = dense<[1, 819]> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Reshape"(%arg0, %0)   : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<1x819xf32>
  %4 = "tf.Identity"(%3)   : (tensor<1x819xf32>) -> tensor<1x819xf32>
  func.return %4 : tensor<1x819xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
func.func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %2 = "tf.Const"()  {value = dense<[2, 0, 1]> : tensor<3xi32>}  : () -> tensor<3xi32>
  %3 = "tf.Transpose"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  func.return %3 : tensor<3x13x21xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 4, 11, 1>, start = array<i64: 6, 8, 0>}
func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %2 = "tf.Const"()  {value = dense<[6, 8, 0]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %3 = "tf.Const"()  {value = dense<[4, 11, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %4 = "tf.Slice"(%arg0, %2, %3)   : (tensor<13x21x3xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x11x1xf32>
  func.return %4 : tensor<4x11x1xf32>
}

// -----

// CHECK-LABEL: test_strided_slice
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 21, 2>, start = array<i64: 4, 0, 1>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 9, 1, 7, 3, 2, 1>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.slice"(%[[VAR1]]) {size = array<i64: 9, 1, 7, 1, 2, 1>, start = array<i64: 0, 0, 0, 0, 0, 0>}
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 9, 7, 2>}
func.func @test_strided_slice(%arg0: tensor<13x21x3xf32>) -> tensor<9x7x2xf32> {
  %2 = "tf.Const"()  {value = dense<[4, 0, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %3 = "tf.Const"()  {value = dense<[13, 21, 3]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %4 = "tf.Const"()  {value = dense<[1, 3, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %5 = "tf.StridedSlice"(%arg0, %2, %3, %4)  {begin_mask = 2 : i64, ellipsis_mask = 0 : i64, end_mask = 3 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}  : (tensor<13x21x3xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<9x7x2xf32>
  func.return %5 : tensor<9x7x2xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg2) {new_shape = array<i64: 1, 1, 1>} : (tensor<1xi1>) -> tensor<1x1x1xi1>
// CHECK: %[[VAR2:.*]] = "tosa.select"(%[[VAR1]], %arg0, %arg1)
func.func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<1xi1>) -> tensor<13x21x3xf32> {
  %2 = "tf.SelectV2"(%arg2, %arg0, %arg1)   : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK-DAG: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.add"(%arg2, %[[VAR0]])
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg3, %[[VAR1]])
func.func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.AddN"(%arg0, %arg1, %arg2, %arg3)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
func.func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3, %2)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<i32>) -> tensor<52x21x3xf32>
  func.return %3 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK-DAG: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 4, 13, 21, 3>}
func.func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %2 = "tf.Pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i64}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  func.return %2 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 32, 32, 8>}
func.func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32> {
  %2 = "tf.Unpack"(%arg0)  {axis = 0 : i64}  : (tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32>
  %3 = "tf.Identity"(%2)   : (tensor<32x32x8xf32>) -> tensor<32x32x8xf32>
  func.return %3 : tensor<32x32x8xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1> : tensor<3x2xi32>}
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK: %[[VAR1:.*]] = "tosa.pad"(%arg0, %[[VAR0]], %[[PVAL]])
func.func @test_pad(%arg0: tensor<13x21x3xf32>) -> tensor<15x23x5xf32> {
  %2 = "tf.Const"()  {value = dense<1> : tensor<3x2xi32>}  : () -> tensor<3x2xi32>
  %3 = "tf.Pad"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<15x23x5xf32>
  func.return %3 : tensor<15x23x5xf32>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 21, 3>}
func.func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ExpandDims"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<1x13x21x3xf32>
  func.return %3 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_negative_index
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 13, 1, 21, 3>}
func.func @test_expand_dims_negative_index(%arg0: tensor<13x21x3xf32>) -> tensor<13x1x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<-2> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.ExpandDims"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<13x1x21x3xf32>
  func.return %3 : tensor<13x1x21x3xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[13, 21, 3]> : tensor<3xi32>}
func.func @test_shape() -> tensor<3xi32> {
  %3 = "tf.Const"()  {value = dense<[13, 21, 3]> : tensor<3xi32>}  : () -> tensor<3xi32>
  func.return %3 : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<3> : tensor<i32>}
func.func @test_rank() -> tensor<i32> {
  %3 = "tf.Const"()  {value = dense<3> : tensor<i32>}  : () -> tensor<i32>
  func.return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: test_elu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR4:.*]] = "tosa.sub"(%[[VAR2]], %[[VAR0]])
// CHECK-DAG: %[[VAR6:.*]] = "tosa.greater_equal"(%arg0, %[[VAR1]])
// CHECK: %[[VAR7:.*]] = "tosa.select"(%[[VAR6]], %arg0, %[[VAR4]])
func.func @test_elu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Elu"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Softmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
// CHECK: %[[VAR4:.*]] = "tosa.log"(%[[VAR3]])
func.func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.LogSoftmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_batch_matmul_3d
// CHECK: %[[VAR0:.*]] = "tosa.matmul"(%arg0, %arg1)
func.func @test_batch_matmul_3d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x3x42xf32>) -> tensor<13x21x42xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false, device = ""} : (tensor<13x21x3xf32>, tensor<13x3x42xf32>) -> tensor<13x21x42xf32>
  func.return %0 : tensor<13x21x42xf32>
}

// -----

// CHECK-LABEL: test_batch_matmul_4d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 65, 21, 3>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 65, 3, 42>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.matmul"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 5, 13, 21, 42>}
func.func @test_batch_matmul_4d(%arg0: tensor<5x13x21x3xf32>, %arg1: tensor<5x13x3x42xf32>) -> tensor<5x13x21x42xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = false, adj_y = false, device = ""} : (tensor<5x13x21x3xf32>, tensor<5x13x3x42xf32>) -> tensor<5x13x21x42xf32>
  func.return %0 : tensor<5x13x21x42xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 14, 19>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 19, 28>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.matmul"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 14, 28>}
func.func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %2 = "tf.MatMul"(%arg0, %arg1)  {transpose_a = false, transpose_b = false}  : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  func.return %2 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1xf32>}
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg0, %[[VAR0]])
func.func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<1.000000e+00> : tensor<f32>}  : () -> tensor<f32>
  %3 = "tf.Add"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<13x21x3xf32>
  func.return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg1) {axis = 0 : i64}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 1 : i64}
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg0, %[[VAR1]])
func.func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tf.Const"()  {value = dense<[0, 1]> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Sum"(%arg1, %0)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3xf32>
  %4 = "tf.Add"(%arg0, %3)   : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<13x21x3xf32>
  func.return %4 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_split
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 0, 0>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 7, 0>}
// CHECK: %[[VAR2:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 14, 0>}
func.func @test_split(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %6 = "tf.Const"()  {value = dense<1> : tensor<i32>}  : () -> tensor<i32>
  %7:3 = "tf.Split"(%6, %arg0)   : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  func.return %7#0, %7#1, %7#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_tile
// CHECK: tosa.tile
func.func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %2 = "tf.Const"()  {value = dense<[3, 1, 2]> : tensor<3xi32>}  : () -> tensor<3xi32>
  %3 = "tf.Tile"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<39x21x6xf32>
  %4 = "tf.Identity"(%3)   : (tensor<39x21x6xf32>) -> tensor<39x21x6xf32>
  func.return %4 : tensor<39x21x6xf32>
}

// -----

// CHECK-LABEL: test_reverse
// CHECK: tosa.reverse
func.func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.ReverseV2"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<13x21x3xf32>
  func.return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_space_to_batch
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[}}[0, 0], [0, 1], [0, 0]]> : tensor<3x2xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>}
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.pad"(%arg0, %[[VAR0]], %[[PVAL]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 13, 11, 2, 3>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = array<i64: 26, 11, 3>}
func.func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %2 = "tf.Const"()  {value = dense<2> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Const"()  {value = dense<[[0, 1]]> : tensor<1x2xi32>}  : () -> tensor<1x2xi32>
  %4 = "tf.SpaceToBatchND"(%arg0, %2, %3)   : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  func.return %4 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[3, 1, 2, 0]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 2, 2, 2, 32, 32, 1>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = array<i64: 2, 64, 64, 1>}
// CHECK: return %[[VAR5]]
func.func @test_batch_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<2x64x64x1xf32> {
  %2 = "tf.Const"()  {value = dense<2> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Const"()  {value = dense<0> : tensor<2x2xi32>}  : () -> tensor<2x2xi32>
  %4 = "tf.Const"()  {value = dense<[3, 1, 2, 0]> : tensor<4xi32>}  : () -> tensor<4xi32>
  %5 = "tf.Transpose"(%arg0, %4)   : (tensor<1x32x32x8xf32>, tensor<4xi32>) -> tensor<8x32x32x1xf32>
  %6 = "tf.BatchToSpaceND"(%5, %2, %3)   : (tensor<8x32x32x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<2x64x64x1xf32>
  func.return %6 : tensor<2x64x64x1xf32>
}

// -----

// CHECK-LABEL: test_space_to_depth
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 16, 2, 16, 2, 8>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 1, 16, 16, 32>}
func.func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %2 = "tf.SpaceToDepth"(%arg0)  {block_size = 2 : i64, data_format = "NHWC"}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  func.return %2 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 32, 32, 2, 2, 2>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 1, 64, 64, 2>}
func.func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %2 = "tf.DepthToSpace"(%arg0)  {block_size = 2 : i64, data_format = "NHWC"}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  func.return %2 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: test_left_shift
// CHECK: %[[VAR0:.*]] = "tosa.logical_left_shift"(%arg0, %arg1)
func.func @test_left_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1xi32>) -> tensor<4x4xi32> {
  %0 = "tf.LeftShift"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<1x1xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: test_right_shift
// CHECK: %[[VAR0:.*]] = "tosa.arithmetic_right_shift"(%arg0, %arg1) {round = false}
func.func @test_right_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1xi32>) -> tensor<4x4xi32> {
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<1x1xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @test_one_hot
// CHECK-SAME:      %[[ARG0_0:.*]]: tensor<4x4xi32>, %[[ARG1_0:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>
// CHECK:         %[[RESHAPE_0:.*]] = "tosa.reshape"(%[[ARG1_0]]) {new_shape = array<i64: 1, 1, 1>}
// CHECK:         %[[TILE:.*]] = "tosa.tile"(%[[RESHAPE_0]]) {multiples = array<i64: 16, 1, 1>}
// CHECK:         %[[RESHAPE_1:.*]] = "tosa.reshape"(%[[ARG2]]) {new_shape = array<i64: 1, 1, 1>}
// CHECK:         %[[TILE_0:.*]] = "tosa.tile"(%[[RESHAPE_1]]) {multiples = array<i64: 16, 2, 1>}
// CHECK:         %[[RESHAPE_2:.*]] = "tosa.reshape"(%[[ARG0_0]]) {new_shape = array<i64: 16, 1>}
// CHECK:         %[[SCATTER:.*]] = "tosa.scatter"(%[[TILE_0]], %[[RESHAPE_2]], %[[TILE]])
// CHECK:         %[[RESHAPE_3:.*]] = "tosa.reshape"(%[[SCATTER]]) {new_shape = array<i64: 4, 4, 2>}
// CHECK:         return %[[RESHAPE_3]]
func.func @test_one_hot(%arg0: tensor<4x4xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<4x4x2xf32> {
  %0 = "tf.Const"()  {value = dense<2> : tensor<i32>}  : () -> tensor<i32>
  %1 = "tf.OneHot"(%arg0, %0, %arg1, %arg2) {axis = -1 : i64} : (tensor<4x4xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<4x4x2xf32>
  func.return %1 : tensor<4x4x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<-2.00003052> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<1.99996948> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<6.10360876E-5> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() {value = dense<16383.75> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.minimum"(%arg0, %[[VAR1]])
// CHECK-DAG: %[[VAR8:.*]] = "tosa.maximum"(%[[VAR6]], %[[VAR0]])
// CHECK-DAG: %[[VAR10:.*]] = "tosa.sub"(%[[VAR8]], %[[VAR0]])
// CHECK-DAG: %[[VAR12:.*]] = "tosa.mul"(%[[VAR10]], %[[VAR3]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR14:.*]] = "tosa.add"(%[[VAR12]], %[[VAR4]])
// CHECK-DAG: %[[VAR15:.*]] = "tosa.floor"(%[[VAR14]])
// CHECK-DAG: %[[VAR17:.*]] = "tosa.mul"(%[[VAR15]], %[[VAR2]]) {shift = 0 : i32}
// CHECK: %[[VAR19:.*]] = "tosa.add"(%[[VAR17]], %[[VAR0]])
func.func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.FakeQuantWithMinMaxArgs"(%arg0)  {max = 2.000000e+00 : f32, min = -2.000000e+00 : f32, narrow_range = false, num_bits = 16 : i64}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: test_gather
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<1x49xi32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 63>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.gather"(%[[VAR4]], %[[VAR0]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: 7, 7, 21, 3>}
// CHECK: return %[[VAR7]]
func.func @test_gather(%arg0: tensor<13x21x3xf32>) -> tensor<7x7x21x3xf32> {
  %0 = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {device = "", value = dense<[[9, 8, 11, 10, 11, 0, 12], [7, 0, 1, 5, 2, 9, 6], [7, 9, 10, 8, 0, 3, 0], [8, 9, 10, 4, 9, 12, 6], [0, 2, 4, 3, 6, 11, 8], [5, 8, 7, 2, 4, 10, 5], [4, 12, 8, 3, 12, 9, 1]]> : tensor<7x7xi32>} : () -> tensor<7x7xi32>
  %2 = "tf.GatherV2"(%arg0, %1, %0) {batch_dims = 0 : i64, device = ""} : (tensor<13x21x3xf32>, tensor<7x7xi32>, tensor<i32>) -> tensor<7x7x21x3xf32>
  %3 = "tf.Identity"(%2) {device = ""} : (tensor<7x7x21x3xf32>) -> tensor<7x7x21x3xf32>
  func.return %2 : tensor<7x7x21x3xf32>
}

// -----
// CHECK-LABEL: test_gather_nd
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<1x42xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 63>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.gather"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 6, 7, 21, 3>}
func.func @test_gather_nd(%arg0: tensor<13x21x3xf32>) -> tensor<6x7x21x3xf32> {
  %0 = "tf.Const"() {device = "", value = dense<[[[0], [5], [3], [12], [2], [4], [3]], [[11], [1], [11], [10], [3], [12], [8]], [[5], [3], [1], [11], [3], [10], [0]], [[0], [8], [4], [7], [3], [12], [2]], [[7], [6], [11], [4], [2], [10], [11]], [[11], [1], [11], [1], [1], [11], [8]]]> : tensor<6x7x1xi32>} : () -> tensor<6x7x1xi32>
  %1 = "tf.GatherNd"(%arg0, %0) {device = ""} : (tensor<13x21x3xf32>, tensor<6x7x1xi32>) -> tensor<6x7x21x3xf32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<6x7x21x3xf32>) -> tensor<6x7x21x3xf32>
  func.return %1 : tensor<6x7x21x3xf32>
}


// -----

// CHECK-LABEL: test_fused_batch_norm
func.func @test_fused_batch_norm(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK:  %[[ONE:.+]] = "tosa.const"() {value = dense<1.000000e-03> : tensor<1xf32>}
  // CHECK:  %[[RES0:.+]] = "tosa.reshape"(%arg3) {new_shape = array<i64: 1, 1, 1, 8>}
  // CHECK:  %[[SUB0:.+]] = "tosa.sub"(%arg0, %[[RES0]])
  // CHECK:  %[[ADD0:.+]] = "tosa.add"(%arg4, %[[ONE]])
  // CHECK:  %[[RSQR:.+]] = "tosa.rsqrt"(%[[ADD0]])
  // CHECK:  %[[RES1:.+]] = "tosa.reshape"(%[[RSQR]]) {new_shape = array<i64: 1, 1, 1, 8>}
  // CHECK:  %[[MUL0:.+]] = "tosa.mul"(%[[SUB0]], %[[RES1]]) {shift = 0 : i32}
  // CHECK:  %[[RES1:.+]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 1, 1, 8>}
  // CHECK:  %[[MUL1:.+]] = "tosa.mul"(%[[MUL0]], %[[RES1]]) {shift = 0 : i32}
  // CHECK:  %[[RES2:.+]] = "tosa.reshape"(%arg2) {new_shape = array<i64: 1, 1, 1, 8>}
  // CHECK:  %[[ADD1:.+]] = "tosa.add"(%[[MUL1]], %[[RES2]])
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)

  // CHECK: return %[[ADD1]]
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: test_fused_batch_norm_training
func.func @test_fused_batch_norm_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "tf.FusedBatchNormV3"
  %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// CHECK-LABEL: mirrorpad_symmetric
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5x10xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 10>, start = array<i64: 0, 0>} : (tensor<5x10xf32>)
// CHECK: %[[VAL_2:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 2, 10>, start = array<i64: 3, 0>} : (tensor<5x10xf32>)
// CHECK: %[[VAL_3:.*]] = "tosa.reverse"(%[[VAL_2]]) {axis = 0 : i64} : (tensor<2x10xf32>)
// CHECK: %[[VAL_4:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_0]], %[[VAL_3]]) {axis = 0 : i64} : (tensor<1x10xf32>, tensor<5x10xf32>, tensor<2x10xf32>)
// CHECK: %[[VAL_5:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 8, 1>, start = array<i64: 0, 0>} : (tensor<8x10xf32>)
// CHECK: %[[VAL_6:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 8, 2>, start = array<i64: 0, 8>} : (tensor<8x10xf32>)
// CHECK: %[[VAL_7:.*]] = "tosa.reverse"(%[[VAL_6]]) {axis = 1 : i64} : (tensor<8x2xf32>)
// CHECK: %[[VAL_8:.*]] = "tosa.concat"(%[[VAL_5]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : i64} : (tensor<8x1xf32>, tensor<8x10xf32>, tensor<8x2xf32>)
func.func @mirrorpad_symmetric(%arg0: tensor<5x10xf32>) -> tensor<8x13xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[[1, 2], [1, 2]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %0 = "tf.MirrorPad"(%arg0, %cst) {device = "", mode = "SYMMETRIC"} : (tensor<5x10xf32>, tensor<2x2xi32>) -> tensor<8x13xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<8x13xf32>) -> tensor<8x13xf32>
  return %0 : tensor<8x13xf32>
}

// -----

// CHECK-LABEL: mirrorpad_reflect
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 21, 3>, start = array<i64: 1, 0, 0>} : (tensor<13x21x3xf32>)
// CHECK: %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_0]]) {axis = 0 : i64} : (tensor<1x21x3xf32>, tensor<13x21x3xf32>)
// CHECK: %[[VAL_3:.*]] = "tosa.slice"(%[[VAL_2]]) {size = array<i64: 14, 1, 3>, start = array<i64: 0, 1, 0>} : (tensor<14x21x3xf32>)
// CHECK: %[[VAL_4:.*]] = "tosa.concat"(%[[VAL_3]], %[[VAL_2]]) {axis = 1 : i64} : (tensor<14x1x3xf32>, tensor<14x21x3xf32>)
// CHECK: %[[VAL_5:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 14, 22, 1>, start = array<i64: 0, 0, 1>} : (tensor<14x22x3xf32>)
// CHECK: %[[VAL_6:.*]] = "tosa.concat"(%[[VAL_5]], %[[VAL_4]]) {axis = 2 : i64} : (tensor<14x22x1xf32>, tensor<14x22x3xf32>)
func.func @mirrorpad_reflect(%arg0: tensor<13x21x3xf32>) -> tensor<14x22x4xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[[1, 0], [1, 0], [1, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %0 = "tf.MirrorPad"(%arg0, %cst) {device = "", mode = "REFLECT"} : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<14x22x4xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<14x22x4xf32>) -> tensor<14x22x4xf32>
  return %0 : tensor<14x22x4xf32>
}
