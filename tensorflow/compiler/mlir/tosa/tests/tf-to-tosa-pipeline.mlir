// RUN: tf-opt --tf-to-tosa-pipeline  --verify-each %s | FileCheck %s

// Operations for testing tf-to-tosa-pipeline
// TODO: These tests are fairly minimal. Expand the checks to be more robust.

// -----

// CHECK-LABEL: test_conv2d
// CHECK: tosa.const
// CHECK: tosa.const
// CHECK: tosa.transpose
// CHECK: tosa.conv2d
func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x8x16xf32>) -> tensor<1x32x32x16xf32> {
  %3 = "tf.Conv2D"(%arg0, %arg1)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : (tensor<1x32x32x8xf32>, tensor<1x1x8x16xf32>) -> tensor<1x32x32x16xf32>
  return %3 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d
// CHECK: tosa.const
// CHECK: tosa.depthwise_conv2d
func @test_depthwise_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x8x2xf32>) -> tensor<1x32x32x16xf32> {
  %5 = "tf.DepthwiseConv2dNative"(%arg0, %arg1)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>, tensor<1x1x8x2xf32>) -> tensor<1x32x32x16xf32>
  %6 = "tf.Identity"(%5)   : (tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
  return %6 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d
// CHECK-DAG: "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>}
// CHECK-DAG: "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK-DAG: tosa.transpose
// CHECK: tosa.transpose_conv2d
func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x1x16x8xf32>) -> tensor<1x32x32x16xf32> {
  %3 = "tf.Const"()  {value = dense<[1, 32, 32, 16]> : tensor<4xi32>}  : () -> tensor<4xi32>
  %4 = "tf.Conv2DBackpropInput"(%3, %arg1, %arg0)  {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}  : (tensor<4xi32>, tensor<1x1x16x8xf32>, tensor<1x32x32x8xf32>) -> tensor<1x32x32x16xf32>
  return %4 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_add
// CHECK: tosa.add
func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Add"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: tosa.sub
func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Sub"(%arg0, %arg1)   : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK: tosa.mul
func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Mul"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_exp
// CHECK: tosa.exp
func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Exp"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK: tosa.reciprocal
func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Reciprocal"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu
// CHECK: tosa.reluN
func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Relu"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: tosa.reluN
func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Relu6"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.LeakyRelu"(%arg0)  {alpha = 0.707330704 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concat
// CHECK: tosa.concat
func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ConcatV2"(%arg0, %arg1, %2)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<i32>) -> tensor<26x21x3xf32>
  return %3 : tensor<26x21x3xf32>
}

// -----

// CHECK-LABEL: test_bitwise_and
// CHECK: tosa.bitwise_and
func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseAnd"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_bitwise_or
// CHECK: tosa.bitwise_or
func @test_bitwise_or(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseOr"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_bitwise_not
// CHECK: tosa.bitwise_not
func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %2 = "tf.Invert"(%arg0)   : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  return %2 : tensor<13x21x1xi32>
}

// -----

// CHECK-LABEL: test_bitwise_xor
// CHECK: tosa.bitwise_xor
func @test_bitwise_xor(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %2 = "tf.BitwiseXor"(%arg0, %arg1)   : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %2 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: tosa.logical_and
func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %2 = "tf.LogicalAnd"(%arg0, %arg1)   : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: tosa.logical_or
func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %2 = "tf.LogicalOr"(%arg0, %arg1)   : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: tosa.logical_not
func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %2 = "tf.LogicalNot"(%arg0)   : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %2 : tensor<1x21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK: tosa.reduce_any
// CHECK: tosa.reshape
func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Any"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  return %3 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_all
// CHECK: tosa.reduce_all
// CHECK: tosa.reshape
func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.All"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  return %3 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK: tosa.reduce_min
// CHECK: tosa.reshape
func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Min"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK: tosa.reduce_max
// CHECK: tosa.reshape
func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Max"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Sum"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK: "tosa.const"() {value = dense<0.0769230798>
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
// CHECK: tosa.reshape
// CHECK: tosa.mul
func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Mean"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK: tosa.reduce_prod
// CHECK: tosa.reshape
func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Prod"(%arg0, %2)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %3 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: tosa.minimum
func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Minimum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: tosa.maximum
func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Maximum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: tosa.pow
func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Pow"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: tosa.abs
func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Abs"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: tosa.ceil
func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Ceil"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: tosa.floor
func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Floor"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: tosa.log
func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Log"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_negate
// CHECK: tosa.negate
func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Neg"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK: tosa.rsqrt
func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Rsqrt"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sigmoid
// CHECK: tosa.sigmoid
func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Sigmoid"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK: tosa.mul
func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Square"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: tosa.equal
func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Equal"(%arg0, %arg1)  {incompatible_shape_error = true}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: tosa.greater_equal
func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.GreaterEqual"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: tosa.greater
func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Greater"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: tosa.greater_equal
// CHECK: tosa.logical_not
func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.Less"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK: tosa.greater
// CHECK: tosa.logical_not
func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xi1> {
  %2 = "tf.LessEqual"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xi1>
  return %2 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_argmax
// CHECK: tosa.argmax
func @test_argmax(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xi32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ArgMax"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<21x3xi32>
  return %3 : tensor<21x3xi32>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK: tosa.avg_pool2d
func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %2 = "tf.AvgPool"(%arg0)  {data_format = "NHWC", ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %2 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: tosa.max_pool2d
func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %2 = "tf.MaxPool"(%arg0)  {data_format = "NHWC", explicit_paddings = [], ksize = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %2 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK: tosa.reshape
func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %0 = "tf.Const"()  {value = dense<[1, 819]> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Reshape"(%arg0, %0)   : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<1x819xf32>
  %4 = "tf.Identity"(%3)   : (tensor<1x819xf32>) -> tensor<1x819xf32>
  return %4 : tensor<1x819xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK: tosa.const
// CHECK: tosa.transpose
func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %2 = "tf.Const"()  {value = dense<[2, 0, 1]> : tensor<3xi32>}  : () -> tensor<3xi32>
  %3 = "tf.Transpose"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %3 : tensor<3x13x21xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK: tosa.slice
func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %2 = "tf.Const"()  {value = dense<[6, 8, 0]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %3 = "tf.Const"()  {value = dense<[4, 11, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %4 = "tf.Slice"(%arg0, %2, %3)   : (tensor<13x21x3xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<4x11x1xf32>
  return %4 : tensor<4x11x1xf32>
}

// -----

// CHECK-LABEL: test_strided_slice
// CHECK: tosa.slice
// CHECK: tosa.reshape
// CHECK: tosa.slice
// CHECK: tosa.reshape
func @test_strided_slice(%arg0: tensor<13x21x3xf32>) -> tensor<9x7x2xf32> {
  %2 = "tf.Const"()  {value = dense<[4, 0, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %3 = "tf.Const"()  {value = dense<[13, 21, 3]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %4 = "tf.Const"()  {value = dense<[1, 3, 1]> : tensor<3xi64>}  : () -> tensor<3xi64>
  %5 = "tf.StridedSlice"(%arg0, %2, %3, %4)  {begin_mask = 2 : i64, ellipsis_mask = 0 : i64, end_mask = 3 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}  : (tensor<13x21x3xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<9x7x2xf32>
  return %5 : tensor<9x7x2xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK: tosa.const
// CHECK: tosa.reshape
// CHECK: tosa.select
func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<false> : tensor<1xi1>}  : () -> tensor<1xi1>
  %3 = "tf.SelectV2"(%2, %arg0, %arg1)   : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK: tosa.add
// CHECK: tosa.add
// CHECK: tosa.add
func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.AddN"(%arg0, %arg1, %arg2, %arg3)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.concat
func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3, %2)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<i32>) -> tensor<52x21x3xf32>
  return %3 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.reshape
func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %2 = "tf.Pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i64}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  return %2 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK: tosa.slice
// CHECK: tosa.reshape
// CHECK: tosa.identityn
func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32> {
  %2 = "tf.Unpack"(%arg0)  {axis = 0 : i64}  : (tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32>
  %3 = "tf.Identity"(%2)   : (tensor<32x32x8xf32>) -> tensor<32x32x8xf32>
  return %3 : tensor<32x32x8xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK: tosa.const
// CHECK: tosa.pad
func @test_pad(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<3x2xi32>}  : () -> tensor<3x2xi32>
  %3 = "tf.Pad"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK: tosa.reshape
func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.ExpandDims"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<1x13x21x3xf32>
  return %3 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: tosa.const
func @test_shape() -> tensor<3xi32> {
  %3 = "tf.Const"()  {value = dense<[13, 21, 3]> : tensor<3xi32>}  : () -> tensor<3xi32>
  return %3 : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: tosa.const
func @test_rank() -> tensor<i32> {
  %3 = "tf.Const"()  {value = dense<3> : tensor<i32>}  : () -> tensor<i32>
  return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: test_elu
// CHECK: tosa.const
// CHECK: tosa.const
// CHECK: tosa.exp
// CHECK: tosa.reshape
// CHECK: tosa.sub
// CHECK: tosa.reshape
// CHECK: tosa.greater_equal
// CHECK: tosa.select
func @test_elu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Elu"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Softmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.log
func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.LogSoftmax"(%arg0)   : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK: tosa.matmul
func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %2 = "tf.MatMul"(%arg0, %arg1)  {transpose_a = false, transpose_b = false}  : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %2 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK: tosa.const
// CHECK: tosa.reshape
// CHECK: tosa.add
func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<1.000000e+00> : tensor<f32>}  : () -> tensor<f32>
  %3 = "tf.Add"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<13x21x3xf32>
  return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK: tosa.reduce_sum
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
// CHECK: tosa.reshape
// CHECK: tosa.add
func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tf.Const"()  {value = dense<[0, 1]> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Sum"(%arg1, %0)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3xf32>
  %4 = "tf.Add"(%arg0, %3)   : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<13x21x3xf32>
  return %4 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d_const
// CHECK: tosa.add
func @test_add_1d_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %3 = "tf.Add"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_split
// CHECK: tosa.slice
// CHECK: tosa.slice
// CHECK: tosa.slice
// CHECK: tosa.identityn
func @test_split(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %6 = "tf.Const"()  {value = dense<1> : tensor<i32>}  : () -> tensor<i32>
  %7:3 = "tf.Split"(%6, %arg0)   : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  return %7#0, %7#1, %7#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_tile
// CHECK: tosa.tile
func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %2 = "tf.Const"()  {value = dense<[3, 1, 2]> : tensor<3xi32>}  : () -> tensor<3xi32>
  %3 = "tf.Tile"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<39x21x6xf32>
  %4 = "tf.Identity"(%3)   : (tensor<39x21x6xf32>) -> tensor<39x21x6xf32>
  return %4 : tensor<39x21x6xf32>
}

// -----

// CHECK-LABEL: test_reverse
// CHECK: tosa.reverse
func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.ReverseV2"(%arg0, %2)   : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<13x21x3xf32>
  return %3 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_gather
// CHECK: tosa.const
// CHECK: tosa.gather
func @test_gather(%arg0: tensor<13x21x3xi32>) -> tensor<26x21x3xi32> {
  %2 = "tf.Const"()  {value = dense<0> : tensor<i32>}  : () -> tensor<i32>
  %3 = "tf.Const"()  {value = dense<[2, 2, 7, 6, 6, 1, 5, 4, 2, 11, 10, 11, 7, 7, 5, 3, 12, 7, 11, 0, 9, 5, 4, 12, 1, 9]> : tensor<26xi32>}  : () -> tensor<26xi32>
  %4 = "tf.GatherV2"(%arg0, %3, %2)  {batch_dims = 0 : i64}  : (tensor<13x21x3xi32>, tensor<26xi32>, tensor<i32>) -> tensor<26x21x3xi32>
  return %4 : tensor<26x21x3xi32>
}

// -----

// CHECK-LABEL: test_space_to_batch
// CHECK-DAG: "tosa.const"() {value = dense<{{\[}}[0, 0], [0, 1], [0, 0]]>
// CHECK-DAG: "tosa.const"() {value = dense<[2, 0, 1, 3]>
// CHECK: tosa.pad
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %2 = "tf.Const"()  {value = dense<2> : tensor<1xi32>}  : () -> tensor<1xi32>
  %3 = "tf.Const"()  {value = dense<[[0, 1]]> : tensor<1x2xi32>}  : () -> tensor<1x2xi32>
  %4 = "tf.SpaceToBatchND"(%arg0, %2, %3)   : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  return %4 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: "tosa.const"() {value = dense<[3, 1, 2, 0]>
// CHECK-DAG: "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]>
// CHECK: tosa.transpose
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
// CHECK: tosa.slice
func @test_batch_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<2x64x64x1xf32> {
  %2 = "tf.Const"()  {value = dense<2> : tensor<2xi32>}  : () -> tensor<2xi32>
  %3 = "tf.Const"()  {value = dense<0> : tensor<2x2xi32>}  : () -> tensor<2x2xi32>
  %4 = "tf.Const"()  {value = dense<[3, 1, 2, 0]> : tensor<4xi32>}  : () -> tensor<4xi32>
  %5 = "tf.Transpose"(%arg0, %4)   : (tensor<1x32x32x8xf32>, tensor<4xi32>) -> tensor<8x32x32x1xf32>
  %6 = "tf.BatchToSpaceND"(%5, %2, %3)   : (tensor<8x32x32x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<2x64x64x1xf32>
  return %6 : tensor<2x64x64x1xf32>
}

// -----

// CHECK-LABEL: test_space_to_depth
// CHECK: "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]>
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %2 = "tf.SpaceToDepth"(%arg0)  {block_size = 2 : i64, data_format = "NHWC"}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  return %2 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK: "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]>
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %2 = "tf.DepthToSpace"(%arg0)  {block_size = 2 : i64, data_format = "NHWC"}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  return %2 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: "tosa.const"() {value = dense<16383.75> : tensor<f32>}
// CHECK-DAG: "tosa.const"() {value = dense<-1.000000e+00> : tensor<f32>}
// CHECK-DAG: "tosa.const"() {value = dense<6.10360876E-5> : tensor<f32>}
// CHECK: tosa.reshape
// CHECK: tosa.mul
// CHECK: tosa.reshape
// CHECK: tosa.add
// CHECK: tosa.cast
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tosa.cast
// CHECK: tosa.reshape
// CHECK: tosa.sub
// CHECK: tosa.reshape
// CHECK: tosa.mul
func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "tf.FakeQuantWithMinMaxArgs"(%arg0)  {max = 2.000000e+00 : f32, min = -2.000000e+00 : f32, narrow_range = false, num_bits = 16 : i64}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}
