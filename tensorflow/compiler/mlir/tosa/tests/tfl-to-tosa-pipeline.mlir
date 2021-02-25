// RUN: tf-opt --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s

// Operations for testing tfl-to-tosa-pipeline

// TODO: For all fakequant tests: compute and add checks on rescale attribute
// values
// TODO: These tests are fairly minimal. Expand the checks to be more robust.


// -----

// CHECK-LABEL: test_conv2d
// CHECK: tosa.const
// CHECK: tosa.conv2d
func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %cst_0, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv2d_bias
// CHECK: tosa.conv2d
func @test_conv2d_bias(%arg0: tensor<1x32x32x8xf32>, %cst: tensor<16x1x1x8xf32>, %cst_0: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d
// CHECK: tosa.const
// CHECK: tosa.transpose_conv2d
func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = constant unit
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_fakequant_conv2d
// CHECK: tosa.const
// CHECK: tosa.const
// CHECK: tosa.conv2d
// CHECK: tosa.rescale
func @test_fakequant_conv2d(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<"0x851F811ED39B1160E8BFD11A44C8815EC054BEB7658131420857498B9B7FA28499818C7AB44894E64B81C6C350A581E8042F48DB13B85A81EEE481FD28A43BBBC381A70384A46F47811C2A4D64D8D285DEDCE37F1FFC6B5BB0A3794EED7F98D9060BA5ED5EC6A37F7FF4E67364062F078AE9DDDF778155794C54AE536D7FAC05"> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// TODO: Compute and add checks on rescale attribute values

// CHECK-LABEL: test_fakequant_depthwise_conv2d_bias
// CHECK-DAG: "tosa.const"() {value = dense<[{{\[}}[{{\[}}-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x1x1x16xi8>} : () -> tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
// CHECK-DAG: "tosa.const"() {value = dense<[1, 2, 3, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG: "tosa.const"() {value = dense<[-2879, 6636, 3531, 23376, -79787, -6142, 5582, -30384, 17330, -4549, -3518, 16215, 2695, -2670, 8399, -12223]> : tensor<16xi32>} : () -> tensor<16xi32>
// CHECK: tosa.transpose
// CHECK: tosa.reshape
// CHECK: tosa.depthwise_conv2d
// CHECK: tosa.rescale
func @test_fakequant_depthwise_conv2d_bias(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x1x1x16xi8>} : () -> tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3,  {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5}>>, value = dense<[-2879, 6636, 3531, 23376, -79787, -6142, 5582, -30384, 17330, -4549, -3518, 16215, 2695, -2670, 8399, -12223]> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %0, %1) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>, tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0,   {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_add
// CHECK: tosa.add
func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: test_gather
// CHECK: tosa.gather
func @test_gather(%arg0: tensor<100x25xf32>, %arg1: tensor<1x20xi32>) -> tensor<20x25x3xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<100x25xf32>, tensor<1x20xi32>) -> tensor<20x25x3xf32>
  return %0 : tensor<20x25x3xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: tosa.sub
func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK: tosa.mul
func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_exp
// CHECK: tosa.exp
func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK: tosa.const
// CHECK: tosa.reciprocal
// CHECK: tosa.reshape
// CHECK: tosa.mul
func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.div"(%cst, %arg0)  {fused_activation_function = "NONE"}  : (tensor<f32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu
// CHECK: tosa.reluN
func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: tosa.reluN
func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.leaky_relu"(%arg0)  {alpha = 0.707330704 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concat
// CHECK: tosa.concat
func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1)  {axis = 0 : i32, fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: tosa.logical_and
func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: tosa.logical_or
func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: tosa.logical_not
func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %0 = "tfl.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK: tosa.reduce_any
// CHECK: tosa.reshape
func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  return %0 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK: tosa.reduce_min
// CHECK: tosa.reshape
func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_min"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK: tosa.reduce_max
// CHECK: tosa.reshape
func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK: "tosa.const"() {value = dense<0.0769230798>
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
// CHECK: tosa.reshape
// CHECK: tosa.mul
func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.mean"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK: tosa.reduce_prod
// CHECK: tosa.reshape
func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_prod"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: tosa.minimum
func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: tosa.maximum
func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: tosa.pow
func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: tosa.abs
func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: tosa.ceil
func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: tosa.floor
func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: tosa.log
func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_negate
// CHECK: tosa.negate
func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.neg"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK: tosa.rsqrt
func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sigmoid
// CHECK: tosa.sigmoid
func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK: tosa.mul
func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.square"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: tosa.equal
func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: tosa.greater_equal
func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: tosa.greater
func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: tosa.greater_equal
// CHECK: tosa.logical_not
func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK: tosa.greater
// CHECK: tosa.logical_not
func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK: tosa.avg_pool2d
func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: tosa.max_pool2d
func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK: tosa.reshape
func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %cst = constant dense<[1, 819]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK: tosa.const
// CHECK: tosa.transpose
func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %cst = constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK: tosa.slice
func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %cst = constant dense<[6, 8, 0]> : tensor<3xi32>
  %cst_0 = constant dense<[4, 11, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x11x1xf32>
  return %0 : tensor<4x11x1xf32>
}

// -----

// CHECK-LABEL: test_strided_slice
// CHECK: tosa.slice
// CHECK: tosa.reshape
// CHECK: tosa.slice
// CHECK: tosa.reshape
func @test_strided_slice(%arg0: tensor<13x21x3xf32>) -> tensor<9x7x2xf32> {
  %cst = constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<9x7x2xf32>
  return %0 : tensor<9x7x2xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK: tosa.const
// CHECK: tosa.reshape
// CHECK: tosa.select
func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<false> : tensor<1xi1>
  %0 = "tfl.select_v2"(%cst, %arg0, %arg1) : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK: tosa.add
// CHECK: tosa.add
// CHECK: tosa.add
func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2, %arg3) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.concat
func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  return %0 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.reshape
func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %0 = "tfl.pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, values_count = 4 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  return %0 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK: tosa.slice
// CHECK: tosa.reshape
// CHECK: tosa.identityn
func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32> {
  %0 = "tfl.unpack"(%arg0)  {axis = 0 : i32, num = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32>
  return %0 : tensor<32x32x8xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK: tosa.const
// CHECK: tosa.pad
func @test_pad(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<0> : tensor<3x2xi32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK: tosa.reshape
func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32> {
  %cst = constant dense<[1, 13, 21, 3]> : tensor<4xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<4xi32>) -> tensor<1x13x21x3xf32>
  return %0 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: tosa.const
func @test_shape() -> tensor<3xi32> {
  %cst = constant dense<[13, 21, 3]> : tensor<3xi32>
  return %cst : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: tosa.const
func @test_rank() -> tensor<i32> {
  %cst = constant dense<3> : tensor<i32>
  return %cst : tensor<i32>
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
  %0 = "tfl.elu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK: tosa.log
func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log_softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG: "tosa.const"() {value = dense<0.000000e+00> : tensor<28xf32>} : () -> tensor<28xf32>
// CHECK: tosa.transpose
// CHECK: tosa.fully_connected
func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %cst = constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = constant unit
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<19x28xf32>, tensor<2xi32>) -> tensor<28x19xf32>
  %1 = "tfl.fully_connected"(%arg0, %0, %cst_0)  {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}  : (tensor<14x19xf32>, tensor<28x19xf32>, none) -> tensor<14x28xf32>
  return %1 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK: tosa.const
// CHECK: tosa.reshape
// CHECK: tosa.add
func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK: tosa.reduce_sum
// CHECK: tosa.reduce_sum
// CHECK: tosa.reshape
// CHECK: tosa.reshape
// CHECK: tosa.add
func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<[0, 1]> : tensor<2xi32>
  %0 = "tfl.sum"(%arg1, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3xf32>
  %1 = "tfl.add"(%arg0, %0)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<13x21x3xf32>
  return %1 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d_const
// CHECK: tosa.add
func @test_add_1d_const(%arg0: tensor<13x21x3xf32>, %cst: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = tfl.add %arg0, %cst  {fused_activation_function = "NONE"}  : tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_split
// CHECK: tosa.slice
// CHECK: tosa.slice
// CHECK: tosa.slice
// CHECK: tosa.identityn
func @test_split(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %cst_0 = constant dense<1> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  return %0#0, %0#1, %0#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_tile
// CHECK: tosa.tile
func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %cst = constant dense<[3, 1, 2]> : tensor<3xi32>
  %0 = "tfl.tile"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<39x21x6xf32>
  return %0 : tensor<39x21x6xf32>
}

// -----

// CHECK-LABEL: test_space_to_batch
// CHECK-DAG: "tosa.const"() {value = dense<[{{\[}}0, 0], [0, 1], [0, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
// CHECK-DAG: "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: tosa.pad
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %cst = constant dense<2> : tensor<1xi32>
  %cst_0 = constant dense<[[0, 1]]> : tensor<1x2xi32>
  %0 = "tfl.space_to_batch_nd"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  return %0 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: "tosa.const"() {value = dense<[3, 1, 2, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG: "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
// CHECK: tosa.transpose
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
// CHECK: tosa.slice
func @test_batch_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<2x64x64x1xf32> {
  %cst = constant dense<2> : tensor<2xi32>
  %cst_0 = constant dense<0> : tensor<2x2xi32>
  %cst_1 = constant dense<[3, 1, 2, 0]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<1x32x32x8xf32>, tensor<4xi32>) -> tensor<8x32x32x1xf32>
  %1 = "tfl.batch_to_space_nd"(%0, %cst, %cst_0) : (tensor<8x32x32x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<2x64x64x1xf32>
  return %1 : tensor<2x64x64x1xf32>
}

// -----

// CHECK-LABEL: test_space_to_depth
// CHECK: "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %0 = "tfl.space_to_depth"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  return %0 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK: "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK: tosa.reshape
// CHECK: tosa.transpose
// CHECK: tosa.reshape
func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %0 = "tfl.depth_to_space"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  return %0 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: "tosa.const"() {value = dense<16383.75> : tensor<f32>}
// CHECK-DAG: "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
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
  %0 = "tfl.quantize"(%arg0)  {qtype = tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>}  : (tensor<13x21x3xf32>) -> tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>
  %1 = "tfl.dequantize"(%0) : (tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  %2 = "tfl.dequantize"(%0) : (tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_fakequant_add
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tosa.add
// CHECK: tosa.rescale
func @test_fakequant_add(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
}

// -----

// CHECK-LABEL: test_fakequant_sub
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tosa.sub
// CHECK: tosa.rescale
func @test_fakequant_sub(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
}

// -----

// CHECK-LABEL: test_fakequant_mul
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tosa.mul
// CHECK: tosa.rescale
func @test_fakequant_mul(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>>
}

// -----

// CHECK-LABEL: test_fakequant_avg_pool2d
// CHECK: tosa.avg_pool2d
func @test_fakequant_avg_pool2d(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
  return %0 : tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
}

// -----

// CHECK-LABEL: test_fakequant_max_pool2d
// CHECK: tosa.max_pool2d
func @test_fakequant_max_pool2d(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
  return %0 : tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
}

// -----

// TODO: add additional checks on the quantized softmax lowering,
// as it is one of the most complicated lowerings overall.

// CHECK-LABEL: test_fakequant_softmax
// CHECK-DAG: "tosa.const"() {value = dense<"{{.*}}"> : tensor<513xi16>} : () -> tensor<513x!quant.uniform<i16:f32, 1.000000e+00>>
// CHECK-DAG: "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: "tosa.const"() {value = dense<34> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: "tosa.const"() {value = dense<-2147483648> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: "tosa.const"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: "tosa.const"() {value = dense<"{{.*}}"> : tensor<513xi16>} : () -> tensor<513x!quant.uniform<i16:f32, 1.000000e+00>>
// CHECK: tosa.rescale
// CHECK: tosa.reduce_max
// CHECK: tosa.sub
// CHECK: tosa.rescale
// CHECK: tosa.table
// CHECK: tosa.reshape
// CHECK: tosa.arithmetic_right_shift
// CHECK: tosa.reduce_sum
// CHECK: tosa.clz
// CHECK: tosa.reshape
// CHECK: tosa.sub
// CHECK: tosa.logical_left_shift
// CHECK: tosa.reshape
// CHECK: tosa.sub
// CHECK: tosa.reshape
// CHECK: tosa.arithmetic_right_shift
// CHECK: tosa.cast
// CHECK: tosa.table
// CHECK: tosa.rescale
// CHECK: tosa.rescale
// CHECK: tosa.mul
// CHECK: tosa.arithmetic_right_shift
// CHECK: tosa.rescale
func @test_fakequant_softmax(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.0156164625659585>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.0156164625659585>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_fakequant_sigmoid
// CHECK: tosa.const
// CHECK: tosa.rescale
// CHECK: tosa.table
// CHECK: tosa.rescale
func @test_fakequant_sigmoid(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_fakequant_tanh
// CHECK: tosa.const
// CHECK: tosa.rescale
// CHECK: tosa.table
// CHECK: tosa.rescale
func @test_fakequant_tanh(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>> {
  %0 = "tfl.tanh"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>>
}

// -----

// CHECK-LABEL: test_fakequant_relu
// CHECK: tosa.rescale
// CHECK: tosa.reluN
// CHECK: tosa.rescale
func @test_fakequant_relu(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
}

// -----

// CHECK-LABEL: test_fakequant_relu6
// CHECK: tosa.rescale
// CHECK: tosa.reluN
// CHECK: tosa.rescale
func @test_fakequant_relu6(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>
}

// -----

// CHECK-LABEL: test_fakequant_leaky_relu
func @test_fakequant_leaky_relu(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015563514083623886:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015563514083623886:-1>> {
  %0 = "tfl.leaky_relu"(%arg0)  {alpha = 0.368738383 : f32}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015563514083623886:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015563514083623886:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015563514083623886:-1>>
}
