// RUN: tf-opt --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s

// Operations for testing tfl-to-tosa-pipeline

// TODO: For all quantized tests: compute and add checks on rescale attribute
// values
// TODO: These tests are fairly minimal. Expand the checks to be more robust.


// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.conv2d"(%arg0, %arg1, %[[VAR0]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv2d_bias
// CHECK: %[[VAR0:.*]] = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
func @test_conv2d_bias(%arg0: tensor<1x32x32x8xf32>, %cst: tensor<16x1x1x8xf32>, %cst_0: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose_conv2d"(%arg0, %arg1, %[[VAR0]]) {dilation = [1, 1], out_pad = [0, 0], out_shape = [1, 32, 32, 16], stride = [1, 1]}
func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = constant unit
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv2d_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<16x1x1x8xi8>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0> : tensor<16xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.conv2d"(%arg0, %[[VAR0]], %[[VAR1]]) {dilation = [1, 1], pad = [0, 0, 0, 0], quantization_info = {input_zp = 0 : i32, weight_zp = 0 : i32}, stride = [1, 1]}
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_conv2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<"0x851F811ED39B1160E8BFD11A44C8815EC054BEB7658131420857498B9B7FA28499818C7AB44894E64B81C6C350A581E8042F48DB13B85A81EEE481FD28A43BBBC381A70384A46F47811C2A4D64D8D285DEDCE37F1FFC6B5BB0A3794EED7F98D9060BA5ED5EC6A37F7FF4E67364062F078AE9DDDF778155794C54AE536D7FAC05"> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d_bias_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<1x1x1x16xi8>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[1, 2, 3, 0]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<16xi32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.transpose"(%[[VAR0]], %[[VAR1]])
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%[[VAR3]]) {new_shape = [1, 1, 8, 2]}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.depthwise_conv2d"(%arg0, %[[VAR4]], %[[VAR2]]) {dilation = [1, 1], pad = [0, 0, 0, 0], quantization_info = {input_zp = -1 : i32, weight_zp = 0 : i32}, stride = [1, 1]}
// CHECK: %[[VAR6:.*]] = "tosa.rescale"(%[[VAR5]])
func @test_depthwise_conv2d_bias_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x1x1x16xi8>} : () -> tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3,  {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5}>>, value = dense<[-2879, 6636, 3531, 23376, -79787, -6142, 5582, -30384, 17330, -4549, -3518, 16215, 2695, -2670, 8399, -12223]> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %0, %1) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>, tensor<1x1x1x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0,   {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_add
// CHECK: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: %[[VAR0:.*]] = "tosa.sub"(%arg0, %arg1)
func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32}
func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_exp
// CHECK: %[[VAR0:.*]] = "tosa.exp"(%arg0)
func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reciprocal"(%arg0)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR1]], %[[VAR2]]) {shift = 0 : i32}
func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.div"(%cst, %arg0)  {fused_activation_function = "NONE"}  : (tensor<f32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu
// CHECK: %[[VAR0:.*]] = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 0 : i64}
func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: %[[VAR0:.*]] = "tosa.reluN"(%arg0) {max_fp = 6.000000e+00 : f32, max_int = 0 : i64}
func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.707330704> : tensor<f32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.mul"(%arg0, %[[VAR2]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.greater_equal"(%arg0, %[[VAR4]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR5]], %arg0, %[[VAR3]])
func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.leaky_relu"(%arg0)  {alpha = 0.707330704 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: %[[VAR0:.*]] = "tosa.logical_and"(%arg0, %arg1)
func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: %[[VAR0:.*]] = "tosa.logical_or"(%arg0, %arg1)
func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: %[[VAR0:.*]] = "tosa.logical_not"(%arg0)
func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %0 = "tfl.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_any"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [21, 3]}
func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  return %0 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_min"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [21, 3]}
func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_min"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_max"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [21, 3]}
func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [21, 3]}
func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.0769230798> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [21, 3]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1]}
// CHECK: %[[VAR4:.*]] = "tosa.mul"(%[[VAR2]], %[[VAR3]]) {shift = 0 : i32}
func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.mean"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_prod"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [21, 3]}
func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_prod"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: %[[VAR0:.*]] = "tosa.minimum"(%arg0, %arg1)
func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: %[[VAR0:.*]] = "tosa.maximum"(%arg0, %arg1)
func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: %[[VAR0:.*]] = "tosa.pow"(%arg0, %arg1)
func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: %[[VAR0:.*]] = "tosa.abs"(%arg0)
func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: %[[VAR0:.*]] = "tosa.ceil"(%arg0)
func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: %[[VAR0:.*]] = "tosa.floor"(%arg0)
func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: %[[VAR0:.*]] = "tosa.log"(%arg0)
func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_negate
// CHECK: %[[VAR0:.*]] = "tosa.negate"(%arg0)
func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.neg"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK: %[[VAR0:.*]] = "tosa.rsqrt"(%arg0)
func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sigmoid
// CHECK: %[[VAR0:.*]] = "tosa.sigmoid"(%arg0)
func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32}
func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.square"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: %[[VAR0:.*]] = "tosa.equal"(%arg0, %arg1)
func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: %[[VAR0:.*]] = "tosa.greater_equal"(%arg0, %arg1)
func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: %[[VAR0:.*]] = "tosa.greater"(%arg0, %arg1)
func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK-DAG: %[[VAR0:.*]] = "tosa.greater_equal"(%arg0, %arg1)
// CHECK: %[[VAR1:.*]] = "tosa.logical_not"(%[[VAR0]])
func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK-DAG: %[[VAR0:.*]] = "tosa.greater"(%arg0, %arg1)
// CHECK: %[[VAR1:.*]] = "tosa.logical_not"(%[[VAR0]])
func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 819]}
func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %cst = constant dense<[1, 819]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %cst = constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = [4, 11, 1], start = [6, 8, 0]}
func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %cst = constant dense<[6, 8, 0]> : tensor<3xi32>
  %cst_0 = constant dense<[4, 11, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<4x11x1xf32>
  return %0 : tensor<4x11x1xf32>
}

// -----

// CHECK-LABEL: test_strided_slice
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = [9, 21, 2], start = [4, 0, 1]}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [9, 1, 7, 3, 2, 1]}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.slice"(%[[VAR1]]) {size = [9, 1, 7, 1, 2, 1], start = [0, 0, 0, 0, 0, 0]}
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [9, 7, 2]}
func @test_strided_slice(%arg0: tensor<13x21x3xf32>) -> tensor<9x7x2xf32> {
  %cst = constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<9x7x2xf32>
  return %0 : tensor<9x7x2xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<false> : tensor<1xi1>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK: %[[VAR2:.*]] = "tosa.select"(%[[VAR1]], %arg0, %arg1)
func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<false> : tensor<1xi1>
  %0 = "tfl.select_v2"(%cst, %arg0, %arg1) : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK-DAG: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.add"(%arg2, %[[VAR0]])
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg3, %[[VAR1]])
func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2, %arg3) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  return %0 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK-DAG: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [4, 13, 21, 3]}
func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %0 = "tfl.pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, values_count = 4 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  return %0 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = [1, 32, 32, 8], start = [0, 0, 0, 0]}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [32, 32, 8]}
func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32> {
  %0 = "tfl.unpack"(%arg0)  {axis = 0 : i32, num = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<32x32x8xf32>
  return %0 : tensor<32x32x8xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[\[}}1, 1], {{\[}}2, 2]]> : tensor<2x2xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.pad"(%arg0, %[[VAR0]])
func @test_pad(%arg0: tensor<2x3xf32>) -> tensor<4x7xf32> {
  %cst = constant dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<4x7xf32>
  return %0 : tensor<4x7xf32>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 13, 21, 3]}
func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<1x13x21x3xf32> {
  %cst = constant dense<[1, 13, 21, 3]> : tensor<4xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<4xi32>) -> tensor<1x13x21x3xf32>
  return %0 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[13, 21, 3]> : tensor<3xi32>}
func @test_shape() -> tensor<3xi32> {
  %cst = constant dense<[13, 21, 3]> : tensor<3xi32>
  return %cst : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<3> : tensor<i32>}
func @test_rank() -> tensor<i32> {
  %cst = constant dense<3> : tensor<i32>
  return %cst : tensor<i32>
}

// -----

// CHECK-LABEL: test_elu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.sub"(%[[VAR2]], %[[VAR3]])
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.greater_equal"(%arg0, %[[VAR5]])
// CHECK: %[[VAR7:.*]] = "tosa.select"(%[[VAR6]], %arg0, %[[VAR4]])
func @test_elu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.elu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
// CHECK: %[[VAR4:.*]] = "tosa.log"(%[[VAR3]])
func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log_softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<28xf32>}
// CHECK: %[[VAR2:.*]] = "tosa.transpose"(%arg1, %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.fully_connected"(%arg0, %[[VAR2]], %[[VAR1]])
func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %cst = constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = constant unit
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<19x28xf32>, tensor<2xi32>) -> tensor<28x19xf32>
  %1 = "tfl.fully_connected"(%arg0, %0, %cst_0)  {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}  : (tensor<14x19xf32>, tensor<28x19xf32>, none) -> tensor<14x28xf32>
  return %1 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg0, %[[VAR1]])
func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg1) {axis = 0 : i64}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [3]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [1, 1, 3]}
// CHECK: %[[VAR4:.*]] = "tosa.add"(%arg0, %[[VAR3]])
func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %cst = constant dense<[0, 1]> : tensor<2xi32>
  %0 = "tfl.sum"(%arg1, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3xf32>
  %1 = "tfl.add"(%arg0, %0)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<13x21x3xf32>
  return %1 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_split
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = [13, 7, 3], start = [0, 0, 0]}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = [13, 7, 3], start = [0, 7, 0]}
// CHECK: %[[VAR2:.*]] = "tosa.slice"(%arg0) {size = [13, 7, 3], start = [0, 14, 0]}
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
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[}}[0, 0], [0, 1], [0, 0]]> : tensor<3x2xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.pad"(%arg0, %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [13, 11, 2, 3]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = [26, 11, 3]}
func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %cst = constant dense<2> : tensor<1xi32>
  %cst_0 = constant dense<[[0, 1]]> : tensor<1x2xi32>
  %0 = "tfl.space_to_batch_nd"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  return %0 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[3, 1, 2, 0]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [2, 2, 2, 32, 32, 1]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = [2, 64, 64, 1]}
// CHECK: %[[VAR6:.*]] = "tosa.slice"(%[[VAR5]]) {size = [2, 64, 64, 1], start = [0, 0, 0, 0]}
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
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 16, 2, 16, 2, 8]}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [1, 16, 16, 32]}
func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %0 = "tfl.space_to_depth"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  return %0 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 32, 32, 2, 2, 2]}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [1, 64, 64, 2]}
func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %0 = "tfl.depth_to_space"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  return %0 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: test_one_hot
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.tile"(%[[VAR1]]) {multiples = [16, 1, 1]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%arg2) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.tile"(%[[VAR3]]) {multiples = [16, 2, 1]}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%arg0) {new_shape = [16, 1]}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.scatter"(%[[VAR4]], %[[VAR5]], %[[VAR2]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = [16, 1, 2]}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.transpose"(%[[VAR7]], %[[VAR0]])
// CHECK: %[[VAR9:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = [4, 4, 2]}
func @test_one_hot(%arg0: tensor<4x4xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<4x4x2xf32> {
  %0 = constant dense<2> : tensor<i32>
  %1 = "tfl.one_hot"(%arg0, %0, %arg1, %arg2) {axis = -1 : i32} : (tensor<4x4xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<4x4x2xf32>
  return %1 : tensor<4x4x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<16383.75> : tensor<f32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<6.10360876E-5> : tensor<f32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.mul"(%arg0, %[[VAR3]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.add"(%[[VAR4]], %[[VAR5]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.cast"(%[[VAR6]])
// CHECK-DAG: %[[VAR8:.*]] = "tosa.cast"(%[[VAR7]])
// CHECK-DAG: %[[VAR9:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR10:.*]] = "tosa.sub"(%[[VAR8]], %[[VAR9]])
// CHECK-DAG: %[[VAR11:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = [1, 1, 1]}
// CHECK: %[[VAR12:.*]] = "tosa.mul"(%[[VAR10]], %[[VAR11]]) {shift = 0 : i32}
func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.quantize"(%arg0)  {qtype = tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>}  : (tensor<13x21x3xf32>) -> tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>
  %1 = "tfl.dequantize"(%0) : (tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  %2 = "tfl.dequantize"(%0) : (tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  return %2 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.add"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_add_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
}

// -----

// CHECK-LABEL: test_sub_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.sub"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_sub_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
}

// -----

// CHECK-LABEL: test_mul_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR1]]) {shift = 0 : i32}
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_mul_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0078376950696110725>>
}

// -----

// CHECK-LABEL: test_avg_pool2d_qi8
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], quantization_info = {input_zp = -1 : i32, output_zp = -1 : i32}, stride = [1, 1]}
func @test_avg_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
  return %0 : tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
}

// -----

// CHECK-LABEL: test_max_pool2d_qi8
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]}
func @test_max_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
  return %0 : tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
}

// -----

// CHECK-LABEL: test_softmax_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<513xi16>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<513xi16>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() {value = dense<9> : tensor<i32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() {value = dense<7> : tensor<i32>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.const"() {value = dense<32768> : tensor<i32>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.const"() {value = dense<12> : tensor<i32>}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.const"() {value = dense<1> : tensor<i32>}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.const"() {value = dense<4> : tensor<i32>}
// CHECK-DAG: %[[VAR9:.*]] = "tosa.const"() {value = dense<536870912> : tensor<i32>}
// CHECK-DAG: %[[VAR10:.*]] = "tosa.const"() {value = dense<1515870810> : tensor<i32>}
// CHECK-DAG: %[[VAR11:.*]] = "tosa.const"() {value = dense<-1010580540> : tensor<i32>}
// CHECK-DAG: %[[VAR12:.*]] = "tosa.const"() {value = dense<35> : tensor<i32>}
// CHECK-DAG: %[[VAR13:.*]] = "tosa.rescale"(%arg0) {double_round = false, input_zp = 0 : i32, multiplier = [1073741824 : i32], output_zp = 0 : i32, per_channel = false, scale32 = true, shift = [30 : i32]}
// CHECK-DAG: %[[VAR14:.*]] = "tosa.reduce_max"(%[[VAR13]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR15:.*]] = "tosa.sub"(%[[VAR13]], %[[VAR14]])
// CHECK-DAG: %[[VAR16:.*]] = "tosa.rescale"(%[[VAR15]]) {double_round = false, input_zp = 0 : i32, multiplier = [1073741824 : i32], output_zp = 0 : i32, per_channel = false, scale32 = true, shift = [23 : i32]}
// CHECK-DAG: %[[VAR17:.*]] = "tosa.table"(%[[VAR16]], %[[VAR1]])
// CHECK-DAG: %[[VAR18:.*]] = "tosa.table"(%[[VAR16]], %[[VAR2]])
// CHECK-DAG: %[[VAR19:.*]] = "tosa.reshape"(%[[VAR3]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR20:.*]] = "tosa.logical_left_shift"(%[[VAR17]], %[[VAR19]])
// CHECK-DAG: %[[VAR21:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR22:.*]] = "tosa.arithmetic_right_shift"(%[[VAR18]], %[[VAR21]]) {round = true}
// CHECK-DAG: %[[VAR23:.*]] = "tosa.reshape"(%[[VAR5]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR24:.*]] = "tosa.add"(%[[VAR22]], %[[VAR23]])
// CHECK-DAG: %[[VAR25:.*]] = "tosa.add"(%[[VAR20]], %[[VAR24]])
// CHECK-DAG: %[[VAR26:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR27:.*]] = "tosa.arithmetic_right_shift"(%[[VAR25]], %[[VAR26]]) {round = true}
// CHECK-DAG: %[[VAR28:.*]] = "tosa.reduce_sum"(%[[VAR27]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR29:.*]] = "tosa.clz"(%[[VAR28]])
// CHECK-DAG: %[[VAR30:.*]] = "tosa.reshape"(%[[VAR7]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR31:.*]] = "tosa.sub"(%[[VAR29]], %[[VAR30]])
// CHECK-DAG: %[[VAR32:.*]] = "tosa.logical_left_shift"(%[[VAR28]], %[[VAR31]])
// CHECK-DAG: %[[VAR33:.*]] = "tosa.reshape"(%[[VAR11]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR34:.*]] = "tosa.mul"(%[[VAR32]], %[[VAR33]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR35:.*]] = "tosa.reshape"(%[[VAR10]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR36:.*]] = "tosa.add"(%[[VAR34]], %[[VAR35]])
// CHECK-DAG: %[[VAR37:.*]] = "tosa.mul"(%[[VAR36]], %[[VAR32]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR38:.*]] = "tosa.reshape"(%[[VAR9]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR39:.*]] = "tosa.sub"(%[[VAR38]], %[[VAR37]])
// CHECK-DAG: %[[VAR40:.*]] = "tosa.mul"(%[[VAR36]], %[[VAR39]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR41:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR42:.*]] = "tosa.mul"(%[[VAR40]], %[[VAR41]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR43:.*]] = "tosa.add"(%[[VAR36]], %[[VAR42]])
// CHECK-DAG: %[[VAR44:.*]] = "tosa.mul"(%[[VAR43]], %[[VAR32]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR45:.*]] = "tosa.reshape"(%[[VAR9]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR46:.*]] = "tosa.sub"(%[[VAR45]], %[[VAR44]])
// CHECK-DAG: %[[VAR47:.*]] = "tosa.mul"(%[[VAR43]], %[[VAR46]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR48:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR49:.*]] = "tosa.mul"(%[[VAR47]], %[[VAR48]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR50:.*]] = "tosa.add"(%[[VAR43]], %[[VAR49]])
// CHECK-DAG: %[[VAR51:.*]] = "tosa.mul"(%[[VAR50]], %[[VAR32]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR52:.*]] = "tosa.reshape"(%[[VAR9]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR53:.*]] = "tosa.sub"(%[[VAR52]], %[[VAR51]])
// CHECK-DAG: %[[VAR54:.*]] = "tosa.mul"(%[[VAR50]], %[[VAR53]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR55:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR56:.*]] = "tosa.mul"(%[[VAR54]], %[[VAR55]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR57:.*]] = "tosa.add"(%[[VAR50]], %[[VAR56]])
// CHECK-DAG: %[[VAR58:.*]] = "tosa.mul"(%[[VAR25]], %[[VAR57]]) {shift = 30 : i32}
// CHECK-DAG: %[[VAR59:.*]] = "tosa.reshape"(%[[VAR12]]) {new_shape = [1, 1, 1]}
// CHECK-DAG: %[[VAR60:.*]] = "tosa.sub"(%[[VAR59]], %[[VAR29]])
// CHECK-DAG: %[[VAR61:.*]] = "tosa.arithmetic_right_shift"(%[[VAR58]], %[[VAR60]]) {round = true}
// CHECK: %[[VAR62:.*]] = "tosa.rescale"(%[[VAR61]]) {double_round = false, input_zp = 0 : i32, multiplier = [1073741824 : i32], output_zp = -128 : i32, per_channel = false, scale32 = true, shift = [30 : i32]}
func @test_softmax_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.0156164625659585>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.0156164625659585>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_sigmoid_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<513xi16>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.table"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_sigmoid_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_tanh_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<513xi16>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.table"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func @test_tanh_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>> {
  %0 = "tfl.tanh"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 7.812500e-03>>
}

// -----

// CHECK-LABEL: test_relu_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reluN"(%[[VAR0]]) {max_fp = 0.000000e+00 : f32, max_int = 2147483647 : i64}
// CHECK: %[[VAR2:.*]] = "tosa.rescale"(%[[VAR1]])
func @test_relu_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
}

// -----

// CHECK-LABEL: test_relu6_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reluN"(%[[VAR0]]) {max_fp = 0.000000e+00 : f32, max_int = 384 : i64}
// CHECK: %[[VAR2:.*]] = "tosa.rescale"(%[[VAR1]])
func @test_relu6_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>
  return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>
}

// -----

// CHECK-LABEL: test_leaky_relu_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0> : tensor<i32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [1, 1]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%[[VAR1]], %[[VAR2]])
// CHECK-DAG: %[[VAR4:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR5:.*]] = "tosa.rescale"(%arg0)
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %[[VAR5]], %[[VAR4]])
func @test_leaky_relu_qi8(%arg0: tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>> {
  %0 = "tfl.leaky_relu"(%arg0) {alpha = 0.948724806 : f32} : (tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
  return %0 : tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.resize"(%arg0) {mode = "BILINEAR", offset = [-448, -448], offset_fp = [0.000000e+00 : f32, 0.000000e+00 : f32], output_size = [640, 640], shift = 10 : i32, stride = [128, 128], stride_fp = [0.000000e+00 : f32, 0.000000e+00 : f32]}
// CHECK: %[[VAR2:.*]] = "tosa.rescale"(%[[VAR1]]) {double_round = false, input_zp = 0 : i32, multiplier = [1073741824 : i32], output_zp = 0 : i32, per_channel = false, scale32 = true, shift = [50 : i32]}
func @test_resize_bilinear_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----
// CHECK-LABEL: test_gather_nd
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<6x7x1xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<1> : tensor<1xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%arg0) {new_shape = [1, 13, 63]}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = [42, 1]}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = [1, 1]}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.mul"(%[[VAR3]], %[[VAR4]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.reduce_sum"(%[[VAR5]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = [1, 42]}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.gather"(%[[VAR2]], %[[VAR7]])
// CHECK: %[[VAR9:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = [6, 7, 21, 3]}
func @test_gather_nd(%arg0: tensor<13x21x3xf32>) -> tensor<6x7x21x3xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<[[[0], [5], [3], [12], [2], [4], [3]], [[11], [1], [11], [10], [3], [12], [8]], [[5], [3], [1], [11], [3], [10], [0]], [[0], [8], [4], [7], [3], [12], [2]], [[7], [6], [11], [4], [2], [10], [11]], [[11], [1], [11], [1], [1], [11], [8]]]> : tensor<6x7x1xi32>} : () -> tensor<6x7x1xi32>
    %1 = "tfl.gather_nd"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<6x7x1xi32>) -> tensor<6x7x21x3xf32>
  return %1 : tensor<6x7x21x3xf32>
}
