// RUN: tf-opt --split-input-file --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s
// REQUIRES: tf_tosa
// RUN: tf-opt --split-input-file --tf-tfl-to-tosa-pipeline --verify-each %s | FileCheck %s
// REQUIRES: tf_tosa

// Operations for testing tfl-to-tosa-pipeline

// TODO: For all quantized tests: compute and add checks on rescale attribute
// values
// TODO: These tests are fairly minimal. Expand the checks to be more robust.


// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<16xf32>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_4:.*]] = tosa.conv2d %arg0, %arg1, %[[VAL_2]], %[[VAL_3]], %[[VAL_3]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
func.func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv2d_dynamic
// CHECK: tosa.conv2d
// CHECK-SAME: tensor<?x32x32x16xf32>
func.func @test_conv2d_dynamic(%arg0: tensor<?x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv2d_bias
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_4:.*]] = tosa.conv2d %arg0, %arg1, %arg2, %[[VAL_3]], %[[VAL_3]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
// CHECK-SAME: tensor<1x32x32x16xf32>
func.func @test_conv2d_bias(%arg0: tensor<1x32x32x8xf32>, %cst: tensor<16x2x2x8xf32>, %cst_0: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv2d_slicing
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[2, 31, 30, 8]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_6:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]]
// CHECK: %[[VAL_7:.*]] = tosa.conv2d %[[VAL_6]], %arg1, %arg2, %[[VAL_5]], %[[VAL_5]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>}
// CHECK-SAME: tensor<2x15x10x16xf32>
func.func @test_conv2d_slicing(%arg0: tensor<2x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>, %arg2: tensor<16xf32>) -> tensor<2x15x10x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 3 : i32}  : (tensor<2x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>) -> tensor<2x15x10x16xf32>
  func.return %0 : tensor<2x15x10x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAR2:.*]] = tosa.transpose_conv2d %arg0, %arg1, %[[VAR1]], %[[VAR1]], %[[VAR1]] {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = arith.constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d_relu
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAR2:.*]] = tosa.transpose_conv2d %arg0, %arg1, %[[VAR1]], %[[VAR1]], %[[VAR1]] {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR3:.*]] = tosa.clamp %[[VAR2]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32}
func.func @test_transpose_conv2d_relu(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = arith.constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU"}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv2d_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<16x2x2x8xi8>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0> : tensor<16xi32>}>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.conv2d %arg0, %[[VAR0]], %[[VAR1]], %[[VAR2]], %[[VAR2]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR4:.*]] = tosa.rescale %[[VAR3]]
func.func @test_conv2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<42> : tensor<16x2x2x8xi8>} : () -> tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_conv2d_qi8_2
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<16x2x2x8xi8>}>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1> : tensor<16xi32>}>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAL_6:.*]] = tosa.conv2d %arg0, %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_5]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
// CHECK: %[[VAL_7:.*]] = tosa.rescale %[[VAL_6]]
func.func @test_conv2d_qi8_2(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<42> : tensor<16x2x2x8xi8>} : () -> tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<1> : tensor<16xi8>} : () -> tensor<16x!quant.uniform<i8:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i8:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_conv2d_qi16
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<0> : tensor<16xi48>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<16x1x1x8xi8>}>
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}>
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR5:.*]] = tosa.conv2d %arg0, %[[VAR1]], %[[VAR0]], %[[VAR3]], %[[VAR4]] {acc_type = i48, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR6:.*]] = tosa.rescale %[[VAR5]]
func.func @test_conv2d_qi16(%arg0: tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "arith.constant"() {value = dense<0> : tensor<16xi64>} : () -> tensor<16xi64>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16xi64>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_depthwise_conv2d_bias_qi8
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>
// CHECK-DAG:     %[[shift:.*]] = "tosa.const"() <{values = dense<[36, 36, 36, 36, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]> : tensor<16xi8>}> : () -> tensor<16xi8>
// CHECK-DAG:     %[[multiplier:.*]] = "tosa.const"() <{values = dense<[1373724854, 1373724854, 1373724854, 1373724854, 1803013871, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854]> : tensor<16xi32>}> : () -> tensor<16xi32>
// CHECK-DAG:     %[[CONST0:.*]] = tosa.const_shape {values = dense<[2, 2, 8, 2]> : tensor<4xindex>}
// CHECK-DAG:     %[[CONST1:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<16xi32>}>
// CHECK-DAG:     %[[CONST2:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<1x2x2x16xi8>}>
// CHECK-DAG:     %[[INPUT_ZP:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}>
// CHECK-DAG:     %[[WEIGHT_ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG:     %[[RESHAPE:.*]] = tosa.reshape %[[CONST2]], %[[CONST0]]
// CHECK-DAG:     %[[DEPTHWISE:.*]] = tosa.depthwise_conv2d %[[ARG0]], %[[RESHAPE]], %[[CONST1]], %[[INPUT_ZP]], %[[WEIGHT_ZP]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
// CHECK-DAG:     %[[RESCALE:.*]] = tosa.rescale %[[DEPTHWISE]]
// CHECK:         return %[[RESCALE]]
func.func @test_depthwise_conv2d_bias_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127], [-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]], [[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127], [-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x2x2x16xi8>} : () -> tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3,  {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5}>>, value = dense<[-2879, 6636, 3531, 23376, -79787, -6142, 5582, -30384, 17330, -4549, -3518, 16215, 2695, -2670, 8399, -12223]> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %0, %1) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>, tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0,   {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: @test_conv2d_grouped_convolution
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 4, 1, 64]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[64, 1, 1, 64]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<64> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 64]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[64, 0, 0, 0]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[INPUT_SLICE_1:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_1:.*]] = tosa.slice %arg1, %[[VAL_4]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_1:.*]] = tosa.slice %arg2, %[[VAL_7]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_1:.*]] = tosa.conv2d %[[INPUT_SLICE_1]], %[[FILTER_SLICE_1]], %[[BIAS_SLICE_1]], %[[VAL_10]], %[[VAL_10]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK-DAG: %[[INPUT_SLICE_2:.*]] = tosa.slice %arg0, %[[VAL_8]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_2:.*]] = tosa.slice %arg1, %[[VAL_9]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_2:.*]] = tosa.slice %arg2, %[[VAL_6]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_2:.*]] = tosa.conv2d %[[INPUT_SLICE_2]], %[[FILTER_SLICE_2]], %[[BIAS_SLICE_2]], %[[VAL_10]], %[[VAL_10]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK-DAG: %[[CONCAT:.*]] = tosa.concat %[[CONV_1]], %[[CONV_2]] {axis = 3 : i32}
// CHECK: return %[[CONCAT]]
func.func @test_conv2d_grouped_convolution(%input: tensor<1x4x1x128xf32>, %weights: tensor<128x1x1x64xf32>, %bias: tensor<128xf32>) -> tensor<1x4x1x128xf32> {
  %0 = "tfl.conv_2d"(%input, %weights, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x4x1x128xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>)  -> (tensor<1x4x1x128xf32>)
  return %0 : tensor<1x4x1x128xf32>
}

// -----

// CHECK-LABEL: @test_conv2d_grouped_strided_convolution
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 3, 1, 16]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[128, 3, 1, 16]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<128> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 16]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[128, 0, 0, 0]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 32]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_11:.*]] = tosa.const_shape  {values = dense<[256, 0, 0, 0]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_12:.*]] = tosa.const_shape  {values = dense<256> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 48]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_14:.*]] = tosa.const_shape  {values = dense<[384, 0, 0, 0]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_15:.*]] = tosa.const_shape  {values = dense<384> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[INPUT_SLICE_1:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_1:.*]] = tosa.slice %arg1, %[[VAL_4]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_1:.*]] = tosa.slice %arg2, %[[VAL_7]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_1:.*]] = tosa.conv2d %[[INPUT_SLICE_1]], %[[FILTER_SLICE_1]], %[[BIAS_SLICE_1]], %[[VAL_16]], %[[VAL_16]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 0, 0>, stride = array<i64: 2, 1>}
// CHECK-DAG: %[[INPUT_SLICE_2:.*]] = tosa.slice %arg0, %[[VAL_8]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_2:.*]] = tosa.slice %arg1, %[[VAL_9]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_2:.*]] = tosa.slice %arg2, %[[VAL_6]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_2:.*]] = tosa.conv2d %[[INPUT_SLICE_2]], %[[FILTER_SLICE_2]], %[[BIAS_SLICE_2]], %[[VAL_16]], %[[VAL_16]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 0, 0>, stride = array<i64: 2, 1>}
// CHECK-DAG: %[[INPUT_SLICE_3:.*]] = tosa.slice %arg0, %[[VAL_10]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_3:.*]] = tosa.slice %arg1, %[[VAL_11]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_3:.*]] = tosa.slice %arg2, %[[VAL_12]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_3:.*]] = tosa.conv2d %[[INPUT_SLICE_3]], %[[FILTER_SLICE_3]], %[[BIAS_SLICE_3]], %[[VAL_16]], %[[VAL_16]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 0, 0>, stride = array<i64: 2, 1>}
// CHECK-DAG: %[[INPUT_SLICE_4:.*]] = tosa.slice %arg0, %[[VAL_13]], %[[VAL_3]]
// CHECK-DAG: %[[FILTER_SLICE_4:.*]] = tosa.slice %arg1, %[[VAL_14]], %[[VAL_5]]
// CHECK-DAG: %[[BIAS_SLICE_4:.*]] = tosa.slice %arg2, %[[VAL_15]], %[[VAL_6]]
// CHECK-DAG: %[[CONV_4:.*]] = tosa.conv2d %[[INPUT_SLICE_4]], %[[FILTER_SLICE_4]], %[[BIAS_SLICE_4]], %[[VAL_16]], %[[VAL_16]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 0, 0>, stride = array<i64: 2, 1>}
// CHECK-DAG: %[[CONCAT:.*]] = tosa.concat %[[CONV_1]], %[[CONV_2]], %[[CONV_3]], %[[CONV_4]] {axis = 3 : i32}
// CHECK: return %[[CONCAT]]
func.func @test_conv2d_grouped_strided_convolution(%input: tensor<1x3x1x64xf32>, %weights: tensor<512x3x1x16xf32>, %bias: tensor<512xf32>) -> tensor<1x2x1x512xf32> {
  %0 = "tfl.conv_2d"(%input, %weights, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 1 : i32} : (tensor<1x3x1x64xf32>, tensor<512x3x1x16xf32>, tensor<512xf32>)  -> (tensor<1x2x1x512xf32>)
  return %0 : tensor<1x2x1x512xf32>
}

// -----
// CHECK-LABEL: test_conv2d_q_grouped_convolution
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x1x16x!quant.uniform<i8:f32, 0.015684768557548523>>
// CHECK: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[8, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<36> : tensor<8xi8>}> : () -> tensor<8xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1374257539> : tensor<8xi32>}> : () -> tensor<8xi32>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<8> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK: %[[VAL_9:.*]] = tosa.const_shape  {values = dense<[8, 1, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<42> : tensor<16x1x1x8xi8>}> : () -> tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
// CHECK: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0> : tensor<16xi32>}> : () -> tensor<16x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,2.000000e+00,2.400000e+00,1.700000e+00,2.300000e+00,2.400000e+00,2.400000e+00,2.300000e+00,2.100000e+00,2.400000e+00,2.100000e+00,2.400000e+00}>>
// CHECK: %[[VAL_12:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_13:.*]] = tosa.const_shape  {values = dense<[1, 4, 1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_14:.*]] = tosa.slice %[[VAL_0]], %[[VAL_12]], %[[VAL_13]] : (tensor<1x4x1x16x!quant.uniform<i8:f32, 0.015684768557548523>>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x4x1x8x!quant.uniform<i8:f32, 0.015684768557548523>>
// CHECK: %[[VAL_15:.*]] = tosa.slice %[[VAL_10]], %[[VAL_12]], %[[VAL_9]] : (tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<8x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
// CHECK: %[[VAL_16:.*]] = tosa.slice %[[VAL_11]], %[[VAL_8]], %[[VAL_7]] : (tensor<16x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,2.000000e+00,2.400000e+00,1.700000e+00,2.300000e+00,2.400000e+00,2.400000e+00,2.300000e+00,2.100000e+00,2.400000e+00,2.100000e+00,2.400000e+00}>>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,2.000000e+00,2.400000e+00,1.700000e+00}>>
// CHECK: %[[VAL_17:.*]] = tosa.conv2d %[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[VAL_6]], %[[VAL_6]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x1x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<8x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>, tensor<8x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,2.000000e+00,2.400000e+00,1.700000e+00}>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x1x8xi32>
// CHECK: %[[VAL_18:.*]] = tosa.rescale %[[VAL_17]], %[[VAL_4]], %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x4x1x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x1x8x!quant.uniform<i8:f32, 0.078431375324726104>>
// CHECK: %[[VAL_19:.*]] = tosa.slice %[[VAL_0]], %[[VAL_2]], %[[VAL_13]] : (tensor<1x4x1x16x!quant.uniform<i8:f32, 0.015684768557548523>>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x4x1x8x!quant.uniform<i8:f32, 0.015684768557548523>>
// CHECK: %[[VAL_20:.*]] = tosa.slice %[[VAL_10]], %[[VAL_1]], %[[VAL_9]] : (tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<8x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
// CHECK: %[[VAL_21:.*]] = tosa.slice %[[VAL_11]], %[[VAL_7]], %[[VAL_7]] : (tensor<16x!quant.uniform<i32:f32:0, {2.000000e+00,2.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,2.000000e+00,2.400000e+00,1.700000e+00,2.300000e+00,2.400000e+00,2.400000e+00,2.300000e+00,2.100000e+00,2.400000e+00,2.100000e+00,2.400000e+00}>>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8x!quant.uniform<i32:f32:0, {2.300000e+00,2.400000e+00,2.400000e+00,2.300000e+00,2.100000e+00,2.400000e+00,2.100000e+00,2.400000e+00}>>
// CHECK: %[[VAL_22:.*]] = tosa.conv2d %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_6]], %[[VAL_6]] {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x1x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<8x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>, tensor<8x!quant.uniform<i32:f32:0, {2.300000e+00,2.400000e+00,2.400000e+00,2.300000e+00,2.100000e+00,2.400000e+00,2.100000e+00,2.400000e+00}>>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x1x8xi32>
// CHECK: %[[VAL_23:.*]] = tosa.rescale %[[VAL_22]], %[[VAL_4]], %[[VAL_3]], %[[VAL_5]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x4x1x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x1x8x!quant.uniform<i8:f32, 0.078431375324726104>>
// CHECK: %[[VAL_24:.*]] = tosa.concat %[[VAL_18]], %[[VAL_23]] {axis = 3 : i32} : (tensor<1x4x1x8x!quant.uniform<i8:f32, 0.078431375324726104>>, tensor<1x4x1x8x!quant.uniform<i8:f32, 0.078431375324726104>>) -> tensor<1x4x1x16x!quant.uniform<i8:f32, 0.078431375324726104>>
func.func @test_conv2d_q_grouped_convolution(%input: tensor<1x4x1x16x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x4x1x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%input, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x4x1x16x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x1x1x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x4x1x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  return %2 : tensor<1x4x1x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d_bias_inferred
func.func @test_depthwise_conv2d_bias_inferred(%arg0: tensor<?x32x32x8xf32>, %arg1 : tensor<1x1x1x16xf32>, %arg2 : tensor<16xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: tosa.depthwise_conv2d
  // CHECK-SAME: tensor<?x32x32x16xf32>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x32x32x8xf32>, tensor<1x1x1x16xf32>, tensor<16xf32>) -> tensor<?x?x?x?xf32>
  func.return %2 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_depthwise_conv2d_slicing
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[3, 3, 8, 2]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[1, 31, 31, 8]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_7:.*]] = tosa.reshape %arg1, %[[VAL_3]]
// CHECK: %[[VAL_8:.*]] = tosa.slice %arg0, %[[VAL_5]], %[[VAL_4]]
// CHECK: %[[VAL_9:.*]] = tosa.depthwise_conv2d %[[VAL_8]], %[[VAL_7]], %arg2, %[[VAL_6]], %[[VAL_6]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}
// CHECK-SAME: tensor<1x15x15x16xf32>
func.func @test_depthwise_conv2d_slicing(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<1x3x3x16xf32>, %arg2: tensor<16xf32>) -> tensor<1x15x15x16xf32> {
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2)  {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}  : (tensor<1x32x32x8xf32>, tensor<1x3x3x16xf32>, tensor<16xf32>) -> tensor<1x15x15x16xf32>
  func.return %0 : tensor<1x15x15x16xf32>
}

// -----

// CHECK-LABEL: test_conv3d
// CHECK-SAME: %[[VAL_0:.*]]: tensor<2x2x7x7x2xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x2x4xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 4, 0, 1, 2, 3>}
// CHECK: %[[VAL_6:.*]] = tosa.conv3d %[[VAL_0]], %[[VAL_5]], %[[VAL_4]], %[[VAL_4]], %[[VAL_4]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d(%arg0: tensor<2x2x7x7x2xf32>, %arg1: tensor<2x3x3x2x4xf32>) -> tensor<2x2x7x7x4xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<2x2x7x7x2xf32>, tensor<2x3x3x2x4xf32>, none) -> tensor<2x2x7x7x4xf32>
  func.return %0 : tensor<2x2x7x7x4xf32>
}

// -----

// CHECK-LABEL: test_conv3d_dynamic
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x11x32x32x8xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<3x1x1x8x16xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 4, 0, 1, 2, 3>}
// CHECK: %[[VAL_6:.*]] = tosa.conv3d %[[VAL_0]], %[[VAL_5]], %[[VAL_4]], %[[VAL_4]], %[[VAL_4]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 1, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d_dynamic(%arg0: tensor<?x11x32x32x8xf32>, %arg1: tensor<3x1x1x8x16xf32>) -> tensor<*xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x11x32x32x8xf32>, tensor<3x1x1x8x16xf32>, none) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv3d_bias
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10x3x64x64x12xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<16x2x2x12x8xf32>
// CHECK-SAME: %[[VAL_2:.*]]: tensor<8xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_5:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 4, 0, 1, 2, 3>}
// CHECK: %[[VAL_6:.*]] = tosa.conv3d %[[VAL_0]], %[[VAL_5]], %[[VAL_2]], %[[VAL_4]], %[[VAL_4]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 7, 8, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d_bias(%arg0: tensor<10x3x64x64x12xf32>, %arg1: tensor<16x2x2x12x8xf32>, %cst: tensor<8xf32>) -> tensor<10x3x64x64x8xf32> {
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<10x3x64x64x12xf32>, tensor<16x2x2x12x8xf32>, tensor<8xf32>) -> tensor<10x3x64x64x8xf32>
  func.return %0 : tensor<10x3x64x64x8xf32>
}

// -----

// CHECK-LABEL: test_conv3d_slicing
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 31, 31, 31, 8]> : tensor<5xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<5xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]]
// CHECK: %[[VAL_8:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 4, 0, 1, 2, 3>}
// CHECK: %[[VAL_9:.*]] = tosa.conv3d %[[VAL_7]], %[[VAL_8]], %arg2, %[[VAL_6]], %[[VAL_6]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 2, 2, 2>}
func.func @test_conv3d_slicing(%arg0: tensor<1x32x32x32x8xf32>, %arg1: tensor<3x3x3x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<1x15x15x15x16xf32> {
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x32x32x32x8xf32>, tensor<3x3x3x8x16xf32>, tensor<16xf32>) -> tensor<1x15x15x15x16xf32>
  func.return %0 : tensor<1x15x15x15x16xf32>
}

// -----

// CHECK-LABEL: test_conv3d_qi8(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x17x34xf32>) -> tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.0156862643> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1.11982894> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<-4> : tensor<1x1x1x1x1xi32>}
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[BIAS_ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAL_8:.*]] = tosa.cast %[[VAL_0]]
// CHECK: %[[VAL_10:.*]] = tosa.mul %[[VAL_8]], %[[VAL_3]], %[[SHIFT]]
// CHECK: %[[VAL_11:.*]] = tosa.transpose %[[VAL_1]] {perms = array<i32: 4, 0, 1, 2, 3>}
// CHECK: %[[VAL_12:.*]] = tosa.conv3d %[[VAL_10]], %[[VAL_11]], %[[BIAS_ZP]], %[[BIAS_ZP]], %[[BIAS_ZP]] {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 2>}
// CHECK: %[[VAL_13:.*]] = tosa.mul %[[VAL_12]], %[[VAL_4]], %[[SHIFT]]
// CHECK: %[[VAL_14:.*]] = tosa.cast %[[VAL_13]]
// CHECK: %[[VAL_15:.*]] = tosa.add %[[VAL_14]], %[[VAL_5]]
// CHECK: %[[VAL_16:.*]] = tosa.cast %[[VAL_15]]
// CHECK: return %[[VAL_16]]
func.func @test_conv3d_qi8(%arg0: tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>, %arg1: tensor<2x3x3x17x34xf32>) -> (tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>) -> tensor<1x4x8x21x17xf32>
  %2 = "tfl.no_value"() {value} : () -> none
  %3 = "tfl.conv_3d"(%0, %arg1, %2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 2 : i32} : (tensor<1x4x8x21x17xf32>, tensor<2x3x3x17x34xf32>, none) -> tensor<1x4x8x11x34xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>} : (tensor<1x4x8x11x34xf32>) -> tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
  return %4 : tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
}

// -----

// CHECK-LABEL: test_conv3d_qi16
// CHECK: %[[BIAS:.+]] = "tosa.const"() <{values = dense<123> : tensor<16xi48>}> : () -> tensor<16xi48>
// CHECK: tosa.conv3d {{.+}}, %[[BIAS]], %{{.+}} {acc_type = i48, {{.+}}} : {{.+}} -> tensor<1x15x15x15x16xi48>
func.func @test_conv3d_qi16(%input: tensor<1x32x32x32x8x!quant.uniform<i16:f32, 1.0>>, %filter: tensor<3x3x3x8x16x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x15x15x15x16x!quant.uniform<i16:f32, 1.0>> {
  %bias = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i16:f32, 1.0>>, value = dense<123> : tensor<16xi16>} : () -> tensor<16x!quant.uniform<i16:f32, 1.0>>
  %0 = "tfl.conv_3d"(%input, %filter, %bias) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_d = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x32x32x32x8x!quant.uniform<i16:f32, 1.0>>, tensor<3x3x3x8x16x!quant.uniform<i8:f32, 1.0>>, tensor<16x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x15x15x15x16x!quant.uniform<i16:f32, 1.0>>
  func.return %0 : tensor<1x15x15x15x16x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: test_add
// CHECK: %[[VAR0:.*]] = tosa.add %arg0, %arg1
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_unranked
// CHECK: %[[VAR0:.*]] = tosa.add %arg0, %arg1
func.func @test_add_unranked(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: %[[VAR0:.*]] = tosa.sub %arg0, %arg1
func.func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sub_unranked
// CHECK: %[[VAR0:.*]] = tosa.sub %arg0, %arg1
func.func @test_sub_unranked(%arg0: tensor<1x21x3xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<1x1x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR0:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul_unranked
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR0:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
func.func @test_mul_unranked(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_exp
// CHECK: %[[VAR0:.*]] = tosa.exp %arg0
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_exp_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<{{.+}}> : tensor<256xi8>}>
// CHECK: %[[VAL_2:.*]] = tosa.table %[[VAL_0]], %[[VAL_1]]
func.func @test_exp_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>) -> (tensor<13x21x3x!quant.uniform<i8:f32, 0.028976691886782646:128>>) {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.028976691886782646:128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.028976691886782646:128>>
}

// -----

// CHECK-LABEL: test_exp_qi16
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<{{.+}}> : tensor<513xi16>}>
// CHECK: %[[VAL_2:.*]] = tosa.table %[[VAL_0]], %[[VAL_1]]
func.func @test_exp_qi16(%arg0: tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>) -> (tensor<13x21x3x!quant.uniform<i16:f32, 2.2550298308487982E-4>>) {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>) -> tensor<13x21x3x!quant.uniform<i16:f32, 2.2550298308487982E-4>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i16:f32, 2.2550298308487982E-4>>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK-DAG: %[[VAR0:.*]] = tosa.reciprocal %arg0
func.func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.div"(%cst, %arg0)  {fused_activation_function = "NONE"}  : (tensor<f32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_div
// CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape %arg1
// CHECK: %[[VAR0:.*]] = tosa.int_div %arg0, %[[RESHAPE]]
func.func @test_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "tfl.div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL:   func.func @test_floor_div(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<13x21x3xi32>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<i32>) -> tensor<13x21x3xi32> {
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
// CHECK:           %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
// CHECK:           %[[VAL_5:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           %[[VAL_6:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_5]] : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
// CHECK:           %[[VAL_7:.*]] = tosa.int_div %[[VAL_0]], %[[VAL_6]] : (tensor<13x21x3xi32>, tensor<1x1x1xi32>) -> tensor<13x21x3xi32>
// CHECK:           %[[VAL_8:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_5]] : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
// CHECK:           %[[VAL_9:.*]] = tosa.mul %[[VAL_0]], %[[VAL_8]], %[[VAL_2]] : (tensor<13x21x3xi32>, tensor<1x1x1xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
// CHECK:           %[[VAL_10:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_5]] : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
// CHECK:           %[[VAL_11:.*]] = tosa.mul %[[VAL_10]], %[[VAL_7]], %[[VAL_2]] : (tensor<1x1x1xi32>, tensor<13x21x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
// CHECK:           %[[VAL_12:.*]] = tosa.equal %[[VAL_0]], %[[VAL_11]] : (tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi1>
// CHECK:           %[[VAL_13:.*]] = tosa.logical_not %[[VAL_12]] : (tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
// CHECK:           %[[VAL_14:.*]] = tosa.greater %[[VAL_3]], %[[VAL_9]] : (tensor<1x1x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi1>
// CHECK:           %[[VAL_15:.*]] = tosa.sub %[[VAL_7]], %[[VAL_4]] : (tensor<13x21x3xi32>, tensor<1x1x1xi32>) -> tensor<13x21x3xi32>
// CHECK:           %[[VAL_16:.*]] = tosa.logical_and %[[VAL_13]], %[[VAL_14]] : (tensor<13x21x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
// CHECK:           %[[VAL_17:.*]] = tosa.select %[[VAL_16]], %[[VAL_15]], %[[VAL_7]] : (tensor<13x21x3xi1>, tensor<13x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
// CHECK:           return %[[VAL_17]] : tensor<13x21x3xi32>
// CHECK:         }
func.func @test_floor_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<i32>) -> tensor<13x21x3xi32> {
  %0 = "tfl.floor_div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<13x21x3xi32>
  func.return %0 : tensor<13x21x3xi32>
}

// -----

// CHECK-LABEL: test_relu1
// CHECK: %[[VAL0:.*]] = tosa.clamp %arg0 {max_val = 1.000000e+00 : f32, min_val = -1.000000e+00 : f32}
func.func @test_relu1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu_n1_to_1"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu0To1
// CHECK: %[[VAL0:.*]] = tosa.clamp %arg0 {max_val = 1.000000e+00 : f32, min_val = 0.000000e+00 : f32}
func.func @test_relu0To1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu_0_to_1"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: %[[VAR0:.*]] = tosa.clamp %arg0 {max_val = 6.000000e+00 : f32, min_val = 0.000000e+00 : f32}
func.func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_relu6_dynamic
// CHECK: %[[VAR0:.*]] = tosa.clamp %arg0 {max_val = 6.000000e+00 : f32, min_val = 0.000000e+00 : f32}
// CHECK-SAME: -> tensor<?x21x3xf32>
func.func @test_relu6_dynamic(%arg0: tensor<?x21x3xf32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<?x21x3xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<0.707330704> : tensor<1x1x1xf32>}>
// CHECK: %[[VAR1:.*]] = tosa.mul %arg0, %[[VAR0]], %[[SHIFT]]
// CHECK: %[[VAR2:.*]] = tosa.maximum %[[VAR1]], %arg0 : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAR2]] : tensor<13x21x3xf32>
func.func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.leaky_relu"(%arg0)  {alpha = 0.707330704 : f32}  : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_prelu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 2, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg1, %[[VAR10]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.mul %arg0, %[[VAR1]], %[[SHIFT]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.greater_equal %arg0, %[[VAR0]]
// CHECK: %[[VAR4:.*]] = tosa.select %[[VAR3]], %arg0, %[[VAR2]]
func.func @test_prelu(%arg0: tensor<4x2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x2x3xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<4x2x3xf32>, tensor<2x3xf32>) -> tensor<4x2x3xf32>
  func.return %0 : tensor<4x2x3xf32>
}

// -----

// CHECK-LABEL: test_prelu_qu8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x4x17x!quant.uniform<u8:f32, 0.015686038881540298:128>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1472433039> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<37> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1130006236> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 8, 4, 17]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<"0x1{{.*}}"> : tensor<8x4x17xi8>}> : () -> tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.023982547223567963>>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<5> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<-123> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_12:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_13:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_15:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<u8:f32, 0.015686038881540298:128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>
// CHECK: %[[VAL_16:.*]] = tosa.rescale %[[VAL_15]], %[[VAL_11]], %[[VAL_12]], %[[VAL_14]], %[[VAL_14]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>
// CHECK: %[[VAL_17:.*]] = tosa.rescale %[[VAL_16]], %[[VAL_11]], %[[VAL_12]], %[[VAL_14]], %[[VAL_7]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_18:.*]] = tosa.greater_equal %[[VAL_17]], %[[VAL_6]] : (tensor<1x8x4x17xi32>, tensor<1x1x1x1xi32>) -> tensor<1x8x4x17xi1>
// CHECK: %[[VAL_19:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_11]], %[[VAL_12]], %[[VAL_14]], %[[VAL_7]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.023982547223567963>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<8x4x17xi32>
// CHECK: %[[VAL_20:.*]] = tosa.reshape %[[VAL_19]], %[[VAL_5]] : (tensor<8x4x17xi32>, !tosa.shape<4>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_21:.*]] = tosa.mul %[[VAL_17]], %[[VAL_20]], %[[VAL_14]] : (tensor<1x8x4x17xi32>, tensor<1x8x4x17xi32>, tensor<1xi8>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_22:.*]] = tosa.rescale %[[VAL_21]], %[[VAL_4]], %[[VAL_3]], %[[VAL_7]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>
// CHECK: %[[VAL_23:.*]] = tosa.rescale %[[VAL_16]], %[[VAL_2]], %[[VAL_1]], %[[VAL_14]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>
// CHECK: %[[VAL_24:.*]] = tosa.select %[[VAL_18]], %[[VAL_23]], %[[VAL_22]] : (tensor<1x8x4x17xi1>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>
// CHECK: %[[VAL_25:.*]] = tosa.rescale %[[VAL_24]], %[[VAL_11]], %[[VAL_12]], %[[VAL_9]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>
// CHECK: %[[VAL_26:.*]] = tosa.rescale %[[VAL_25]], %[[VAL_11]], %[[VAL_12]], %[[VAL_9]], %[[VAL_10]] {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<u8:f32, 0.045754898339509964:133>>
func.func @test_prelu_qu8(%arg0: tensor<1x8x4x17x!quant.uniform<u8:f32, 0.015686038881540298:128>>) -> tensor<1x8x4x17x!quant.uniform<u8:f32, 0.045754898339509964:133>> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>} : (tensor<1x8x4x17x!quant.uniform<u8:f32, 0.015686038881540298:128>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.023982547223567963>>, value = dense<"0x191D0557FF212FA1137FDE2B247CE8BA2A8B2213F6B109FA12232EC613FEEE03EF2D265BE5E4F6CB0E09F7F0A95606DA1709EDE632D0F92A2002E98E61F9213997D3FCEBFA0D2DFC4DD00D0700C60C0705F3CFCB01D30C3617C7144C294DAE27061A62E70665021AF50827F40EC9E0172D42B9FB01FB076A09553006F7F710211A031EC9F11BCF130FCC1906D5FED8E5F64E06EAEAFEFD2515F20BB6E3401023C89DFCF8DEC0390B37D8CA2001E1F7BC270ADDE92DFC6D230CE1FEEE1DE8F90ABF9E3ECAEEBC311DF6FDE41F0E31ED0AC309B3121533E7EC2D1B0F1E04D44513E627F4ED5E491D10E53EEA45FF23E31D11D1DE2E0A3B1015AF06102329DEED5C1C180402000B0D071BF0D4FBC0DE0C3BF012E018D80716351D1922F8D508CF2708BA0CEAFE14E4972732FDFD283ED9342A1506F4F137200A12F436D6C9EC071FBCBDEBF4F8051426B8201EC410F9C3C7EFF7CD04D7AC34E2F9D73A5A05CFFA0FF7FD21D6BBEA03F16AF8330C1105285605C9FFE72BE04726DA06F2DCDCDC14C1310CF4E32F06BE0941420B10C9293DD10EFE28D4D20716E6E6EE0A101FFE3AAF1716120EF62FECEBC0F0D72A0903F9E74425EDF82E290E0413BB69F3F45AF30A22D4D024411B4D243BE13FB9CBE0F5FA16A1D7532007AEF62837C42406E3ED3CCE0408CA1C0CFA18B40C0BF7261E06D3E504B8E714BCF6F010DB12373739E200E609E9DAEF1922A2C338FEF2C519F0E5101E2AE917DCA3FA27D245DD10F0EBCE"> : tensor<8x4x17xi8>} : () -> tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.023982547223567963>>
  %2 = "tfl.prelu"(%0, %1) : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.023982547223567963>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x8x4x17x!quant.uniform<u8:f32, 0.045754898339509964:133>>} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.045754898339509964:5>>) -> tensor<1x8x4x17x!quant.uniform<u8:f32, 0.045754898339509964:133>>
  func.return %3 : tensor<1x8x4x17x!quant.uniform<u8:f32, 0.045754898339509964:133>>
}

// -----

// CHECK-LABEL:   func.func @test_prelu_qi8(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1582183328> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<37> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1103996759> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 8, 4, 17]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<"0xD{{.*}}"> : tensor<8x4x17xi8>}> : () -> tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.021805247291922569>>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_12:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_13:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_14:.*]] = tosa.greater_equal %[[VAL_13]], %[[VAL_7]] : (tensor<1x8x4x17xi32>, tensor<1x1x1x1xi32>) -> tensor<1x8x4x17xi1>
// CHECK: %[[VAL_15:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.021805247291922569>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<8x4x17xi32>
// CHECK: %[[VAL_16:.*]] = tosa.reshape %[[VAL_15]], %[[VAL_6]] : (tensor<8x4x17xi32>, !tosa.shape<4>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_17:.*]] = tosa.mul %[[VAL_13]], %[[VAL_16]], %[[VAL_11]] : (tensor<1x8x4x17xi32>, tensor<1x8x4x17xi32>, tensor<1xi8>) -> tensor<1x8x4x17xi32>
// CHECK: %[[VAL_18:.*]] = tosa.rescale %[[VAL_17]], %[[VAL_5]], %[[VAL_4]], %[[VAL_12]], %[[VAL_3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>
// CHECK: %[[VAL_19:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_2]], %[[VAL_1]], %[[VAL_11]], %[[VAL_3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>
// CHECK: %[[VAL_20:.*]] = tosa.select %[[VAL_14]], %[[VAL_19]], %[[VAL_18]] : (tensor<1x8x4x17xi1>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>
func.func @test_prelu_qi8(%arg0: tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.021805247291922569>>, value = dense<"0xDAFDEBC120CBE1E028231F05CF04F52484B2F0AC0041E618200308F820FE308FFCF2E1E02A06D00606FB1044C928D8D811E3FCCE350E25C4DE2B0D00E20AC1E215940D0D12C809290D480FE9E2DB26E31E50F5F4FDD31EFF21C210E717E187144F27C848E820C5D503E31729218D96D2D6D3D9C43BF13014EFCB043631AE4403FE2D4CDF1F16E2D13BA20AE92CEAB7323405F728CF3DF4E9BBFAFEFEE120ECA7FA120609030FF0FCF0E5D40939172EE7E256BADEC5ECFFB32C35F4E936E2F8092FE2E3EFE22B0C02F5EE1D36DE03CBE02FF346081C30ED882AECCAF4E4E3361604EABF133CB6371DDAFCDA4F2D32034A270BF0120A0048131331E50D11CAEB1DEE0ADFC0F12531E8351DD7BDEB2821FF3ECC34F8D42EE4D6FF2AE5FEEDFC3DF7463CED10192CE4B728151827A92E000EE31CF3C5DF193DAC2836181BD916D339E914192B14F0163C58C500BDC6BAEFFB03EC33DA24E7FF0E292CE30504B3070AB5FDE6D7E7CB4CB0D818F90919EAEF5DFDF2DB6C4132DF8EF2E40AF7EA04F1D496F22F2971420FF01D012E2954D5081C0AF2C5E5DED2CCD8C6157416201AFF3A2B29FBDD9EF06340B021F45C322A202DDD86111EBDF44BE9110E29F3FE7FDEDDFB5FDEDBD933E2ED0DD4E21C4BC6FD28E31934C821CE10F61C12740A100F1BE205CC01434BD7E3FB14F01CE0E406710022E464E0F0D8FB3D01C733C9C94017FAC50BE812D202E2B10C04E70AF326CEFD0DE20ABD153D3D14171C34061DE5FC5A"> : tensor<8x4x17xi8>} : () -> tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.021805247291922569>>
  %1 = "tfl.prelu"(%arg0, %0) : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.021805247291922569>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>
  func.return %1 : tensor<1x8x4x17x!quant.uniform<i8:f32, 0.042581047862768173:1>>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: %[[VAR0:.*]] = tosa.logical_and %arg0, %arg1
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<*xi1> {
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: %[[VAR0:.*]] = tosa.logical_or %arg0, %arg1
func.func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<*xi1> {
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: %[[VAR0:.*]] = tosa.logical_not %arg0
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %0 = "tfl.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  func.return %0 : tensor<1x21x3xi1>
}

// -----

// CHECK: @test_reduce_sum_axis_out_of_bounds
func.func @test_reduce_sum_axis_out_of_bounds(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<1094795585> : tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_1_keep_true
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = tosa.reduce_all %[[VAL_0]] {axis = 1 : i32} : (tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1>
func.func @test_reduce_all_axis_1_keep_true(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = true}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x1x8x19xi1>
  func.return %0 : tensor<1x1x8x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_1_keep_false
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK-DAG: %[[VAL_10:.+]] = tosa.const_shape {values = dense<[1, 8, 19]> : tensor<3xindex>}
// CHECK: %[[VAL_1:.*]] = tosa.reduce_all %[[VAL_0]] {axis = 1 : i32} : (tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1>
// CHECK: %[[VAL_2:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_10]] : (tensor<1x1x8x19xi1>, !tosa.shape<3>) -> tensor<1x8x19xi1>
func.func @test_reduce_all_axis_1_keep_false(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x8x19xi1> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = false}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x8x19xi1>
  func.return %0 : tensor<1x8x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_2_keep_true
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = tosa.reduce_all %[[VAL_0]] {axis = 2 : i32} : (tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1>
func.func @test_reduce_all_axis_2_keep_true(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = true}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x4x1x19xi1>
  func.return %0 : tensor<1x4x1x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_2_keep_false
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 4, 19]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_1:.*]] = tosa.reduce_all %[[VAL_0]] {axis = 2 : i32} : (tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1>
// CHECK: %[[VAL_2:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_10]] : (tensor<1x4x1x19xi1>, !tosa.shape<3>) -> tensor<1x4x19xi1>
func.func @test_reduce_all_axis_2_keep_false(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x4x19xi1> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = false}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x4x19xi1>
  func.return %0 : tensor<1x4x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_any %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  func.return %0 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_min %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_min"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_max %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_sum %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum_nonzero_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10x20x30x40x50xf32>
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape {values = dense<[10, 20, 30, 50]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 3 : i32} : (tensor<10x20x30x40x50xf32>) -> tensor<10x20x30x1x50xf32>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_1]] : (tensor<10x20x30x1x50xf32>, !tosa.shape<4>) -> tensor<10x20x30x50xf32>
// CHECK: return %[[VAL_3]] : tensor<10x20x30x50xf32>
func.func @test_reduce_sum_nonzero_axis(%arg0: tensor<10x20x30x40x50xf32> {tf._user_specified_name = "inp_list"}) -> tensor<10x20x30x50xf32> {
  %cst = arith.constant dense<3> : tensor<i32>
  %0 = "tfl.sum"(%arg0, %cst) {device = "", keep_dims = false} : (tensor<10x20x30x40x50xf32>, tensor<i32>) -> tensor<10x20x30x50xf32>
  func.return %0 : tensor<10x20x30x50xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum_5D
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape {values = dense<[6, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<4x5x6x7x8xf32>) -> tensor<1x5x6x7x8xf32>
// CHECK-DAG: %[[VAR2:.*]] = tosa.reduce_sum %[[VAR1]] {axis = 1 : i32} : (tensor<1x5x6x7x8xf32>) -> tensor<1x1x6x7x8xf32>
// CHECK-DAG: %[[VAR3:.*]] = tosa.reduce_sum %[[VAR2]] {axis = 3 : i32} : (tensor<1x1x6x7x8xf32>) -> tensor<1x1x6x1x8xf32>
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %[[VAR3]], %[[VAR0]] : (tensor<1x1x6x1x8xf32>, !tosa.shape<2>) -> tensor<6x8xf32>
// CHECK: return %[[VAR4]]
func.func @test_reduce_sum_5D(%arg0: tensor<4x5x6x7x8xf32>) -> tensor<6x8xf32> {
  %cst = arith.constant dense<[0, 1, 3]> : tensor<3xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<4x5x6x7x8xf32>, tensor<3xi32>) -> tensor<6x8xf32>
  func.return %0 : tensor<6x8xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<0.0769230798> : tensor<1x1xf32>}>
// CHECK-DAG: %[[VAR1:.*]] = tosa.reduce_sum %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK-DAG: %[[VAR2:.*]] = tosa.reshape %[[VAR1]], %[[VAR10]]
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR4:.*]] = tosa.mul %[[VAR2]], %[[VAR0]], %[[SHIFT]]
func.func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.mean"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean_out_of_bounds
// CHECK: "tfl.mean"
func.func @test_reduce_mean_out_of_bounds(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<123> : tensor<1xi32>
  %0 = "tfl.mean"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x2x2x!quant.uniform<i8:f32, 0.0039208820089697838:-128>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1105078632> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_7:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x2x2x!quant.uniform<i8:f32, 0.0039208820089697838:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x2x2xi32>
// CHECK: %[[VAL_8:.*]] = tosa.reduce_sum %[[VAL_7]] {axis = 2 : i32} : (tensor<1x2x2xi32>) -> tensor<1x2x1xi32>
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_2]], %[[VAL_1]], %[[VAL_6]], %[[VAL_5]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x2x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x1x!quant.uniform<i8:f32, 0.0038096972275525331:-128>>
func.func @test_reduce_mean_qi8(%arg0: tensor<1x2x2x!quant.uniform<i8:f32, 0.0039208820089697838:-128>>) -> (tensor<1x2x1x!quant.uniform<i8:f32, 0.0038096972275525331:-128>>) {
%0 = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
%1 = "tfl.mean"(%arg0, %0) {keep_dims = true} : (tensor<1x2x2x!quant.uniform<i8:f32, 0.0039208820089697838:-128>>, tensor<i32>) -> tensor<1x2x1x!quant.uniform<i8:f32, 0.0038096972275525331:-128>>
return %1 : tensor<1x2x1x!quant.uniform<i8:f32, 0.0038096972275525331:-128>>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_product %arg0 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[21, 3]> : tensor<2xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_prod"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: %[[VAR0:.*]] = tosa.minimum %arg0, %arg1
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: %[[VAR0:.*]] = tosa.maximum %arg0, %arg1
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_dynamic
// CHECK: %[[VAR0:.*]] = tosa.maximum %arg0, %arg1
// CHECK-SAME: -> tensor<13x21x?xf32>
func.func @test_max_dynamic(%arg0: tensor<13x1x?xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x1x?xf32>, tensor<13x21x1xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: %[[VAR0:.*]] = tosa.pow %arg0, %arg1
func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow_dynamic
// CHECK: %[[VAR0:.*]] = tosa.pow %arg0, %arg1
// CHECK-SAME: -> tensor<13x21x3xf32>
func.func @test_pow_dynamic(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: %[[VAR0:.*]] = tosa.abs %arg0
func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: %[[VAR0:.*]] = tosa.ceil %arg0
func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: %[[VAR0:.*]] = tosa.floor %arg0
func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: %[[VAR0:.*]] = tosa.log %arg0
func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_log_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{.+}}> : tensor<256xi8>}>
// CHECK: %[[VAL_2:.*]] = tosa.table %[[VAL_0]], %[[VAL_1]]
func.func @test_log_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>) -> (tensor<13x21x3x!quant.uniform<i8:f32, 0.0027182241901755333:128>>) {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.011764706112444401:43>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0027182241901755333:128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0027182241901755333:128>>
}

// -----

// CHECK-LABEL: test_log_qi16
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{.+}}> : tensor<513xi16>}>
// CHECK: %[[VAL_2:.*]] = tosa.table %[[VAL_0]], %[[VAL_1]]
func.func @test_log_qi16(%arg0: tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>) -> (tensor<13x21x3x!quant.uniform<i16:f32, 2.1153819034225307E-5>>) {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3x!quant.uniform<i16:f32, 6.1037018895149231E-5>>) -> tensor<13x21x3x!quant.uniform<i16:f32, 2.1153819034225307E-5>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i16:f32, 2.1153819034225307E-5>>
}

// -----

// CHECK-LABEL: test_negate
// CHECK-DAG: %[[CONST_0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAR0:.*]] = tosa.negate %arg0, %[[CONST_0]], %[[CONST_0]]
func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.neg"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3xf32>
// CHECK: %[[VAL_1:.*]] = tosa.rsqrt %[[VAL_0]] : (tensor<13x21x3xf32>
func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_rsqrt_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 1.500000e-02:-128>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<{{.+}}> : tensor<256xi8>}>
// CHECK: %[[VAL_2:.*]] = tosa.table %[[VAL_0]], %[[VAL_1]]
func.func @test_rsqrt_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015:-128>>) -> (tensor<13x21x3x!quant.uniform<i8:f32, 3.71:-128>>) {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015:-128>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.71:-128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.71:-128>>
}

// -----

// CHECK-LABEL: test_sign
// CHECK-SAME: %[[VAL_0:.*]]: tensor<21x45xi32>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x1xi32>}>
// CHECK: %[[VAL_4:.*]] = tosa.greater %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_5:.*]] = tosa.greater %[[VAL_1]], %[[VAL_0]]
// CHECK: %[[VAL_6:.*]] = tosa.select %[[VAL_5]], %[[VAL_2]], %[[VAL_1]]
// CHECK: %[[VAL_7:.*]] = tosa.select %[[VAL_4]], %[[VAL_3]], %[[VAL_6]]
func.func @test_sign(%arg0: tensor<21x45xi32>) -> tensor<21x45xi32> {
  %0 = "tfl.sign"(%arg0) : (tensor<21x45xi32>) -> tensor<21x45xi32>
    func.return %0 : tensor<21x45xi32>
}

// -----

// CHECK-LABEL: test_sin
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10xf32>
// CHECK: %[[VAL_1:.*]] = tosa.sin %[[VAL_0]] : (tensor<10xf32>
func.func @test_sin(%arg0: tensor<10xf32>) -> tensor<*xf32> {
  %0 = "tfl.sin"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_cos
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10xf32>
// CHECK: %[[VAL_1:.*]] = tosa.cos %[[VAL_0]] : (tensor<10xf32>
func.func @test_cos(%arg0: tensor<10xf32>) -> tensor<*xf32> {
  %0 = "tfl.cos"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_atan2
// CHECK-SAME: -> tensor<13x21x3xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<3.276700e+04> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<2.38418579E-7> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<1.57079637> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<3.14159274> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<{{.+}}> : tensor<513xi16>}> : () -> tensor<513xi16>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAL_10:.*]] = tosa.abs %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_11:.*]] = tosa.abs %arg1 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_12:.*]] = tosa.minimum %[[VAL_10]], %[[VAL_11]] : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_13:.*]] = tosa.maximum %[[VAL_10]], %[[VAL_11]] : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_14:.*]] = tosa.reciprocal %[[VAL_13]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_15:.*]] = tosa.mul %[[VAL_14]], %[[VAL_12]], %[[SHIFT]] : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_16:.*]] = tosa.mul %[[VAL_15]], %[[VAL_2]], %[[SHIFT]] : (tensor<13x21x3xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_17:.*]] = tosa.sub %[[VAL_16]], %[[VAL_3]] : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_18:.*]] = tosa.mul %[[VAL_17]], %[[VAL_4]], %[[SHIFT]] : (tensor<13x21x3xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_19:.*]] = tosa.cast %[[VAL_18]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xi16>
// CHECK: %[[VAL_20:.*]] = tosa.table %[[VAL_19]], %[[VAL_9]] : (tensor<13x21x3xi16>, tensor<513xi16>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_21:.*]] = tosa.cast %[[VAL_20]] : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_22:.*]] = tosa.mul %[[VAL_21]], %[[VAL_5]], %[[SHIFT]] : (tensor<13x21x3xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_23:.*]] = tosa.sub %[[VAL_6]], %[[VAL_22]] : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_24:.*]] = tosa.greater %[[VAL_10]], %[[VAL_11]] : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_25:.*]] = tosa.select %[[VAL_24]], %[[VAL_23]], %[[VAL_22]] : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_26:.*]] = tosa.sub %[[VAL_7]], %[[VAL_25]] : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_27:.*]] = tosa.greater %[[VAL_8]], %arg1 : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_28:.*]] = tosa.select %[[VAL_27]], %[[VAL_26]], %[[VAL_25]] : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_29:.*]] = tosa.negate %[[VAL_28]], %[[CONST_0]], %[[CONST_0]] : (tensor<13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_30:.*]] = tosa.greater %[[VAL_8]], %arg0 : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_31:.*]] = tosa.select %[[VAL_30]], %[[VAL_29]], %[[VAL_28]] : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAL_31]] : tensor<13x21x3xf32>
func.func @test_atan2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.atan2"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----


// CHECK-LABEL: test_sigmoid
// CHECK: %[[VAR0:.*]] = tosa.sigmoid %arg0
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR0:.*]] = tosa.mul %arg0, %arg0, %[[SHIFT]]
func.func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.square"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: %[[VAR0:.*]] = tosa.equal %arg0, %arg1
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: %[[VAR0:.*]] = tosa.greater_equal %arg0, %arg1
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: %[[VAR0:.*]] = tosa.greater %arg0, %arg1
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: %[[VAR0:.*]] = tosa.greater %arg1, %arg0
func.func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_dynamic
// CHECK: %[[VAR0:.*]] = tosa.greater %arg1, %arg0
// CHECK-SAME: -> tensor<13x?x3xi1>
func.func @test_less_dynamic(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x?x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x?x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK: %[[VAR0:.*]] = tosa.greater_equal %arg1, %arg0
func.func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_equal_dynamic
// CHECK: %[[VAR0:.*]] = tosa.greater_equal %arg1, %arg0
// CHECK-SAME: -> tensor<13x?x3xi1>
func.func @test_less_equal_dynamic(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x?x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x?x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAR0:.*]] = tosa.avg_pool2d %arg0, %[[ZP]], %[[ZP]] {acc_type = f32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_avg_pool2d_dynamic
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAR0:.*]] = tosa.avg_pool2d %arg0, %[[ZP]], %[[ZP]] {acc_type = f32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_avg_pool2d_dynamic(%arg0: tensor<?x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_avg_pool2d_slicing
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[1, 31, 31, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<1x32x32x8xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x31x31x8xf32>
// CHECK: %[[VAL_4:.*]] = tosa.avg_pool2d %[[VAL_3]], %[[ZP]], %[[ZP]] {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x31x31x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x15x15x8xf32>
func.func @test_avg_pool2d_slicing(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: %[[VAR0:.*]] = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d_dynamic
// CHECK: %[[VAR0:.*]] = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d_dynamic(%arg0: tensor<?x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[VAR10]]
func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 819]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_unknown
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[9, 91]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-SAME: -> tensor<9x91xf32>
func.func @test_reshape_unknown(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[9, -1]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_dynamic
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[3, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-SAME: -> tensor<3x?xf32>
func.func @test_reshape_dynamic(%arg0: tensor<13x21x?xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[3, -1]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x?xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_dynamic_ranked_output
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, -1, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[VAR10]]
func.func @test_reshape_dynamic_ranked_output(%arg0: tensor<?x52x52x2xf32>) -> tensor<1x?x2xf32> {
  %cst = arith.constant dense<[1, -1, 2]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<?x52x52x2xf32>, tensor<3xi32>) -> tensor<1x?x2xf32>
  func.return %0 : tensor<1x?x2xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK: %[[VAR1:.*]] = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>}
func.func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_transpose_dynamic
// CHECK: %[[VAR1:.*]] = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>}
func.func @test_transpose_dynamic(%arg0: tensor<13x?x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x?x3xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[4, 11, 1]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[6, 8, 0]> : tensor<3xindex>}
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf32>
func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[6, 8, 0]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[4, 11, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_slice_minus1_size
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[4, 13, 1]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[6, 8, 0]> : tensor<3xindex>}
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x13x1xf32>
func.func @test_slice_minus1_size(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[6, 8, 0]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[4, -1, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_simple
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[9, 7, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[9, 7, 1, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[9, 7, 3, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[9, 21, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_6]], %[[VAL_5]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x21x2xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_4]] : (tensor<9x21x2xf32>, !tosa.shape<4>) -> tensor<9x7x3x2xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : (tensor<9x7x3x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x7x1x2xf32>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<9x7x1x2xf32>, !tosa.shape<3>) -> tensor<9x7x2xf32>
func.func @test_strided_slice_simple(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_simple_negative
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[9, 18, 2]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[9, 6, 3, 2]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[9, 6, 1, 2]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[9, 6, 2]> : tensor<3xindex>}
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]]
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_3]]
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_5]], %[[VAL_4]]
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_6]]
func.func @test_strided_slice_simple_negative(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, -3, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_strideless
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[9, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[9, 1, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_4:.*]] = tosa.slice %arg0, %[[VAL_3]], %[[VAL_2]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x1x2xf32>
// CHECK: %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_1]] : (tensor<9x1x2xf32>, !tosa.shape<2>) -> tensor<9x2xf32>
func.func @test_strided_slice_strideless(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 1, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 2 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_shrink
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<7> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 7, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[1, 7, 3, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 21, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_6]], %[[VAL_5]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x21x1xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_4]] : (tensor<1x21x1xf32>, !tosa.shape<4>) -> tensor<1x7x3x1xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : (tensor<1x7x3x1xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x7x1x1xf32>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<1x7x1x1xf32>, !tosa.shape<1>) -> tensor<7xf32>
func.func @test_strided_slice_shrink(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 5 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_shrink_ignore_stride
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[1, 1, 2]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>}
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>}
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x1x2xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reshape %[[VAL_3]], %[[CONST0]] : (tensor<1x1x2xf32>, !tosa.shape<1>) -> tensor<2xf32>
func.func @test_strided_slice_shrink_ignore_stride(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 3 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_unstrided
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[9, 21, 2]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[4, 0, 1]> : tensor<3xindex>}
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x21x2xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reverse %[[VAL_3]] {axis = 2 : i32} : (tensor<9x21x2xf32>) -> tensor<9x21x2xf32>
func.func @test_strided_slice_unstrided(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 1, -1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_unstrided_shorter
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[9, 21, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[4, 0, 0]> : tensor<3xindex>}
// CHECK: %[[VAL_3:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x21x3xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reverse %[[VAL_3]] {axis = 1 : i32} : (tensor<9x21x3xf32>) -> tensor<9x21x3xf32>
func.func @test_strided_slice_unstrided_shorter(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0]> : tensor<2xi32>
  %cst_0 = arith.constant dense<[13, 21]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, -1]> : tensor<2xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<13x21x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_dynamic_masked
// CHECK-SAME: -> tensor<10x?x?xf32>
// CHECK: %[[VAR0:.*]] = tosa.reverse %arg0 {axis = 1 : i32}
// CHECK: %[[VAR1:.*]] = tosa.reverse %[[VAR0]] {axis = 2 : i32}
// CHECK: return %[[VAR1]]
func.func @test_strided_slice_dynamic_masked(%arg0: tensor<10x?x?xf32>, %arg1: tensor<3xi32>) -> tensor<*xf32> {
  %cst_0 = arith.constant dense<[13, -1, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, -1, -1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %arg1, %cst_0, %cst_1)  {begin_mask = 7 : i32, ellipsis_mask = 0 : i32, end_mask = 7 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Its possible that the begin mask is not set but the begin index is 0, which
// is equivalent to being at the beginning. However as we are bypassing the
// entire operation the operand type may not match and will need to be cast to
// the result type. These casts can be removed during shape inference.
// CHECK-LABEL: test_strided_slice_dynamic_begin
// CHECK-SAME: tensor<10x?x?xf32>
func.func @test_strided_slice_dynamic_begin(%arg0: tensor<10x?x?xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[0, 2, 0]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, -1, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, -1, -1]> : tensor<3xi32>
  // CHECK: %[[VAR0:.*]] = tosa.reverse %arg0 {axis = 1 : i32}
  // CHECK: %[[VAR1:.*]] = tosa.reverse %[[VAR0]] {axis = 2 : i32}
  // CHECK: return %[[VAR1]]
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 7 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_dynamic_end
// CHECK-SAME: 10x?x?xf32>
func.func @test_strided_slice_dynamic_end(%arg0: tensor<10x?x?xf32>) -> tensor<*xf32> {
  %begin = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %end = arith.constant dense<[7, -1, 6]> : tensor<3xi32>
  %stride = arith.constant dense<[1, 2, -1]> : tensor<3xi32>

  // CHECK-DAG: %[[CONST0:.+]] = tosa.const_shape {values = dense<[7, -1, 2, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[CONST1:.+]] = tosa.const_shape {values = dense<[7, -1]> : tensor<2xindex>}
  // CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
  // CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[7, -1, 1, 1]> : tensor<4xindex>}
  // CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[7, -1, 1]> : tensor<3xindex>}
  // CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 1, 2]> : tensor<3xindex>}
  // CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<10x?x?xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x?x1xf32>
  // CHECK: %[[VAL_6:.*]] = tosa.reshape %[[VAL_5]], %[[CONST0]]
  // CHECK: %[[VAL_7:.*]] = tosa.slice %[[VAL_6]], %[[VAL_1]], %[[VAL_2]] : (tensor<7x?x2x1xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<7x?x1x1xf32>
  // CHECK: %[[RESHAPE2:.*]] = tosa.reshape %[[VAL_7]], %[[CONST1]]
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %stride)  {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 2 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 4 : i32, offset = false}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  // CHECK: return %[[RESHAPE2]]
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_padding_even
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[4, 4, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[4, 1, 4, 1, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[4, 2, 4, 2, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[0, 1, 0, 1, 0, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_7:.*]] = tosa.pad %arg0, %[[VAL_5]], %[[VAL_6]] : (tensor<7x7x64xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<8x8x64xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_4]] : (tensor<8x8x64xf32>, !tosa.shape<5>) -> tensor<4x2x4x2x64xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : (tensor<4x2x4x2x64xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x1x4x1x64xf32>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<4x1x4x1x64xf32>, !tosa.shape<3>) -> tensor<4x4x64xf32>
func.func @test_strided_slice_padding_even(%arg0: tensor<7x7x64xf32>) -> tensor<*xf32> {
  %begin = arith.constant dense<[0, 0, 0]> : tensor<3xi32>
  %end = arith.constant dense<[0, 0, 64]> : tensor<3xi32>
  %stride = arith.constant dense<[2, 2, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %stride)  {begin_mask = 15 : i32, ellipsis_mask = 0 : i32, end_mask = 15 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<7x7x64xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_padding_odd
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[5, 5, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[5, 1, 5, 1, 32]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[5, 3, 5, 3, 32]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[0, 1, 0, 1, 0, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_7:.*]] = tosa.pad %arg0, %[[VAL_5]], %[[VAL_6]] : (tensor<14x14x32xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<15x15x32xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_4]] : (tensor<15x15x32xf32>, !tosa.shape<5>) -> tensor<5x3x5x3x32xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : (tensor<5x3x5x3x32xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<5x1x5x1x32xf32>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<5x1x5x1x32xf32>, !tosa.shape<3>) -> tensor<5x5x32xf32>
func.func @test_strided_slice_padding_odd(%arg0: tensor<14x14x32xf32>) -> tensor<*xf32> {
  %begin = arith.constant dense<[0, 0, 0]> : tensor<3xi32>
  %end = arith.constant dense<[0, 0, 32]> : tensor<3xi32>
  %stride = arith.constant dense<[3, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %stride)  {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<14x14x32xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_padding_pad_greater_than_1
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[5, 5, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[5, 1, 5, 1, 32]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[5, 3, 5, 3, 32]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[0, 2, 0, 2, 0, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_7:.*]] = tosa.pad %arg0, %[[VAL_5]], %[[VAL_6]] : (tensor<13x13x32xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<15x15x32xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_4]] : (tensor<15x15x32xf32>, !tosa.shape<5>) -> tensor<5x3x5x3x32xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_2]], %[[VAL_3]] : (tensor<5x3x5x3x32xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<5x1x5x1x32xf32>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<5x1x5x1x32xf32>, !tosa.shape<3>) -> tensor<5x5x32xf32>
func.func @test_strided_slice_padding_pad_greater_than_1(%arg0: tensor<13x13x32xf32>) -> tensor<*xf32> {
  %begin = arith.constant dense<[0, 0, 0]> : tensor<3xi32>
  %end = arith.constant dense<[0, 0, 32]> : tensor<3xi32>
  %stride = arith.constant dense<[3, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %stride)  {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false}  : (tensor<13x13x32xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %arg2, %[[VAR0]] : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
// CHECK: %[[VAR2:.*]] = tosa.select %[[VAR1]], %arg0, %arg1
func.func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<1xi1>) -> tensor<13x21x3xf32> {
  %0 = "tfl.select_v2"(%arg2, %arg0, %arg1) : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_select_with_unranked
func.func @test_select_with_unranked(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> {
  // CHECK: tosa.select
  // CHECK-SAME: (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %57 = "tfl.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  return %57 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK-DAG: %[[VAR0:.*]] = tosa.add %arg0, %arg1
// CHECK-DAG: %[[VAR1:.*]] = tosa.add %arg2, %[[VAR0]]
// CHECK: %[[VAR2:.*]] = tosa.add %arg3, %[[VAR1]]
func.func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2, %arg3) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: %[[VAR0:.*]] = tosa.concat %arg0, %arg1, %arg2, %arg3 {axis = 0 : i32}
func.func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  func.return %0 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK-DAG: %[[VAR0:.*]] = tosa.concat %arg0, %arg1, %arg2, %arg3 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[4, 13, 21, 3]> : tensor<4xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
func.func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %0 = "tfl.pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, values_count = 4 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  func.return %0 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack_end
// CHECK-DAG: %[[VAR0:.*]] = tosa.concat %arg0, %arg1 {axis = 0 : i32}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[2, 13, 21, 3]> : tensor<4xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %[[VAR0]], %[[VAR10]]
// CHECK: %[[TRANSPOSE:.*]] = tosa.transpose %[[VAR1]] {perms = array<i32: 1, 2, 3, 0>}
func.func @test_stack_end(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3x2xf32> {
  %0 = "tfl.pack"(%arg0, %arg1)  {axis = 3 : i32, values_count = 2 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3x2xf32>
  func.return %0 : tensor<13x21x3x2xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape {values = dense<[32, 32, 8]> : tensor<3xindex>}
// CHECK: %[[VAR1:.*]] = tosa.reshape %arg0, %[[VAR0]]
func.func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.unpack"(%arg0)  {axis = 0 : i32, num = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape {values = dense<[1, 1, 2, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAR1:.*]] = tosa.pad %arg0, %[[VAR0]], %[[PVAL]]
func.func @test_pad(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}


// -----

// CHECK-LABEL: test_pad_v2
// CHECK-SAME: -> tensor<1x257x9x28xf32>
func.func @test_pad_v2(%arg0: tensor<1x256x8x25xf32>) -> (tensor<*xf32>) {
  // CHECK-DAG: %[[PADDING:.+]] = tosa.const_shape {values = dense<[0, 0, 1, 0, 0, 1, 1, 2]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [1, 0], [0, 1], [1, 2]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>

  // CHECK-DAG: %[[VAL:.+]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<1xf32>}>
  %1 = "tfl.pseudo_const"() {value = dense<-3.40282347E+38> : tensor<f32>} : () -> tensor<f32>

  // CHECK-DAG: %[[PAD:.+]] = tosa.pad %arg0, %[[PADDING]], %[[VAL]] : (tensor<1x256x8x25xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x257x9x28xf32>
  %2 = "tfl.padv2"(%arg0, %0, %1) : (tensor<1x256x8x25xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: return %[[PAD]]
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_pad_v2_quant
// CHECK-DAG: %[[VAL0:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1x!quant.uniform<i8:f32, 0.023529145866632462:42>>
// CHECK-DAG: %[[VAL1:.*]] = tosa.const_shape  {values = dense<[0, 1, 0, 1, 0, 1, 0, 1]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK: %[[VAL2:.*]] = tosa.pad %arg0, %[[VAL1]], %[[VAL0]]
// CHECK: return %[[VAL2]]
func.func @test_pad_v2_quant(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.023529145866632462:42>>) -> (tensor<2x8x8x10x!quant.uniform<i8:f32, 0.023529145866632462:42>>) {
  %0 = "tfl.pseudo_const"() <{value = dense<[[0, 1], [0, 1], [0, 1], [0, 1]]> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
  %1 = "tfl.pseudo_qconst"() <{qtype = tensor<!quant.uniform<i8:f32, 0.023529145866632462:42>>, value = dense<-128> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 0.023529145866632462:42>>
  %2 = "tfl.padv2"(%arg0, %0, %1) : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.023529145866632462:42>>, tensor<4x2xi32>, tensor<!quant.uniform<i8:f32, 0.023529145866632462:42>>) -> tensor<2x8x8x10x!quant.uniform<i8:f32, 0.023529145866632462:42>>
  return %2 : tensor<2x8x8x10x!quant.uniform<i8:f32, 0.023529145866632462:42>>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 13, 21, 3]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[VAR10]]
func.func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 13, 21, 3]> : tensor<4xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<4xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_minus_1
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[13, 21, 3, 1]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[SHAPE]]
func.func @test_expand_dims_minus_1(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?x?xf32> {
  %cst = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_minus_2
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[13, 21, 1, 3]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[SHAPE]]
func.func @test_expand_dims_minus_2(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?x?xf32> {
  %cst = "tfl.pseudo_const"() {value = dense<-2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_0
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[1, 13, 21, 3]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[SHAPE]]
func.func @test_expand_dims_0(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?x?xf32> {
  %cst = "tfl.pseudo_const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_2
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[13, 21, 1, 3]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[SHAPE]]
func.func @test_expand_dims_2(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?x?xf32> {
  %cst = "tfl.pseudo_const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_expand_dims_size
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[13, 21, 3, 1]> : tensor<4xindex>}
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg0, %[[SHAPE]]
func.func @test_expand_dims_size(%arg0: tensor<13x21x3xf32>) -> tensor<?x?x?x?xf32> {
  %cst = "tfl.pseudo_const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  %0 = "tfl.expand_dims"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{values = dense<[13, 21, 3]> : tensor<3xi32>}>
func.func @test_shape() -> tensor<3xi32> {
  %cst = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  func.return %cst : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: %[[VAR0:.*]] = "tosa.const"() <{values = dense<3> : tensor<i32>}>
func.func @test_rank() -> tensor<i32> {
  %cst = arith.constant dense<3> : tensor<i32>
  func.return %cst : tensor<i32>
}

// -----

// CHECK-LABEL: test_elu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR2:.*]] = tosa.exp %arg0
// CHECK-DAG: %[[VAR4:.*]] = tosa.sub %[[VAR2]], %[[VAR0]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.greater_equal %arg0, %[[VAR1]]
// CHECK: %[[VAR7:.*]] = tosa.select %[[VAR6]], %arg0, %[[VAR4]]
func.func @test_elu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.elu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_max %arg0
// CHECK-DAG: %[[VAR1:.*]] = tosa.sub %arg0, %[[VAR0]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.exp %[[VAR1]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.reduce_sum %[[VAR2]] {axis = 2 : i32}
// CHECK-DAG: %[[VAR4:.*]] = tosa.reciprocal %[[VAR3]]
// CHECK: %[[VAR5:.*]] = tosa.mul %[[VAR2]], %[[VAR4]], %[[SHIFT]]
func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_l2normalization
func.func @test_l2normalization(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
  // CHECK-DAG: %[[MIN:.+]] = "tosa.const"() <{values = dense<1.08420217E-19> : tensor<1x1xf32>}>
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
  // CHECK-DAG: %[[SQR:.+]] = tosa.mul %arg0, %arg0, %[[SHIFT]]
  // CHECK-DAG: %[[SUM:.+]] = tosa.reduce_sum %[[SQR]] {axis = 1 : i32}
  // CHECK-DAG: %[[MAX:.+]] = tosa.maximum %[[SUM]], %[[MIN]]
  // CHECK-DAG: %[[RSQRT:.+]] = tosa.rsqrt %[[MAX]]
  // CHECK-DAG: %[[MUL:.+]] = tosa.mul %[[RSQRT]], %arg0, %[[SHIFT]]
  // CHECK: %[[CLAMP:.+]] = tosa.clamp %[[MUL]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32}
  %0 = "tfl.l2_normalization"(%arg0) {fused_activation_function = "RELU"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = tosa.exp %arg0
// CHECK-DAG: %[[VAR1:.*]] = tosa.reduce_sum %[[VAR0]] {axis = 2 : i32}
// CHECK-DAG: %[[VAR2:.*]] = tosa.reciprocal %[[VAR1]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.mul %[[VAR0]], %[[VAR2]], %[[SHIFT]]
// CHECK: %[[VAR4:.*]] = tosa.log %[[VAR3]]
func.func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log_softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape {values = dense<[14, 1, 1, 19]> : tensor<4xindex>}
// CHECK-DAG: %[[CONST1:.*]] = tosa.const_shape {values = dense<[28, 1, 1, 19]> : tensor<4xindex>}
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape {values = dense<[14, 28]> : tensor<2xindex>}
// CHECK-DAG: %[[CONST3:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[VAR2:.*]] = tosa.transpose %arg1 {perms = array<i32: 1, 0>}
// CHECK: %[[VAR3:.*]] = tosa.reshape %arg0, %[[CONST0]]
// CHECK: %[[VAR4:.*]] = tosa.reshape %[[VAR2]], %[[CONST1]]
// CHECK: %[[VAR5:.*]] = tosa.conv2d %[[VAR3]], %[[VAR4]], %[[CONST3]], %[[CONST3]], %[[CONST3]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR6:.*]] = tosa.reshape %[[VAR5]], %[[CONST2]]
func.func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst_0 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose"(%arg1, %cst) : (tensor<19x28xf32>, tensor<2xi32>) -> tensor<*xf32>
  %1 = "tfl.fully_connected"(%arg0, %0, %cst_0)  {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}  : (tensor<14x19xf32>, tensor<*xf32>, none) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul
func.func @test_batch_matmul(%arg0: tensor<1x16x128xf32>, %arg1: tensor<1x128x32xf32>) -> (tensor<1x16x32xf32> ) {
  // CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
  // CHECK: %[[VAR0:.*]] = tosa.matmul %arg0, %arg1, %[[ZP]], %[[ZP]]
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x16x128xf32>, tensor<1x128x32xf32>) -> tensor<1x16x32xf32>
  func.return %0 : tensor<1x16x32xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul2d
func.func @test_batch_matmul2d(%arg0: tensor<16x128xf32>, %arg1: tensor<128x32xf32>) -> (tensor<16x32xf32> ) {
  // CHECK-DAG: %[[VAR_10:.*]] = tosa.const_shape {values = dense<[1, 16, 128]> : tensor<3xindex>}
  // CHECK-DAG: %[[VAR_11:.*]] = tosa.const_shape {values = dense<[1, 128, 32]> : tensor<3xindex>}
  // CHECK-DAG: %[[VAR_12:.*]] = tosa.const_shape {values = dense<[16, 32]> : tensor<2xindex>}
  // CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
  // CHECK: %[[VAL_0:.*]] = tosa.reshape %arg0, %[[VAR_10]]
  // CHECK: %[[VAL_1:.*]] = tosa.reshape %arg1, %[[VAR_11]]
  // CHECK: %[[VAL_2:.*]] = tosa.matmul %[[VAL_0]], %[[VAL_1]], %[[ZP]], %[[ZP]]
  // CHECK: %[[VAL_3:.*]] = tosa.reshape %[[VAL_2]], %[[VAR_12]]
  // CHECK: return %[[VAL_3]]
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<16x128xf32>, tensor<128x32xf32>) -> tensor<16x32xf32>
  func.return %0 : tensor<16x32xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul_4d
func.func @test_batch_matmul_4d(%arg0: tensor<4x5x16x128xf32>, %arg1: tensor<4x5x128x32xf32>) -> (tensor<4x5x16x32xf32> ) {
  // CHECK-DAG: %[[C0:.*]] = tosa.const_shape {values = dense<[20, 16, 128]> : tensor<3xindex>}
  // CHECK-DAG: %[[C1:.*]] = tosa.const_shape {values = dense<[20, 128, 32]> : tensor<3xindex>}
  // CHECK-DAG: %[[C2:.*]] = tosa.const_shape {values = dense<[4, 5, 16, 32]> : tensor<4xindex>}
  // CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
  // CHECK: %[[R0:.*]] = tosa.reshape %arg0, %[[C0]]
  // CHECK: %[[R1:.*]] = tosa.reshape %arg1, %[[C1]]
  // CHECK: %[[MM:.*]] = tosa.matmul %[[R0]], %[[R1]], %[[ZP]], %[[ZP]]
  // CHECK: tosa.reshape %[[MM]], %[[C2]]
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<4x5x16x128xf32>, tensor<4x5x128x32xf32>) -> tensor<4x5x16x32xf32>
  func.return %0 : tensor<4x5x16x32xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul_transpose
func.func @test_batch_matmul_transpose(%arg0: tensor<1x16x128xf32>, %arg1: tensor<1x128x32xf32>) -> (tensor<1x32x16xf32> ) {
  // CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
  // CHECK-DAG: %[[TP0:.+]] = tosa.transpose %arg0 {perms = array<i32: 0, 2, 1>}
  // CHECK-DAG: %[[TP1:.+]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 1>}
  // CHECK: tosa.matmul %[[TP1]], %[[TP0]], %[[ZP]], %[[ZP]]
  %0 = "tfl.batch_matmul"(%arg1, %arg0) {adj_x = true, adj_y = true} : (tensor<1x128x32xf32>, tensor<1x16x128xf32>) -> tensor<1x32x16xf32>
  func.return %0 : tensor<1x32x16xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<40> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1488699087> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 3, 4, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[3, 4, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[3, 4, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_9:.*]] = tosa.reshape %[[VAL_0]], %[[VAL_8]] : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>, !tosa.shape<3>) -> tensor<3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_7]] : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>, !tosa.shape<3>) -> tensor<3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>
// CHECK: %[[VAL_11:.*]] = tosa.matmul %[[VAL_9]], %[[VAL_10]], %[[VAL_6]], %[[VAL_6]] : (tensor<3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>, tensor<3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>, tensor<1xi8>, tensor<1xi8>) -> tensor<3x4x3xi32>
// CHECK: %[[VAL_12:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_5]] : (tensor<3x4x3xi32>, !tosa.shape<4>) -> tensor<1x3x4x3xi32>
// CHECK: %[[VAL_13:.*]] = tosa.rescale %[[VAL_12]], %[[VAL_3]], %[[VAL_2]], %[[VAL_4]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x3x4x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.011357889510691166:-128>>
func.func @test_batch_matmul_qi8(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>, %arg1: tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.011357889510691166:-128>> {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.003921466413885355:-128>>, tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0039215362630784512:-128>>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.011357889510691166:-128>>
  return %0 : tensor<1x3x4x3x!quant.uniform<i8:f32, 0.011357889510691166:-128>>
}

// -----

// CHECK-LABEL: test_batch_matmul_with_input_broadcast
// CHECK-SAME: %[[ARG0:.*]]: tensor<25x12x14x14x64xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<14x64x14xf32>
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape  {values = dense<[1, 1, 14, 64, 14]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<-0.000000e+00> : tensor<25x12x14x64x14xf32>}> : () -> tensor<25x12x14x64x14xf32>
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %[[ARG1]], %[[VAR0]] : (tensor<14x64x14xf32>, !tosa.shape<5>) -> tensor<1x1x14x64x14xf32>
// CHECK: tosa.add %[[VAR5]], %[[VAR1]] : (tensor<1x1x14x64x14xf32>, tensor<25x12x14x64x14xf32>) -> tensor<25x12x14x64x14xf32>
func.func @test_batch_matmul_with_input_broadcast(%arg0: tensor<25x12x14x14x64xf32>, %arg1: tensor<14x64x14xf32>) -> (tensor<25x12x14x14x14xf32> ) {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<25x12x14x14x64xf32>, tensor<14x64x14xf32>) -> tensor<25x12x14x14x14xf32>
  func.return %0 : tensor<25x12x14x14x14xf32>
}

// -----

// CHECK-LABEL: test_batch_matmul_qi16
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x3x4x4x!quant.uniform<i16:f32, 3.0517894629156217E-5>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<1x3x4x3x!quant.uniform<i16:f32, 3.051840394618921E-5>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<20139> : tensor<1xi16>}> : () -> tensor<1xi16>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi48>}> : () -> tensor<1xi48>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 3, 4, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[3, 4, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[3, 4, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_9:.*]] = tosa.reshape %[[VAL_0]], %[[VAL_8]] : (tensor<1x3x4x4x!quant.uniform<i16:f32, 3.0517894629156217E-5>>, !tosa.shape<3>)
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_7]] : (tensor<1x3x4x3x!quant.uniform<i16:f32, 3.051840394618921E-5>>, !tosa.shape<3>)
// CHECK: %[[VAL_11:.*]] = tosa.matmul %[[VAL_9]], %[[VAL_10]], %[[VAL_6]], %[[VAL_6]] : (tensor<3x4x4x!quant.uniform<i16:f32, 3.0517894629156217E-5>>, tensor<3x4x3x!quant.uniform<i16:f32, 3.051840394618921E-5>>, tensor<1xi16>, tensor<1xi16>)
// CHECK: %[[VAL_12:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_5]] : (tensor<3x4x3xi48>, !tosa.shape<4>) -> tensor<1x3x4x3xi48>
// CHECK: %[[VAL_13:.*]] = tosa.rescale %[[VAL_12]], %[[VAL_3]], %[[VAL_2]], %[[VAL_4]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = false} : (tensor<1x3x4x3xi48>, tensor<1xi16>, tensor<1xi8>, tensor<1xi48>, tensor<1xi16>)
// CHECK: return %[[VAL_13]] : tensor<1x3x4x3x!quant.uniform<i16:f32, 9.9311851954553276E-5>>
func.func @test_batch_matmul_qi16(%arg0: tensor<1x3x4x4x!quant.uniform<i16:f32, 3.0517894629156217E-5>>, %arg1: tensor<1x3x4x3x!quant.uniform<i16:f32, 3.051840394618921E-5>>) -> (tensor<1x3x4x3x!quant.uniform<i16:f32, 9.9311851954553276E-5>>) {
%0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<1x3x4x4x!quant.uniform<i16:f32, 3.0517894629156217E-5>>, tensor<1x3x4x3x!quant.uniform<i16:f32, 3.051840394618921E-5>>) -> tensor<1x3x4x3x!quant.uniform<i16:f32, 9.9311851954553276E-5>>
return %0 : tensor<1x3x4x3x!quant.uniform<i16:f32, 9.9311851954553276E-5>>
}

// -----

// CHECK-LABEL: test_batch_matmul_with_input_broadcast_1
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x256x256x32xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x32x4xf32>
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape  {values = dense<[1, 1, 32, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<-0.000000e+00> : tensor<1x256x32x4xf32>}> : () -> tensor<1x256x32x4xf32>
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %[[ARG1]], %[[VAR0]] : (tensor<1x32x4xf32>, !tosa.shape<4>) -> tensor<1x1x32x4xf32>
// CHECK-DAG: %[[VAR6:.*]] = tosa.add %[[VAR5]], %[[VAR1]] : (tensor<1x1x32x4xf32>, tensor<1x256x32x4xf32>) -> tensor<1x256x32x4xf32>
func.func @test_batch_matmul_with_input_broadcast_1(%arg0: tensor<1x256x256x32xf32>, %arg1: tensor<1x32x4xf32>) -> (tensor<1x256x256x4xf32>) {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x256x256x32xf32>, tensor<1x32x4xf32>) -> tensor<1x256x256x4xf32>
  func.return %0 : tensor<1x256x256x4xf32>
}

// -----

// CHECK-LABEL: test_batch_matmul_with_input_broadcast_qi8
// CHECK-SAME: %[[ARG0:.*]]: tensor<25x12x14x14x64x!quant.uniform<i8:f32, 0.061956532299518585:4>>
// CHECK-SAME: %[[ARG1:.*]]: tensor<14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape  {values = dense<[1, 1, 14, 64, 14]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0> : tensor<25x12x14x64x14xi32>}> : () -> tensor<25x12x14x64x14xi32>
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[ARG1]], %[[VAR0]] : (tensor<14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>, !tosa.shape<5>) -> tensor<1x1x14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>
// CHECK-DAG: %[[VAR8:.*]] = tosa.cast %[[VAR7]] : (tensor<1x1x14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>) -> tensor<1x1x14x64x14xi32>
// CHECK-DAG: %[[VAR9:.*]] = tosa.add %[[VAR8]], %[[VAR1]] : (tensor<1x1x14x64x14xi32>, tensor<25x12x14x64x14xi32>) -> tensor<25x12x14x64x14xi32>
// CHECK: tosa.cast %[[VAR9]] : (tensor<25x12x14x64x14xi32>) -> tensor<25x12x14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>
func.func @test_batch_matmul_with_input_broadcast_qi8(%arg0: tensor<25x12x14x14x64x!quant.uniform<i8:f32, 0.061956532299518585:4>>, %arg1: tensor<14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>) -> (tensor<25x12x14x14x14x!quant.uniform<i8:f32, 0.017063580453395844:-47>>) {
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false, asymmetric_quantize_inputs = false} : (tensor<25x12x14x14x64x!quant.uniform<i8:f32, 0.061956532299518585:4>>, tensor<14x64x14x!quant.uniform<i8:f32, 3.9322837255895138E-4:1>>) -> tensor<25x12x14x14x14x!quant.uniform<i8:f32, 0.017063580453395844:-47>>
  func.return %0 : tensor<25x12x14x14x14x!quant.uniform<i8:f32, 0.017063580453395844:-47>>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1xf32>}>
// CHECK: %[[VAR2:.*]] = tosa.add %arg0, %[[VAR0]]
func.func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK-DAG: %[[VAR0:.*]] = tosa.reduce_sum %arg1 {axis = 0 : i32}
// CHECK-DAG: %[[VAR1:.*]] = tosa.reduce_sum %[[VAR0]] {axis = 1 : i32}
// CHECK: %[[VAR2:.*]] = tosa.add %arg0, %[[VAR1]]
func.func @test_add_1d(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  %0 = "tfl.sum"(%arg1, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3xf32>
  %1 = "tfl.add"(%arg0, %0)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun_clamp
func.func @test_fused_activation_relun_clamp(
    %arg0: tensor<10x!quant.uniform<i8:f32, 0.1:-127>>,
    %arg1: tensor<10x!quant.uniform<i8:f32, 0.1:-127>>) ->
    tensor<10x!quant.uniform<i8:f32, 0.1:-127>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: tosa.clamp %{{.+}} {max_val = -67 : i8, min_val = -127 : i8}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU6"} : (tensor<10x!quant.uniform<i8:f32, 0.1:-127>>, tensor<10x!quant.uniform<i8:f32, 0.1:-127>>) -> tensor<10x!quant.uniform<i8:f32, 0.1:-127>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.1:-127>>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun_noclamp
func.func @test_fused_activation_relun_noclamp(
    %arg0: tensor<10x!quant.uniform<i8:f32, 0.01:-129>>,
    %arg1: tensor<10x!quant.uniform<i8:f32, 0.01:-129>>) ->
    tensor<10x!quant.uniform<i8:f32, 0.01:-129>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: tosa.clamp %{{.+}} {max_val = 127 : i8, min_val = -128 : i8}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU6"} : (tensor<10x!quant.uniform<i8:f32, 0.01:-129>>, tensor<10x!quant.uniform<i8:f32, 0.01:-129>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-129>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.01:-129>>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun1to1_noclamp
func.func @test_fused_activation_relun1to1_noclamp(
                         %arg0: tensor<10x!quant.uniform<i8:f32, 0.001:-120>>,
                         %arg1: tensor<10x!quant.uniform<i8:f32, 0.001:-120>>) -> tensor<10x!quant.uniform<i8:f32, 0.001:-120>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK:  tosa.clamp %{{.}} {max_val = 127 : i8, min_val = -128 : i8}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU_N1_TO_1"}  : (tensor<10x!quant.uniform<i8:f32, 0.001:-120>>, tensor<10x!quant.uniform<i8:f32, 0.001:-120>>) -> tensor<10x!quant.uniform<i8:f32, 0.001:-120>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.001:-120>>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun1to1_clamp
func.func @test_fused_activation_relun1to1_clamp(
                         %arg0: tensor<10x!quant.uniform<i8:f32, 0.01:-10>>,
                         %arg1: tensor<10x!quant.uniform<i8:f32, 0.01:-10>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-10>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK:  tosa.clamp %{{.}} {max_val = 90 : i8, min_val = -110 : i8}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU_N1_TO_1"}  : (tensor<10x!quant.uniform<i8:f32, 0.01:-10>>, tensor<10x!quant.uniform<i8:f32, 0.01:-10>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-10>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.01:-10>>
}

// -----

// CHECK-LABEL: test_split
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[0, 14, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[0, 7, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[13, 7, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
// CHECK: %[[VAL_6:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_1]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
func.func @test_split(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_split_dynamic
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[0, 2, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[0, 1, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[13, -1, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[13, 1, -1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[13, 3, -1, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_7:.*]] = tosa.reshape %arg0, %[[VAL_6]] : (tensor<13x?x3xf32>, !tosa.shape<4>) -> tensor<13x3x?x3xf32>
// CHECK: %[[VAL_8:.*]] = tosa.slice %[[VAL_7]], %[[VAL_4]], %[[VAL_5]] : (tensor<13x3x?x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<13x1x?x3xf32>
// CHECK: %[[VAL_9:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_3]] : (tensor<13x1x?x3xf32>, !tosa.shape<3>) -> tensor<13x?x3xf32>
// CHECK: %[[VAL_10:.*]] = tosa.slice %[[VAL_7]], %[[VAL_2]], %[[VAL_5]] : (tensor<13x3x?x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<13x1x?x3xf32>
// CHECK: %[[VAL_11:.*]] = tosa.reshape %[[VAL_10]], %[[VAL_3]] : (tensor<13x1x?x3xf32>, !tosa.shape<3>) -> tensor<13x?x3xf32>
// CHECK: %[[VAL_12:.*]] = tosa.slice %[[VAL_7]], %[[VAL_1]], %[[VAL_5]] : (tensor<13x3x?x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<13x1x?x3xf32>
// CHECK: %[[VAL_13:.*]] = tosa.reshape %[[VAL_12]], %[[VAL_3]] : (tensor<13x1x?x3xf32>, !tosa.shape<3>) -> tensor<13x?x3xf32>
func.func @test_split_dynamic(%arg0: tensor<13x?x3xf32>) -> (tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>) {
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x?x3xf32>) -> (tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>
}


// -----

// CHECK-LABEL: test_split_neg
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[0, 14, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[0, 7, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[13, 7, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
// CHECK: %[[VAL_6:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_1]], %[[VAL_3]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x7x3xf32>
// CHECK: return %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
func.func @test_split_neg(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %cst_0 = arith.constant dense<-2> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_split_axis_0
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[14, 0, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[7, 0, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[7, 13, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<21x13x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x13x3xf32>
// CHECK: %[[VAL_6:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_3]] : (tensor<21x13x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x13x3xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %arg0, %[[VAL_1]], %[[VAL_3]] : (tensor<21x13x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x13x3xf32>
func.func @test_split_axis_0(%arg0: tensor<21x13x3xf32>) -> (tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>) {
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<21x13x3xf32>) -> (tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>
}

// -----

// CHECK-LABEL: test_split_v_neg_axis
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 3]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[2, 3, 3, 5]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[2, 3, 3, 3]> : tensor<4xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>}
// CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<2x3x3x8xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x3x3x3xf32>
// CHECK: %[[VAL_6:.*]] = tosa.slice %arg0, %[[VAL_1]], %[[VAL_2]] : (tensor<2x3x3x8xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x3x3x5xf32>
func.func @test_split_v_neg_axis(%arg0: tensor<2x3x3x8xf32>) -> (tensor<2x3x3x3xf32>, tensor<2x3x3x5xf32>) {
  %split_size = arith.constant dense<[3, 5]> : tensor<2xi32>
  %axis = arith.constant dense<-1> : tensor<i32>
  %0, %1 = "tfl.split_v"(%arg0, %split_size, %axis)  {num_splits = 2 : i32}  : (tensor<2x3x3x8xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<2x3x3x3xf32>, tensor<2x3x3x5xf32>)
  func.return %0, %1 : tensor<2x3x3x3xf32>, tensor<2x3x3x5xf32>
}

// -----

// CHECK-LABEL: test_tile
// CHECK: tosa.tile
func.func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[3, 1, 2]> : tensor<3xi32>
  %0 = "tfl.tile"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_space_to_batch
// CHECK-DAG: %[[VAR0:.*]] = tosa.const_shape {values = dense<[0, 0, 0, 1, 0, 0]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[VAR2:.*]] = tosa.pad %arg0, %[[VAR0]], %[[PVAL]]
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[13, 11, 2, 3]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR10]]
// CHECK-DAG: %[[VAR4:.*]] = tosa.transpose %[[VAR3]] {perms = array<i32: 2, 0, 1, 3>}
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[26, 11, 3]> : tensor<3xindex>}
// CHECK: %[[VAR5:.*]] = tosa.reshape %[[VAR4]], %[[VAR11]]
func.func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %cst_0 = arith.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %0 = "tfl.space_to_batch_nd"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  func.return %0 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_space_to_batch_dyn
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[-1, 81, 1, 80]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[-1, 81, 3, 1, 1, 80]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 2, 0, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_6:.*]] = tosa.pad %arg0, %[[VAL_4]], %[[VAL_5]] : (tensor<?x241x1x80xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<?x243x1x80xf32>
// CHECK: %[[VAL_7:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_3]] : (tensor<?x243x1x80xf32>, !tosa.shape<6>) -> tensor<?x81x3x1x1x80xf32>
// CHECK: %[[VAL_8:.*]] = tosa.transpose %[[VAL_7]] {perms = array<i32: 2, 4, 0, 1, 3, 5>} : (tensor<?x81x3x1x1x80xf32>) -> tensor<3x1x?x81x1x80xf32>
// CHECK: %[[VAL_9:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_1]] : (tensor<3x1x?x81x1x80xf32>, !tosa.shape<4>) -> tensor<?x81x1x80xf32>
func.func @test_space_to_batch_dyn(%arg0 : tensor<?x241x1x80xf32>) -> (tensor<?x81x1x80xf32>) {
    %0 = "tfl.pseudo_const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() {value = dense<[[0, 2], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tfl.space_to_batch_nd"(%arg0, %0, %1) : (tensor<?x241x1x80xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x81x1x80xf32>
    func.return %2 : tensor<?x81x1x80xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: %[[VAR2:.*]] = tosa.transpose %arg0 {perms = array<i32: 3, 1, 2, 0>}
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[2, 2, 2, 32, 32, 1]> : tensor<6xindex>}
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR10]]
// CHECK-DAG: %[[VAR4:.*]] = tosa.transpose %[[VAR3]] {perms = array<i32: 2, 3, 0, 4, 1, 5>}
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[2, 64, 64, 1]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %[[VAR4]], %[[VAR11]]
// CHECK: return %[[VAR5:.*]]
func.func @test_batch_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<2x64x64x1xf32> {
  %cst = arith.constant dense<2> : tensor<2xi32>
  %cst_0 = arith.constant dense<0> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<[3, 1, 2, 0]> : tensor<4xi32>
  %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<1x32x32x8xf32>, tensor<4xi32>) -> tensor<8x32x32x1xf32>
  %1 = "tfl.batch_to_space_nd"(%0, %cst, %cst_0) : (tensor<8x32x32x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<2x64x64x1xf32>
  func.return %1 : tensor<2x64x64x1xf32>
}

// -----

// CHECK-LABEL: @test_batch_to_space_dyn
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[-1, 235, 1, 80]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[-1, 237, 1, 80]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[3, 1, -1, 79, 1, 80]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK: %[[VAL_6:.*]] = tosa.reshape %arg0, %[[VAL_5]] : (tensor<?x79x1x80xf32>, !tosa.shape<6>) -> tensor<3x1x?x79x1x80xf32>
// CHECK: %[[VAL_7:.*]] = tosa.transpose %[[VAL_6]] {perms = array<i32: 2, 3, 0, 4, 1, 5>} : (tensor<3x1x?x79x1x80xf32>) -> tensor<?x79x3x1x1x80xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_3]] : (tensor<?x79x3x1x1x80xf32>, !tosa.shape<4>) -> tensor<?x237x1x80xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_1]], %[[VAL_2]] : (tensor<?x237x1x80xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x235x1x80xf32>
func.func @test_batch_to_space_dyn(%arg0 : tensor<?x79x1x80xf32>) -> (tensor<?x235x1x80xf32>) {
    %0 = "tfl.pseudo_const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() {value = dense<[[0, 2], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tfl.batch_to_space_nd"(%arg0, %0, %1) : (tensor<?x79x1x80xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x235x1x80xf32>
    func.return %2 : tensor<?x235x1x80xf32>
}

// -----

// CHECK-LABEL: @test_batch_to_space_shape_infer
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[1, 135, 240, 384]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 136, 240, 384]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[2, 2, 1, 68, 120, 384]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK: %[[VAL_6:.*]] = tosa.reshape %arg0, %[[VAL_5]] : (tensor<4x68x120x384xf32>, !tosa.shape<6>) -> tensor<2x2x1x68x120x384xf32>
// CHECK: %[[VAL_7:.*]] = tosa.transpose %[[VAL_6]] {perms = array<i32: 2, 3, 0, 4, 1, 5>} : (tensor<2x2x1x68x120x384xf32>) -> tensor<1x68x2x120x2x384xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_7]], %[[VAL_3]] : (tensor<1x68x2x120x2x384xf32>, !tosa.shape<4>) -> tensor<1x136x240x384xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_1]], %[[VAL_2]] : (tensor<1x136x240x384xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x135x240x384xf32>
func.func @test_batch_to_space_shape_infer(%arg0 : tensor<4x68x120x384xf32>) -> (tensor<?x135x240x384xf32>) {
    %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() {value = dense<[[0, 1], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tfl.batch_to_space_nd"(%arg0, %0, %1) : (tensor<4x68x120x384xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x135x240x384xf32>
    func.return %2 : tensor<?x135x240x384xf32>
}

// -----

// CHECK-LABEL: test_space_to_depth
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 16, 2, 16, 2, 8]> : tensor<6xindex>}
// CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.transpose %[[VAR1]] {perms = array<i32: 0, 1, 3, 2, 4, 5>}
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 16, 16, 32]> : tensor<4xindex>}
// CHECK: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR11]]
func.func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %0 = "tfl.space_to_depth"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  func.return %0 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 32, 32, 2, 2, 2]> : tensor<6xindex>}
// CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.transpose %[[VAR1]] {perms = array<i32: 0, 1, 3, 2, 4, 5>}
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 64, 64, 2]> : tensor<4xindex>}
// CHECK: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR11]]
func.func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %0 = "tfl.depth_to_space"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  func.return %0 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: @test_bucketize
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape  {values = dense<[2, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape  {values = dense<[2, 5, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>

// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[\[}}0.000000e+00, 3.000000e+00, 8.000000e+00, 1.100000e+01]]]> : tensor<1x1x4xf32>}>
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[CONST2]]
// CHECK: %[[VAL_2:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_0]]
// CHECK: %[[VAL_3:.*]] = tosa.cast %[[VAL_2]] : (tensor<2x5x4xi1>) -> tensor<2x5x4xi32>
// CHECK: %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 2 : i32}
// CHECK: %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]], %[[CONST0]]
func.func @test_bucketize(%arg0: tensor<2x5xf32>) -> tensor<2x5xi32> {
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x5xf32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @test_bucketize
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape  {values = dense<[2, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape  {values = dense<[2, 5, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>

// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<{{\[\[\[}}0.000000e+00, 3.000000e+00, 8.000000e+00, 1.100000e+01]]]> : tensor<1x1x4xf32>}>
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[CONST2]]
// CHECK: %[[VAL_2:.*]] = tosa.greater_equal %[[VAL_1]], %[[VAL_0]]
// CHECK: %[[VAL_3:.*]] = tosa.cast %[[VAL_2]] : (tensor<2x5x4xi1>) -> tensor<2x5x4xi32>
// CHECK: %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 2 : i32}
// CHECK: %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]], %[[CONST0]]
func.func @test_bucketize(%arg0: tensor<2x5xf32>) -> tensor<2x5xi32> {
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x5xf32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @test_one_hot
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x4xi32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>
// CHECK-DAG:     %[[CST0:.*]] = tosa.const_shape {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:     %[[CST1:.*]] = tosa.const_shape {values = dense<[16, 1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:     %[[CST2:.*]] = tosa.const_shape {values = dense<[16, 2, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:     %[[CST3:.*]] = tosa.const_shape {values = dense<[16, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:     %[[CST4:.*]] = tosa.const_shape {values = dense<[4, 4, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:     %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]], %[[CST0]]
// CHECK-DAG:     %[[TILE:.*]] = tosa.tile %[[RESHAPE]], %[[CST1]]
// CHECK-DAG:     %[[RESHAPE_0:.*]] = tosa.reshape %[[ARG2]], %[[CST0]]
// CHECK-DAG:     %[[TILE_0:.*]] = tosa.tile %[[RESHAPE_0]], %[[CST2]]
// CHECK-DAG:     %[[RESHAPE_1:.*]] = tosa.reshape %[[ARG0]], %[[CST3]]
// CHECK-DAG:     %[[SCATTER:.*]] = tosa.scatter %[[TILE_0]], %[[RESHAPE_1]], %[[TILE]]
// CHECK-DAG:     %[[RESHAPE_2:.*]] = tosa.reshape %[[SCATTER]], %[[CST4]]
// CHECK:         return %[[RESHAPE_2]]
func.func @test_one_hot(%arg0: tensor<4x4xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<4x4x2xf32> {
  %0 = arith.constant dense<2> : tensor<i32>
  %1 = "tfl.one_hot"(%arg0, %0, %arg1, %arg2) {axis = -1 : i32} : (tensor<4x4xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<4x4x2xf32>
  func.return %1 : tensor<4x4x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<6.10360876E-5> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<16383.75> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.mul %arg0, %[[VAR2]], %[[SHIFT]]
// CHECK-DAG: %[[VAR5:.*]] = tosa.cast %[[VAR3]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.cast %[[VAR5]]
// CHECK-DAG: %[[VAR8:.*]] = tosa.mul %[[VAR6]], %[[VAR1]], %[[SHIFT]]
func.func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.quantize"(%arg0)  {qtype = tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>}  : (tensor<13x21x3xf32>) -> tensor<*x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>
  %1 = "tfl.dequantize"(%0) : (tensor<*x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  func.return %1 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: @test_dequantize_float
// CHECK-SAME: -> tensor<10xf32>
func.func @test_dequantize_float(%arg0: tensor<10xf16>) -> tensor<*xf32> {
  // CHECK: %[[VAR0:.+]] = tosa.cast %arg0 : (tensor<10xf16>) -> tensor<10xf32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<10xf16>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_dequantize_quant_uniform
func.func @test_dequantize_quant_uniform(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:-1>>) -> tensor<*xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<1xf32>}>
  // CHECK-DAG: %[[VAL1:.+]] = tosa.cast %arg0
  // CHECK-DAG: %[[VAL2:.+]] = tosa.sub %[[VAL1]], %[[VAL0]]
  %0 = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 1.0:-1>>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// -----

// CHECK-LABEL: @test_dequantize_quant_per_axis
func.func @test_dequantize_quant_per_axis(%arg0: tensor<1x4x!quant.uniform<i8:f32:1, {1.0:5, 2.0:6, 3.0:7, 4.0:8}>>) -> tensor<*xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tosa.const"() <{values = dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tosa.const"() <{values = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]]> : tensor<1x4xf32>}>
  // CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
  // CHECK-DAG: %[[VAL2:.+]] = tosa.cast %arg0 : (tensor<1x4x!quant.uniform<i8:f32:1, {1.000000e+00:5,2.000000e+00:6,3.000000e+00:7,4.000000e+00:8}>>) -> tensor<1x4xf32>
  // CHECK-DAG: %[[VAL3:.+]] = tosa.sub %[[VAL2]], %[[VAL1]] : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[VAL4:.+]] = tosa.mul %[[VAL3]], %[[VAL0]], %[[SHIFT]] : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1xi8>) -> tensor<1x4xf32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x4x!quant.uniform<i8:f32:1, {1.0:5, 2.0:6, 3.0:7, 4.0:8}>>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_quantfork.stats
func.func @test_quantfork.stats(%arg0: tensor<2x1xf32>) -> (tensor<2x1xf32>) {
  // CHECK-NOT: quantfork.stats
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[1.0, 1.0]> : tensor<2xf32>}: (tensor<2x1xf32>) -> tensor<2x1xf32>
  func.return %0 : tensor<2x1xf32>
}

// -----

// CHECK-LABEL: test_add_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<1075580483> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<2147311776> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_11:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x1xi32>
// CHECK: %[[VAL_12:.*]] = tosa.rescale %[[VAL_11]], %[[VAL_6]], %[[VAL_5]], %[[VAL_10]], %[[VAL_10]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<13x21x1xi32>
// CHECK: %[[VAL_13:.*]] = tosa.rescale %[[VAL_1]], %[[VAL_7]], %[[VAL_4]], %[[VAL_9]], %[[VAL_10]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_14:.*]] = tosa.add %[[VAL_12]], %[[VAL_13]] : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_15:.*]] = tosa.rescale %[[VAL_14]], %[[VAL_3]], %[[VAL_2]], %[[VAL_10]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
// CHECK: return %[[VAL_15]] : tensor<13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
func.func @test_add_qi8(%arg0: tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>> {
  %0 = tfl.add(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
}

// -----

// CHECK-LABEL: test_sub_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x21x3x!quant.uniform<i8:f32, 0.015685770660638809:-1>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686184167861938:-1>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1076408862> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<2147427038> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_12:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x21x3x!quant.uniform<i8:f32, 0.015685770660638809:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x21x3xi32>
// CHECK: %[[VAL_13:.*]] = tosa.rescale %[[VAL_12]], %[[VAL_7]], %[[VAL_6]], %[[VAL_11]], %[[VAL_11]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x21x3xi32>
// CHECK: %[[VAL_14:.*]] = tosa.rescale %[[VAL_1]], %[[VAL_8]], %[[VAL_5]], %[[VAL_10]], %[[VAL_11]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686184167861938:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_15:.*]] = tosa.sub %[[VAL_13]], %[[VAL_14]] : (tensor<1x21x3xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_16:.*]] = tosa.rescale %[[VAL_15]], %[[VAL_4]], %[[VAL_3]], %[[VAL_11]], %[[VAL_2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031294636428356171>>
func.func @test_sub_qi8(%arg0: tensor<1x21x3x!quant.uniform<i8:f32, 0.015685770660638809:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686184167861938:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031294636428356171>> {
  %0 = tfl.sub(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x21x3x!quant.uniform<i8:f32, 0.015685770660638809:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015686184167861938:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.031294636428356171>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.031294636428356171>>
}

// -----

// CHECK-LABEL: test_mul_qi8
// CHECK-DAG: %[[shift35:.*]] = "tosa.const"() <{values = dense<35> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: %[[mult1075664768:.*]] = "tosa.const"() <{values = dense<1075664768> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-DAG: %[[const0:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: %[[mult1073741824:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-DAG: %[[shift30:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK-DAG: %[[VAR0:.*]] = tosa.rescale %arg0, %[[mult1073741824]], %[[shift30]]
// CHECK-DAG: %[[VAR1:.*]] = tosa.rescale %arg1, %[[mult1073741824]], %[[shift30]]
// CHECK: %[[VAR2:.*]] = tosa.mul %[[VAR0]], %[[VAR1]], %[[const0]]
// CHECK: %[[VAR3:.*]] = tosa.rescale %[[VAR2]], %[[mult1075664768]], %[[shift35]]
func.func @test_mul_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
}

// -----

// CHECK-LABEL: test_avg_pool2d_qi8
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAR0:.*]] = tosa.avg_pool2d %arg0, %[[ZP]], %[[ZP]] {acc_type = i32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK-SAME: -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
func.func @test_avg_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
}

// -----

// CHECK-LABEL: test_avg_pool2d_i16
// CHECK-DAG: %[[ZP:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
// CHECK: %[[VAR0:.*]] = tosa.avg_pool2d %arg0, %[[ZP]], %[[ZP]] {acc_type = i32, kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK-SAME: -> tensor<1x32x32x8xi16>
func.func @test_avg_pool2d_i16(%arg0: tensor<1x32x32x8xi16>) -> tensor<*xi16> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xi16>) -> tensor<*xi16>
  func.return %0 : tensor<*xi16>
}

// -----

// CHECK-LABEL: test_max_pool2d_qi8
// CHECK: %[[VAR0:.*]] = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
}

// -----

// CHECK-LABEL: test_softmax_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<5> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<31> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<-1010580540> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<1515870810> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<536870912> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<4> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<13> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<7> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_12:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_13:.*]] = "tosa.const"() <{values = dense<9> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_14:.*]] = "tosa.const"() <{values = dense<17> : tensor<1x1x1xi32>}>
// CHECK-DAG: %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}>
// CHECK-DAG: %[[VAL_16:.*]] = "tosa.const"() <{values = dense<23> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_17:.*]] = "tosa.const"() <{values = dense<"0x5{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_18:.*]] = "tosa.const"() <{values = dense<"0xE{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_19:.*]] = "tosa.const"() <{values = dense<"0x4{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_20:.*]] = "tosa.const"() <{values = dense<"0x0{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_21:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_22:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_23:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_24:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_25:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
// CHECK-DAG: %[[VAL_26:.*]] = tosa.reduce_max %[[VAL_25]]
// CHECK-DAG: %[[VAL_27:.*]] = tosa.sub %[[VAL_25]], %[[VAL_26]]
// CHECK-DAG: %[[VAL_28:.*]] = tosa.rescale %[[VAL_27]], %[[VAL_21]], %[[VAL_16]], %[[VAL_24]], %[[VAL_15]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<13x21x3x!quant.uniform<i16:f32, 1.000000e+00>>
// CHECK-DAG: %[[VAL_29:.*]] = tosa.table %[[VAL_28]], %[[VAL_20]]
// CHECK-DAG: %[[VAL_30:.*]] = tosa.table %[[VAL_28]], %[[VAL_19]]
// CHECK-DAG: %[[VAL_31:.*]] = tosa.table %[[VAL_28]], %[[VAL_18]]
// CHECK-DAG: %[[VAL_32:.*]] = tosa.table %[[VAL_28]], %[[VAL_17]]
// CHECK-DAG: %[[VAL_33:.*]] = tosa.logical_left_shift %[[VAL_29]], %[[VAL_14]]
// CHECK-DAG: %[[VAL_34:.*]] = tosa.logical_left_shift %[[VAL_30]], %[[VAL_13]]
// CHECK-DAG: %[[VAL_35:.*]] = tosa.logical_left_shift %[[VAL_31]], %[[VAL_12]]
// CHECK-DAG: %[[VAL_36:.*]] = tosa.arithmetic_right_shift %[[VAL_32]], %[[VAL_11]]
// CHECK-DAG: %[[VAL_37:.*]] = tosa.add %[[VAL_33]], %[[VAL_34]]
// CHECK-DAG: %[[VAL_38:.*]] = tosa.add %[[VAL_37]], %[[VAL_35]]
// CHECK-DAG: %[[VAL_39:.*]] = tosa.add %[[VAL_38]], %[[VAL_36]]
// CHECK-DAG: %[[VAL_40:.*]] = tosa.arithmetic_right_shift %[[VAL_39]], %[[VAL_10]]
// CHECK-DAG: %[[VAL_41:.*]] = tosa.reduce_sum %[[VAL_40]]
// CHECK-DAG: %[[VAL_42:.*]] = tosa.clz %[[VAL_41]]
// CHECK-DAG: %[[VAL_43:.*]] = tosa.sub %[[VAL_42]], %[[VAL_12]]
// CHECK-DAG: %[[VAL_44:.*]] = tosa.logical_left_shift %[[VAL_41]], %[[VAL_43]]
// CHECK-DAG: %[[VAL_45:.*]] = tosa.mul %[[VAL_44]], %[[VAL_6]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_46:.*]] = tosa.add %[[VAL_45]], %[[VAL_7]]
// CHECK-DAG: %[[VAL_47:.*]] = tosa.mul %[[VAL_46]], %[[VAL_44]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_48:.*]] = tosa.sub %[[VAL_8]], %[[VAL_47]]
// CHECK-DAG: %[[VAL_49:.*]] = tosa.mul %[[VAL_46]], %[[VAL_48]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_50:.*]] = tosa.mul %[[VAL_49]], %[[VAL_9]], %[[VAL_4]]
// CHECK-DAG: %[[VAL_51:.*]] = tosa.add %[[VAL_46]], %[[VAL_50]]
// CHECK-DAG: %[[VAL_52:.*]] = tosa.mul %[[VAL_51]], %[[VAL_44]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_53:.*]] = tosa.sub %[[VAL_8]], %[[VAL_52]]
// CHECK-DAG: %[[VAL_54:.*]] = tosa.mul %[[VAL_51]], %[[VAL_53]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_55:.*]] = tosa.mul %[[VAL_54]], %[[VAL_9]], %[[VAL_4]]
// CHECK-DAG: %[[VAL_56:.*]] = tosa.add %[[VAL_51]], %[[VAL_55]]
// CHECK-DAG: %[[VAL_57:.*]] = tosa.mul %[[VAL_56]], %[[VAL_44]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_58:.*]] = tosa.sub %[[VAL_8]], %[[VAL_57]]
// CHECK-DAG: %[[VAL_59:.*]] = tosa.mul %[[VAL_56]], %[[VAL_58]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_60:.*]] = tosa.mul %[[VAL_59]], %[[VAL_9]], %[[VAL_4]]
// CHECK-DAG: %[[VAL_61:.*]] = tosa.add %[[VAL_56]], %[[VAL_60]]
// CHECK-DAG: %[[VAL_62:.*]] = tosa.mul %[[VAL_39]], %[[VAL_61]], %[[VAL_22]]
// CHECK-DAG: %[[VAL_63:.*]] = tosa.sub %[[VAL_3]], %[[VAL_42]]
// CHECK-DAG: %[[VAL_64:.*]] = tosa.arithmetic_right_shift %[[VAL_62]], %[[VAL_2]]
// CHECK-DAG: %[[VAL_65:.*]] = tosa.arithmetic_right_shift %[[VAL_64]], %[[VAL_63]]
// CHECK-DAG: %[[VAL_66:.*]] = tosa.rescale %[[VAL_65]], %[[VAL_21]], %[[VAL_22]], %[[VAL_24]], %[[VAL_1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
func.func @test_softmax_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_softmax_qi16
// CHECK-SAME: %[[VAL_0:.*]]: tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<31> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<"0xF{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<32768> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<14> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<1> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<7> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<32767> : tensor<1x1xi32>}>
// CHECK-DAG: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<1717965619> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_12:.*]] = "tosa.const"() <{values = dense<"0x0{{.*}}"> : tensor<513xi16>}>
// CHECK-DAG: %[[VAL_13:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_14:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_15:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}>
// CHECK-DAG: %[[VAL_16:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_17:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK-DAG: %[[VAL_18:.*]] = tosa.reduce_max %[[VAL_17]]
// CHECK-DAG: %[[VAL_19:.*]] = tosa.sub %[[VAL_17]], %[[VAL_18]]
// CHECK-DAG: %[[VAL_20:.*]] = tosa.rescale %[[VAL_19]], %[[VAL_11]], %[[VAL_10]], %[[VAL_16]], %[[VAL_16]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK-DAG: %[[VAL_21:.*]] = tosa.add %[[VAL_20]], %[[VAL_9]]
// CHECK-DAG: %[[VAL_22:.*]] = tosa.cast %[[VAL_21]]
// CHECK-DAG: %[[VAL_23:.*]] = tosa.table %[[VAL_22]], %[[VAL_12]]
// CHECK-DAG: %[[VAL_24:.*]] = tosa.arithmetic_right_shift %[[VAL_23]], %[[VAL_8]]
// CHECK-DAG: %[[VAL_25:.*]] = tosa.reduce_sum %[[VAL_24]]
// CHECK-DAG: %[[VAL_26:.*]] = tosa.clz %[[VAL_25]]
// CHECK-DAG: %[[VAL_27:.*]] = tosa.sub %[[VAL_26]], %[[VAL_7]]
// CHECK-DAG: %[[VAL_28:.*]] = tosa.logical_left_shift %[[VAL_25]], %[[VAL_27]]
// CHECK-DAG: %[[VAL_29:.*]] = tosa.sub %[[VAL_28]], %[[VAL_6]]
// CHECK-DAG: %[[VAL_30:.*]] = tosa.arithmetic_right_shift %[[VAL_29]], %[[VAL_5]]
// CHECK-DAG: %[[VAL_31:.*]] = tosa.sub %[[VAL_30]], %[[VAL_4]]
// CHECK-DAG: %[[VAL_32:.*]] = tosa.cast %[[VAL_31]]
// CHECK-DAG: %[[VAL_33:.*]] = tosa.table %[[VAL_32]], %[[VAL_3]]
// CHECK-DAG: %[[VAL_34:.*]] = tosa.arithmetic_right_shift %[[VAL_33]], %[[VAL_8]]
// CHECK-DAG: %[[VAL_35:.*]] = tosa.mul %[[VAL_34]], %[[VAL_24]], %[[VAL_2]]
// CHECK-DAG: %[[VAL_36:.*]] = tosa.sub %[[VAL_1]], %[[VAL_26]]
// CHECK-DAG: %[[VAL_37:.*]] = tosa.arithmetic_right_shift %[[VAL_35]], %[[VAL_36]]
// CHECK-DAG: %[[VAL_38:.*]] = tosa.rescale %[[VAL_37]], %[[VAL_13]], %[[VAL_14]], %[[VAL_16]], %[[VAL_15]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<14x19xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>)
func.func @test_softmax_qi16(%arg0: tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>) -> tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>> {
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>) -> tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>>
  func.return %0 : tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>>
}

// -----

// CHECK-LABEL: test_sigmoid_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<256xi8>}>
// CHECK: %[[VAR1:.*]] = tosa.table %arg0, %[[VAR0]]
func.func @test_sigmoid_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_tanh_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<256xi8>}>
// CHECK: %[[VAR1:.*]] = tosa.table %arg0, %[[VAR0]]
func.func @test_tanh_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<*x!quant.uniform<i8:f32, 7.812500e-03>> {
  %0 = "tfl.tanh"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<*x!quant.uniform<i8:f32, 7.812500e-03>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 7.812500e-03>>
}

// -----

// CHECK-LABEL: test_relu_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015685949474573135:-1>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<2147471153> : tensor<1xi32>}>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}>
// CHECK: %[[VAL_5:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015685949474573135:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>)
// CHECK: %[[VAL_6:.*]] = tosa.clamp %[[VAL_5]] {max_val = 127 : i8, min_val = -128 : i8} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.0078430203720927238:-128>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078430203720927238:-128>>
func.func @test_relu_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015685949474573135:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078430203720927238:-128>> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015685949474573135:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078430203720927238:-128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0078430203720927238:-128>>
}

// -----

// CHECK-LABEL: test_relu0To1_qi8
// CHECK-DAG: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686025843024254:-1>>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<2147449478> : tensor<1xi32>}>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686025843024254:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>)
// CHECK: %[[VAL_5:.*]] = tosa.clamp %[[VAL_4]] {max_val = 126 : i8, min_val = -128 : i8}
func.func @test_relu0To1_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686025843024254:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>> {
  %0 = "tfl.relu_n1_to_1"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686025843024254:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>>
}

// -----

// CHECK-LABEL: test_relu6_qi8(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>> {
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<2147467328> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>
// CHECK: %[[VAL_6:.*]] = tosa.clamp %[[VAL_5]] {max_val = 127 : i8, min_val = -128 : i8} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>
func.func @test_relu6_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>  {
    %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>
    func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>
}

// -----

// CHECK-LABEL: test_relu6_qu8
// CHECK: %[[VAL_1:.*]] = tosa.rescale %arg0
// CHECK: %[[VAL_2:.*]] = tosa.rescale %[[VAL_1]]
// CHECK: %[[VAL_3:.*]] = tosa.rescale %[[VAL_2]]
// CHECK: %[[VAL_4:.*]] = tosa.clamp %[[VAL_3]] {max_val = 127 : i8, min_val = -128 : i8}
// CHECK: %[[VAL_5:.*]] = tosa.rescale %[[VAL_4]]
// CHECK: %[[VAL_6:.*]] = tosa.rescale %[[VAL_5]]
func.func @test_relu6_qu8(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.015686137601733208:127>>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.0078431284055113792>> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>} : (tensor<13x21x3x!quant.uniform<u8:f32, 0.015686137601733208:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>
  %1 = "tfl.relu6"(%0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015686137601733208:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<13x21x3x!quant.uniform<u8:f32, 0.0078431284055113792>>} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.0078431284055113792:-128>>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.0078431284055113792>>
  func.return %2 : tensor<13x21x3x!quant.uniform<u8:f32, 0.0078431284055113792>>
}

// -----

// CHECK-LABEL: test_leaky_relu_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2037371008> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_7:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK: %[[VAL_8:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_2]], %[[VAL_1]], %[[VAL_5]], %[[VAL_6]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK: %[[VAL_9:.*]] = tosa.maximum %[[VAL_8]], %[[VAL_7]]
// CHECK: %[[VAL_10:.*]] = tosa.rescale %[[VAL_9]], %[[VAL_2]], %[[VAL_1]], %[[VAL_6]], %[[VAL_5]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
func.func @test_leaky_relu_qi8(%arg0: tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>> {
  %0 = "tfl.leaky_relu"(%arg0) {alpha = 0.948724806 : f32} : (tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
}

// -----

// CHECK-LABEL: test_leaky_relu_qi16
// CHECK-SAME: %[[VAL_0:.*]]: tensor<14x19x!quant.uniform<i16:f32, 0.015519863925874233>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1126059648> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_6:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19x!quant.uniform<i16:f32, 0.015519863925874233>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK: %[[VAL_7:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_1]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19x!quant.uniform<i16:f32, 0.015519863925874233>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<14x19xi32>
// CHECK: %[[VAL_8:.*]] = tosa.minimum %[[VAL_7]], %[[VAL_6]]
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_1]], %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<14x19xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>)
func.func @test_leaky_relu_qi16(%arg0: tensor<14x19x!quant.uniform<i16:f32, 0.015519863925874233:0>>) -> tensor<*x!quant.uniform<i16:f32, 0.015519863925874233:0>> {
  %0 = "tfl.leaky_relu"(%arg0) {alpha = 1.048724806 : f32} : (tensor<14x19x!quant.uniform<i16:f32, 0.015519863925874233:0>>) -> tensor<*x!quant.uniform<i16:f32, 0.015519863925874233:0>>
  func.return %0 : tensor<*x!quant.uniform<i16:f32, 0.015519863925874233:0>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[16, 2, 16, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<14> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_8:.*]] = tosa.resize %[[VAL_0]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] {mode = "BILINEAR"} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x640x640x2xi32>
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<1x640x640x2xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
func.func @test_resize_bilinear_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_half_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[16, 2, 16, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<-7> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<7> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "BILINEAR"}
func.func @test_resize_bilinear_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_align_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[1278, 158, 1278, 158]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[CONST0]], %[[CONST0]] {mode = "BILINEAR"}
func.func @test_resize_bilinear_align_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_align_half_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[1278, 158, 1278, 158]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[KM560:.*]] = tosa.const_shape {values = dense<-560> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[KM560]], %[[KM560]] {mode = "BILINEAR"}
func.func @test_resize_bilinear_align_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = true, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[16, 2, 16, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<14> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}


// -----

// CHECK-LABEL: test_resize_nearest_half_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[16, 2, 16, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<15> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_align_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[1278, 158, 1278, 158]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[K639:.*]] = tosa.const_shape {values = dense<639> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[K639]], %[[K639]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_align_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_align_half_qi8
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[1278, 158, 1278, 158]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[K718:.*]] = tosa.const_shape {values = dense<718> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[K718]], %[[K718]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_align_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = true, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_f32_scalar_input
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "BILINEAR"}
func.func @test_resize_bilinear_f32_scalar_input(%arg0: tensor<3x1x1x7xf32>) -> tensor<3x2x2x7xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<3x1x1x7xf32>, tensor<2xi32>) -> tensor<3x2x2x7xf32>
  func.return %1 : tensor<3x2x2x7xf32>
}

// -----

// CHECK-LABEL: test_resize_bilinear_half_qi8_scalar_input
// CHECK-SAME: %[[VAL_0:.*]]: tensor<3x1x1x7x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_8:.*]] = tosa.resize %[[VAL_0]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] {mode = "BILINEAR"} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 1.000000e-01>>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x2x2x7xi32>
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<3x2x2x7xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
func.func @test_resize_bilinear_half_qi8_scalar_input(%arg0: tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>, tensor<2xi32>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
  func.return %1 : tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_align_qi8_scalar_input
// CHECK-SAME: %[[VAL_0:.*]]: tensor<3x1x1x7x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: %[[VAL_8:.*]] = tosa.resize %[[VAL_0]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] {mode = "BILINEAR"} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 1.000000e-01>>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x2x2x7xi32>
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_8]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<3x2x2x7xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>)
func.func @test_resize_bilinear_align_qi8_scalar_input(%arg0: tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>, tensor<2xi32>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
  func.return %1 : tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
}

// -----

// CHECK-LABEL: test_resize_nearest_f32_scalar_input
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAL_1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_f32_scalar_input(%arg0: tensor<3x1x1x7xf32>) -> tensor<3x2x2x7xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<3x1x1x7xf32>, tensor<2xi32>) -> tensor<3x2x2x7xf32>
  func.return %1 : tensor<3x2x2x7xf32>
}

// -----

// CHECK-LABEL: test_resize_nearest_half_qi8_scalar_input
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAL_1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_half_qi8_scalar_input(%arg0: tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>, tensor<2xi32>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
  func.return %1 : tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
}

// -----

// CHECK-LABEL: test_resize_nearest_align_qi8_scalar_input
// CHECK-DAG: %[[SCALE:.*]] = tosa.const_shape {values = dense<[2, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[OFFSET:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[BORDER:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAL_1:.*]] = tosa.resize %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] {mode = "NEAREST_NEIGHBOR"}
func.func @test_resize_nearest_align_qi8_scalar_input(%arg0: tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<3x1x1x7x!quant.uniform<i8:f32, 0.1>>, tensor<2xi32>) -> tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
  func.return %1 : tensor<3x2x2x7x!quant.uniform<i8:f32, 0.1>>
}

// -----

// CHECK-LABEL: test_fullyconnected_qi16
// CHECK: %[[BIAS:.+]] = "tosa.const"() <{values = dense<123> : tensor<3xi48>}> : () -> tensor<3xi48>
// CHECK: tosa.conv2d {{.+}}, %[[BIAS]], %{{.+}} {acc_type = i48, {{.+}}} : {{.+}} -> tensor<1x1x1x3xi48>
func.func @test_fullyconnected_qi16(%input: tensor<1x7x!quant.uniform<i16:f32, 1.0>>, %filter: tensor<3x7x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x3x!quant.uniform<i16:f32, 1.0>> {
  %bias = "tfl.pseudo_qconst"() {qtype = tensor<3x!quant.uniform<i32:f32, 1.0>>, value = dense<123> : tensor<3xi32>} : () -> tensor<3x!quant.uniform<i32:f32, 1.0>>
  %0 = "tfl.fully_connected"(%input, %filter, %bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x7x!quant.uniform<i16:f32, 1.0>>, tensor<3x7x!quant.uniform<i8:f32, 1.0>>, tensor<3x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x3x!quant.uniform<i16:f32, 1.0>>
  return %0 : tensor<1x3x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: test_gather
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 13, 63]> : tensor<3xindex>}
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 49]> : tensor<2xindex>}
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %arg1, %[[VAR11]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.gather %[[VAR4]], %[[VAR5]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.const_shape {values = dense<[7, 7, 21, 3]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[VAR12]]
// CHECK: return %[[VAR7]]
func.func @test_gather(%arg0: tensor<13x21x3xf32>, %arg1: tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_dyn
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, -1, 63]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 49]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %arg1, %[[VAR11]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.gather %[[VAR4]], %[[VAR5]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.const_shape {values = dense<[7, 7, 21, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[VAR12]]
// CHECK: return %[[VAR7]]
func.func @test_gather_dyn(%arg0: tensor<?x21x3xf32>, %arg1 : tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<?x21x3xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}


// -----
// CHECK-LABEL: test_gather_channel_dyn
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 13, -1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 49]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %arg1, %[[VAR11]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.gather %[[VAR4]], %[[VAR5]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.const_shape {values = dense<[7, 7, 21, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[VAR12]]
// CHECK: return %[[VAR7]]
func.func @test_gather_channel_dyn(%arg0: tensor<13x21x?xf32>, %arg1: tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x?xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_indices_dyn
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 13, 63]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %arg1, %[[VAR11]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.gather %[[VAR4]], %[[VAR5]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.const_shape {values = dense<[-1, 7, 21, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[VAR12]]
// CHECK: return %[[VAR7]]
func.func @test_gather_indices_dyn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<?x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<?x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_batch
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 4, 16]> : tensor<3xindex>}
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 3, 4, 4]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}0, 3, 1]]> : tensor<1x3xi32>
// CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.gather %[[VAR1]], %[[VAR0]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR11]]
// CHECK: return %[[VAR3]]
func.func @test_gather_batch(%arg0: tensor<1x4x4x4xi32>) -> tensor<1x3x4x4xi32> {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 3, 1]]> : tensor<1x3xi32>} : () -> tensor<1x3xi32>
  %1 = "tfl.gather"(%arg0, %0) {axis = 1 : i32, batch_dims = 1 : i32} : (tensor<1x4x4x4xi32>, tensor<1x3xi32>) -> tensor<1x3x4x4xi32>
  func.return %1 : tensor<1x3x4x4xi32>
}

// -----

// CHECK-LABEL: test_gather_batch_dyn
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[-1, 4, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAR1:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR2:.*]] = tosa.gather %[[VAR1]], %arg1
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[-1, 3, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR2]], %[[VAR11]]
// CHECK: return %[[VAR3]]
func.func @test_gather_batch_dyn(%arg0: tensor<?x4x4x4xi32>, %arg1: tensor<?x3xi32>) -> tensor<?x3x4x4xi32> {
  %1 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32, batch_dims = 1 : i32} : (tensor<?x4x4x4xi32>, tensor<?x3xi32>) -> tensor<?x3x4x4xi32>
  func.return %1 : tensor<?x3x4x4xi32>
}

// -----
// CHECK-LABEL: test_gather_nd
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape {values = dense<[1, 273, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[CONST1:.*]] = tosa.const_shape {values = dense<[42, 2]> : tensor<2xindex>}
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape {values = dense<[1, 42]> : tensor<2xindex>}
// CHECK-DAG: %[[CONST3:.*]] = tosa.const_shape {values = dense<[6, 7, 3]> : tensor<3xindex>}
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"
// CHECK-DAG: %[[VAR2:.*]] = tosa.reshape %arg0, %[[CONST0]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %arg1, %[[CONST1]]
// CHECK-DAG: %[[VAR5:.*]] = tosa.mul %[[VAR3]], %[[VAR1]], %[[SHIFT]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.reduce_sum %[[VAR5]] {axis = 1 : i32}
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[CONST2]]
// CHECK-DAG: %[[VAR8:.*]] = tosa.gather %[[VAR2]], %[[VAR7]]
// CHECK: %[[VAR9:.*]] = tosa.reshape %[[VAR8]], %[[CONST3]]
func.func @test_gather_nd(%arg0: tensor<13x21x3xf32>, %arg1: tensor<6x7x2xi32>) -> tensor<6x7x3xf32> {
  %1 = "tfl.gather_nd"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<6x7x2xi32>) -> tensor<6x7x3xf32>
  func.return %1 : tensor<6x7x3xf32>
}

// -----
// CHECK-LABEL: test_gather_cast
// CHECK-DAG: %[[VAR1:.*]] = tosa.cast %arg1
// CHECK-DAG: %[[VAR10:.*]] = tosa.const_shape {values = dense<[1, 13, 63]> : tensor<3xindex>}
// CHECK-DAG: %[[VAR2:.*]] = tosa.reshape %arg0, %[[VAR10]]
// CHECK-DAG: %[[VAR11:.*]] = tosa.const_shape {values = dense<[1, 49]> : tensor<2xindex>}
// CHECK-DAG: %[[VAR3:.*]] = tosa.reshape %[[VAR1]], %[[VAR11]]
// CHECK-DAG: %[[VAR4:.*]] = tosa.gather %[[VAR2]], %[[VAR3]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.const_shape {values = dense<[7, 7, 21, 3]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR5:.*]] = tosa.reshape %[[VAR4]], %[[VAR12]]
// CHECK: return %[[VAR5]]
func.func @test_gather_cast(%arg0: tensor<13x21x3xf32>, %arg1: tensor<7x7xi64>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<7x7xi64>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_sparse_to_dense
// CHECK-DAG: %[[CONST0:.*]] = tosa.const_shape {values = dense<[1, -1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[CONST1:.*]] = tosa.const_shape {values = dense<[1, -1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[CONST2:.*]] = tosa.const_shape {values = dense<[1, 48]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<{{\[\[}}48, 1]]> : tensor<1x2xi32>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x48x1xi64>}>
// CHECK-DAG: %[[VAR2:.*]] = tosa.cast %arg0
// CHECK-DAG: %[[VAR4:.*]] = tosa.mul %[[VAR2]], %[[VAR0]], %[[SHIFT]]
// CHECK-DAG: %[[VAR5:.*]] = tosa.reduce_sum %[[VAR4]] {axis = 1 : i32}
// CHECK-DAG: %[[VAR6:.*]] = tosa.reshape %arg1, %[[CONST0]]
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR5]], %[[CONST1]]
// CHECK-DAG: %[[VAR8:.*]] = tosa.scatter %[[VAR1]], %[[VAR7]], %[[VAR6]]
// CHECK-DAG: %[[VAR9:.*]] = tosa.reshape %[[VAR8]], %[[CONST2]]
// CHECK: return %[[VAR9]]
func.func @test_sparse_to_dense(%arg0 : tensor<?x2xi64>, %arg1 : tensor<?xi64>) -> (tensor<1x48xi64>) {
  %0 = arith.constant dense<[1, 48]> : tensor<2xi64>
  %1 = arith.constant dense<-1> : tensor<i64>
  %2 = "tfl.sparse_to_dense"(%arg0, %0, %arg1, %1) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xi64>, tensor<i64>) -> tensor<1x48xi64>
  func.return %2 : tensor<1x48xi64>
}

// -----

// CHECK-LABEL: test_scatter_nd
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x224x512xf32>}>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x2xi32>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.reduce_sum %[[VAR2:.*]] {axis = 1 : i32} : (tensor<1x2xi32>)
// CHECK-DAG: %[[VAR5:.*]] = tosa.scatter %[[VAR1:.*]], %[[VAR3:.*]], %arg0 : (tensor<1x224x512xf32>, tensor<1x1xi32>, tensor<1x1x512xf32>)
// CHECK: return %[[VAR5]]
func.func @test_scatter_nd(%arg0: tensor<1x1x512xf32>) -> tensor<1x224x512xf32> {
  %shape = "tfl.pseudo_const"() <{value = dense<[1, 224, 512]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %indices = "tfl.pseudo_const"() <{value = dense<[[[0, 0]]]> : tensor<1x1x2xi32>}> : () -> tensor<1x1x2xi32>
  %0 = "tfl.scatter_nd"(%indices, %arg0, %shape) : (tensor<1x1x2xi32>, tensor<1x1x512xf32>, tensor<3xi32>) -> tensor<1x224x512xf32>
  func.return %0 : tensor<1x224x512xf32>
}

// -----

// CHECK-LABEL: test_scatter_nd_reshape
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<{{\[\[}}8, 4, 1]]> : tensor<1x3xi32>}> : ()
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x16x4xf32>}> : ()
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() <{values = dense<{{\[\[}}0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3]]> : tensor<8x3xi32>}> : ()
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[NEW_SHAPE:.*]] = tosa.const_shape {values = dense<[1, 8, 4]> : tensor<3xindex>}
// CHECK-DAG: %[[NEW_SHAPE1:.*]] = tosa.const_shape {values = dense<[1, 8]> : tensor<2xindex>}
// CHECK-DAG: %[[NEW_SHAPE2:.*]] = tosa.const_shape {values = dense<[2, 2, 4, 4]> : tensor<4xindex>}
// CHECK-DAG: %[[VAR4:.*]] = tosa.reshape %arg0, %[[NEW_SHAPE]] : (tensor<2x2x2x4xf32>, !tosa.shape<3>)
// CHECK-DAG: %[[VAR5:.*]] = tosa.mul %[[VAR3]], %[[VAR1]], %[[SHIFT]] : (tensor<8x3xi32>, tensor<1x3xi32>, tensor<1xi8>)
// CHECK-DAG: %[[VAR6:.*]] = tosa.reduce_sum %[[VAR5]] {axis = 1 : i32} : (tensor<8x3xi32>)
// CHECK-DAG: %[[VAR7:.*]] = tosa.reshape %[[VAR6]], %[[NEW_SHAPE1]] : (tensor<8x1xi32>, !tosa.shape<2>)
// CHECK-DAG: %[[VAR8:.*]] = tosa.scatter %[[VAR2]], %[[VAR7]], %[[VAR4]] : (tensor<1x16x4xf32>, tensor<1x8xi32>, tensor<1x8x4xf32>)
// CHECK-DAG: %[[VAR9:.*]] = tosa.reshape %[[VAR8]], %[[NEW_SHAPE2]] : (tensor<1x16x4xf32>, !tosa.shape<4>)
// CHECK-DAG: return %[[VAR9]]
func.func @test_scatter_nd_reshape(%arg0: tensor<2x2x2x4xf32>) -> tensor<2x2x4x4xf32> {
  %shape = "tfl.pseudo_const"() <{value = dense<[2, 2, 4, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %indices = "tfl.pseudo_const"() <{value = dense<[[[[0, 0, 0], [0, 0, 1]], [[0, 0, 2], [0, 0, 3]]], [[[1, 0, 0], [1, 0, 1]], [[1, 0, 2], [1, 0, 3]]]]> : tensor<2x2x2x3xi32>}> : () -> tensor<2x2x2x3xi32>
  %0 = "tfl.scatter_nd"(%indices, %arg0, %shape) : (tensor<2x2x2x3xi32>, tensor<2x2x2x4xf32>, tensor<4xi32>) -> tensor<2x2x4x4xf32>
  func.return %0 : tensor<2x2x4x4xf32>
}

// -----

// CHECK-LABEL: test_scatter_nd_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x224x512xi8>}> : () -> tensor<1x224x512x!quant.uniform<i8:f32, 1.000000e-01:-128>>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x2xi32>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.reduce_sum %[[VAR2:.*]] {axis = 1 : i32} : (tensor<1x2xi32>)
// CHECK-DAG: %[[VAR4:.*]] = tosa.scatter %[[VAR1:.*]], %[[VAR3:.*]], %arg0 : (tensor<1x224x512x!quant.uniform<i8:f32, 1.000000e-01:-128>>, tensor<1x1xi32>, tensor<1x1x512x!quant.uniform<i8:f32, 1.000000e-01:-128>>)
// CHECK: return %[[VAR4]]
func.func @test_scatter_nd_qi8(%arg0: tensor<1x1x512x!quant.uniform<i8:f32, 0.1:-128>>) -> tensor<1x224x512x!quant.uniform<i8:f32, 0.1:-128>> {
  %shape = "tfl.pseudo_const"() <{value = dense<[1, 224, 512]> : tensor<3xi32>}> : () -> tensor<3xi32>
  %indices = "tfl.pseudo_const"() <{value = dense<[[[0, 0]]]> : tensor<1x1x2xi32>}> : () -> tensor<1x1x2xi32>
  %0 = "tfl.scatter_nd"(%indices, %arg0, %shape) : (tensor<1x1x2xi32>, tensor<1x1x512x!quant.uniform<i8:f32, 0.1:-128>>, tensor<3xi32>) -> tensor<1x224x512x!quant.uniform<i8:f32, 0.1:-128>>
  func.return %0 : tensor<1x224x512x!quant.uniform<i8:f32, 0.1:-128>>
}

// -----

// CHECK-LABEL: test_scatter_nd_duplicate_indices
// CHECK: tfl.scatter_nd
func.func @test_scatter_nd_duplicate_indices(%arg0: tensor<2x2x2x4xf32>) -> tensor<2x2x4x4xf32> {
  %shape = "tfl.pseudo_const"() <{value = dense<[2, 2, 4, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %indices = "tfl.pseudo_const"() <{value = dense<[[[[0, 0, 0], [0, 0, 1]], [[0, 0, 2], [0, 0, 3]]], [[[1, 0, 0], [1, 0, 0]], [[1, 0, 2], [1, 0, 3]]]]> : tensor<2x2x2x3xi32>}> : () -> tensor<2x2x2x3xi32>
  %0 = "tfl.scatter_nd"(%indices, %arg0, %shape) : (tensor<2x2x2x3xi32>, tensor<2x2x2x4xf32>, tensor<4xi32>) -> tensor<2x2x4x4xf32>
  func.return %0 : tensor<2x2x4x4xf32>
}

// -----

// CHECK-LABEL: @test_arg_max
func.func @test_arg_max(%arg0: tensor<13x21x3xf32>) -> tensor<*xi32> {
  // CHECK: %[[ARGMAX:.+]] = tosa.argmax %arg0 {axis = 1 : i32}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_max"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_arg_max_negative_dim
func.func @test_arg_max_negative_dim(%arg0: tensor<13x21x3xf32>) -> tensor<13x21xi32> {
  // CHECK: %[[ARGMAX:.+]] = tosa.argmax %arg0 {axis = 2 : i32}
  %0 = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_max"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<13x21xi32>
  func.return %1 : tensor<13x21xi32>
}

// -----

// CHECK-LABEL: @test_arg_min_f32
func.func @test_arg_min_f32(%arg0: tensor<13x21x3xf32>) -> tensor<*xi32> {
  // CHECK-DAG: %[[CONST_0:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK-DAG: %[[NEG:.+]] = tosa.negate %arg0, %[[CONST_0]], %[[CONST_0]] : (tensor<13x21x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<13x21x3xf32>
  // CHECK-DAG: tosa.argmax %[[NEG]] {axis = 1 : i32}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_arg_min_i32
func.func @test_arg_min_i32(%arg0: tensor<13x21x3xi32>) -> tensor<*xi32> {
  // CHECK: %[[ONE:.+]] = "tosa.const"() <{values = dense<-1> : tensor<1x1x1xi32>}>
  // CHECK: %[[SUB:.+]] = tosa.sub %[[ONE]], %arg0
  // CHECK: %[[ARGMAX:.+]] = tosa.argmax %[[SUB]] {axis = 1 : i32}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: test_arg_min_ui8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3xui8>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x1x1xi8>}> : () -> tensor<1x1x1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_4]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xui8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[VAL_7:.*]] = tosa.cast %[[VAL_6]] : (tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>) -> tensor<13x21x3xi8>
// CHECK: %[[VAL_8:.*]] = tosa.sub %[[VAL_1]], %[[VAL_7]] : (tensor<1x1x1xi8>, tensor<13x21x3xi8>) -> tensor<13x21x3xi8>
// CHECK: %[[VAL_9:.*]] = tosa.argmax %[[VAL_8]] {axis = 1 : i32} : (tensor<13x21x3xi8>) -> tensor<13x3xi8>
// CHECK: %[[VAL_10:.*]] = tosa.rescale %[[VAL_9]], %[[VAL_4]], %[[VAL_5]], %[[VAL_3]], %[[VAL_2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x3xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[VAL_11:.*]] = tosa.rescale %[[VAL_10]], %[[VAL_4]], %[[VAL_5]], %[[VAL_2]], %[[VAL_3]] {input_unsigned = false, output_unsigned = true, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x3xui8>
// CHECK: %[[VAL_12:.*]] = tensor.cast %[[VAL_11]] : tensor<13x3xui8> to tensor<*xui8>
// CHECK: return %[[VAL_12]] : tensor<*xui8>
func.func @test_arg_min_ui8(%arg0: tensor<13x21x3xui8>) -> tensor<*xui8> {
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xui8>, tensor<i32>) -> tensor<*xui8>
  func.return %1 : tensor<*xui8>
}

// -----

// CHECK-LABEL: test_fakequant
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<-2.00003052> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<1.99996948> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<6.10360876E-5> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() <{values = dense<16383.75> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR6:.*]] = tosa.minimum %arg0, %[[VAR1]]
// CHECK-DAG: %[[VAR8:.*]] = tosa.maximum %[[VAR6]], %[[VAR0]]
// CHECK-DAG: %[[VAR10:.*]] = tosa.sub %[[VAR8]], %[[VAR0]]
// CHECK-DAG: %[[VAR12:.*]] = tosa.mul %[[VAR10]], %[[VAR3]], %[[SHIFT]]
// CHECK-DAG: %[[VAR14:.*]] = tosa.add %[[VAR12]], %[[VAR4]]
// CHECK-DAG: %[[VAR15:.*]] = tosa.floor %[[VAR14]]
// CHECK-DAG: %[[VAR17:.*]] = tosa.mul %[[VAR15]], %[[VAR2]], %[[SHIFT]]
// CHECK: %[[VAR19:.*]] = tosa.add %[[VAR17]], %[[VAR0]]
func.func @test_fakequant(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %2 = "tfl.fake_quant"(%arg0)  {max = 2.000000e+00 : f32, min = -2.000000e+00 : f32, narrow_range = false, num_bits = 16 : i32}  : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_conv2d_infer
// CHECK: -> tensor<1x32x32x16xf32>
func.func @test_conv2d_infer(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  // CHECK: tosa.add
  // CHECK: tosa.conv2d
  %0 = "tfl.add"(%arg1, %arg1) { fused_activation_function = "NONE" } : (tensor<16x2x2x8xf32>, tensor<16x2x2x8xf32>) -> tensor<*xf32>
  %1 = "tfl.conv_2d"(%arg0, %0, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_conv2d_no_bias
func.func @test_conv2d_no_bias(%input: tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>, %filter: tensor<3x3x8x16x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x32x32x3x!quant.uniform<i16:f32, 1.0>> {
  %bias = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.conv_2d"(%input, %filter, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>, tensor<3x3x8x16x!quant.uniform<i8:f32, 1.0>>, none) -> tensor<1x32x32x3x!quant.uniform<i16:f32, 1.0>>
  return %0 : tensor<1x32x32x3x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_squeeze
func.func @test_squeeze(%arg0: tensor<2x1x3x1xf32>) -> tensor<2x3x1xf32> {
  // CHECK: tosa.reshape
  // CHECK: -> tensor<2x3x1xf32>
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [1]} : (tensor<2x1x3x1xf32>) -> tensor<2x3x1xf32>
  func.return %0 : tensor<2x3x1xf32>
}

// -----

// CHECK-LABEL: @test_squeeze_neg
func.func @test_squeeze_neg(%arg0: tensor<2x1x3x1xf32>) -> tensor<2x1x3xf32> {
  // CHECK: tosa.reshape
  // CHECK: -> tensor<2x1x3xf32>
  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [-1]} : (tensor<2x1x3x1xf32>) -> tensor<2x1x3xf32>
  func.return %0 : tensor<2x1x3xf32>
}

// -----

// CHECK-LABEL: test_gelu
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x8x19xf32>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<3.000000e+00> : tensor<1x1x1x1xf32>}>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<4.471500e-02> : tensor<1x1x1x1xf32>}>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.797884583> : tensor<1x1x1x1xf32>}>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1x1x1xf32>}>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x1x1x1xf32>}>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAL_6:.*]] = tosa.pow %[[VAL_0]], %[[VAL_1]]
// CHECK: %[[VAL_7:.*]] = tosa.mul %[[VAL_6]], %[[VAL_2]], %[[SHIFT]]
// CHECK: %[[VAL_8:.*]] = tosa.add %[[VAL_0]], %[[VAL_7]]
// CHECK: %[[VAL_9:.*]] = tosa.mul %[[VAL_8]], %[[VAL_3]], %[[SHIFT]]
// CHECK: %[[VAL_10:.*]] = tosa.tanh %[[VAL_9]]
// CHECK: %[[VAL_11:.*]] = tosa.add %[[VAL_10]], %[[VAL_4]]
// CHECK: %[[VAL_12:.*]] = tosa.mul %[[VAL_0]], %[[VAL_5]], %[[SHIFT]]
// CHECK: %[[VAL_13:.*]] = tosa.mul %[[VAL_12]], %[[VAL_11]], %[[SHIFT]]
func.func @test_gelu(%arg0: tensor<1x4x8x19xf32>) -> tensor<1x4x8x19xf32> {
  %0 = "tfl.gelu"(%arg0) {approximate = true} : (tensor<1x4x8x19xf32>) -> tensor<1x4x8x19xf32>
  func.return %0 : tensor<1x4x8x19xf32>
}

// -----

// CHECK-LABEL: test_gelu_qi8
// CHECK-SAME: %[[VAR0:.*]]: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>
// CHECK: %[[VAR1:.*]] = "tosa.const"() <{values = dense<{{.*}}> : tensor<256xi8>}>
// CHECK: %[[VAR2:.*]] = tosa.table %[[VAR0]], %[[VAR1]] : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>, tensor<256x!quant.uniform<i8:f32, 1.000000e+00>>
func.func @test_gelu_qi8(%arg0: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>> {
  %0 = "tfl.gelu"(%arg0) {approximate = true} : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
  func.return %0 : tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
}

// -----

// CHECK-LABEL: mirrorpad_reflect
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[0, 7]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[7, 1]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[0, 1]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[7, 2]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[2, 0]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 9]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_7:.*]] = tosa.const_shape  {values = dense<[2, 9]> : tensor<2xindex>}
// CHECK-DAG: %[[VAL_8:.*]] = tosa.const_shape  {values = dense<[1, 0]> : tensor<2xindex>}
// CHECK: %[[VAL_9:.*]] = tosa.slice %arg0, %[[VAL_8]], %[[VAL_7]] : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_10:.*]] = tosa.reverse %[[VAL_9]] {axis = 0 : i32} : (tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_11:.*]] = tosa.slice %arg0, %[[VAL_5]], %[[VAL_6]] : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_12:.*]] = tosa.concat %[[VAL_10]], %arg0, %[[VAL_11]] {axis = 0 : i32} : (tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<1x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_13:.*]] = tosa.slice %[[VAL_12]], %[[VAL_3]], %[[VAL_4]] : (tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_14:.*]] = tosa.reverse %[[VAL_13]] {axis = 1 : i32} : (tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_15:.*]] = tosa.slice %[[VAL_12]], %[[VAL_1]], %[[VAL_2]] : (tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x1x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_16:.*]] = tosa.concat %[[VAL_14]], %[[VAL_12]], %[[VAL_15]] {axis = 1 : i32} : (tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<7x1x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
func.func @mirrorpad_reflect(%arg0: tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>> {
  %0 = "tfl.pseudo_const"() {value = dense<[[2, 1], [2, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tfl.mirror_pad"(%arg0, %0) {mode = #tfl<mirror_pad_attr REFLECT>} : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<2x2xi32>) -> tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
  return %1 : tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
}

// -----

// CHECK-LABEL: mirrorpad_symmetric
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[16, 24, 1]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[16, 1, 2]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 23, 2]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK: %[[VAL_5:.*]] = tosa.slice %arg0, %[[VAL_4]], %[[VAL_3]] : (tensor<15x23x2xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x23x2xf32>
// CHECK: %[[VAL_6:.*]] = tosa.concat %[[VAL_5]], %arg0 {axis = 0 : i32} : (tensor<1x23x2xf32>, tensor<15x23x2xf32>) -> tensor<16x23x2xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %[[VAL_6]], %[[VAL_4]], %[[VAL_2]] : (tensor<16x23x2xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x1x2xf32>
// CHECK: %[[VAL_8:.*]] = tosa.concat %[[VAL_7]], %[[VAL_6]] {axis = 1 : i32} : (tensor<16x1x2xf32>, tensor<16x23x2xf32>) -> tensor<16x24x2xf32>
// CHECK: %[[VAL_9:.*]] = tosa.slice %[[VAL_8]], %[[VAL_4]], %[[VAL_1]] : (tensor<16x24x2xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x24x1xf32>
// CHECK: %[[VAL_10:.*]] = tosa.concat %[[VAL_9]], %[[VAL_8]] {axis = 2 : i32} : (tensor<16x24x1xf32>, tensor<16x24x2xf32>) -> tensor<16x24x3xf32>
func.func @mirrorpad_symmetric(%arg0: tensor<15x23x2xf32>) -> tensor<16x24x3xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<[[1, 0], [1, 0], [1, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tfl.mirror_pad"(%arg0, %0) {mode = #tfl<mirror_pad_attr SYMMETRIC>} : (tensor<15x23x2xf32>, tensor<3x2xi32>) -> tensor<16x24x3xf32>
  return %1 : tensor<16x24x3xf32>
}

// -----

// CHECK-LABEL: @test_reverse_works
func.func @test_reverse_works(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: %[[VAL0:.+]] = tosa.reverse %arg0 {axis = 1 : i32}
  // CHECK: %[[VAL1:.+]] = tosa.reverse %[[VAL0]] {axis = 2 : i32}
  %0 = "tfl.pseudo_const"() {value = dense<[1, -2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.reverse_v2"(%arg0, %0): (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<1x2x3x4xf32>
  func.return %1 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: @test_reverse_fail
func.func @test_reverse_fail(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: "tfl.reverse_v2"
  %0 = "tfl.pseudo_const"() {value = dense<[1, 1111]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.reverse_v2"(%arg0, %0): (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<1x2x3x4xf32>
  func.return %1 : tensor<1x2x3x4xf32>
}

// -----

// CHECK-LABEL: test_tfl_custom
// CHECK-SAME: %[[ARG_0:.*]]: tensor<1x64x64x32xf32>
// CHECK: %[[VAL_0:.*]] = tosa.custom %[[ARG_0]] {domain_name = "TFL", implementation_attrs = "{{.*}}", operator_name = "MaxPoolingWithArgmax2D"} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
func.func @test_tfl_custom(%arg0: tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) {
  // custom op for "tfl.max_pooling_with_argmax_2d"(%arg0) {filter_h = 2 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  %0, %1 = "tfl.custom"(%arg0) {custom_option = #tfl<const_bytes : "0x01000000020000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "MaxPoolingWithArgmax2D"} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  func.return %0, %1 : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>
}

// -----

// CHECK-LABEL: test_tfl_while_loop
// CHECK: %[[VAL_0:.*]]: tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["placeholder_0"]}) -> (tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["output_0"]}) {
// CHECK-DAG: %[[VAL_20:.*]] = tosa.const_shape {values = dense<1> : tensor<1xindex>}
// CHECK-DAG: %[[VAL_21:.*]] = tosa.const_shape {values = dense<> : tensor<0xindex>}
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_2:.*]] = tosa.while_loop (%[[VAL_3:.*]] = %[[VAL_0]]) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
// CHECK:   %[[VAL_4:.*]] = tosa.reduce_sum %[[VAL_3]] {axis = 1 : i32} : (tensor<1x4x4x4xf32>) -> tensor<1x1x4x4xf32>
// CHECK:   %[[VAL_5:.*]] = tosa.reduce_sum %[[VAL_4]] {axis = 2 : i32} : (tensor<1x1x4x4xf32>) -> tensor<1x1x1x4xf32>
// CHECK:   %[[VAL_6:.*]] = tosa.reduce_sum %[[VAL_5]] {axis = 3 : i32} : (tensor<1x1x1x4xf32>) -> tensor<1x1x1x1xf32>
// CHECK:   %[[VAL_7:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_20]] : (tensor<1x1x1x1xf32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:   %[[VAL_8:.*]] = tosa.greater %[[VAL_1]], %[[VAL_7]] : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// CHECK:   %[[VAL_9:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_21]] : (tensor<1xi1>, !tosa.shape<0>) -> tensor<i1>
// CHECK:   tosa.yield %[[VAL_9]] : tensor<i1>
// CHECK:   } do {
// CHECK:   ^bb0(%[[VAL_10:.*]]: tensor<1x4x4x4xf32>):
// CHECK:     %[[VAL_11:.*]] = tosa.sigmoid %[[VAL_10]] : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
// CHECK:     %[[VAL_12:.*]] = tosa.add %[[VAL_10]], %[[VAL_11]] : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
// CHECK:     tosa.yield %[[VAL_12]] : tensor<1x4x4x4xf32>
// CHECK:   }
// CHECK:   return %[[VAL_2]] : tensor<1x4x4x4xf32>
// CHECK: }
func.func @test_tfl_while_loop(%arg0: tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["placeholder_0"]}) -> (tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["output_0"]}) {
  %0 = "tfl.while"(%arg0) ({
  ^bb0(%arg1: tensor<1x4x4x4xf32>):
    %1 = func.call @result_cond(%arg1) : (tensor<1x4x4x4xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg1: tensor<1x4x4x4xf32>):
    %1 = func.call @result_body(%arg1) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    "tfl.yield"(%1) : (tensor<1x4x4x4xf32>) -> ()
  }) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  func.return %0 : tensor<1x4x4x4xf32>
}
func.func private @result_cond(%arg0: tensor<1x4x4x4xf32>) -> tensor<i1> {
  %0 = "tfl.pseudo_const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tfl.sum"(%arg0, %0) {keep_dims = false} : (tensor<1x4x4x4xf32>, tensor<4xi32>) -> tensor<f32>
  %2 = "tfl.pseudo_const"() {value = dense<2.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %3 = tfl.less(%1, %2) : (tensor<f32>, tensor<1xf32>) -> tensor<1xi1>
  %4 = "tfl.pseudo_const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %5 = "tfl.reshape"(%3, %4) : (tensor<1xi1>, tensor<0xi32>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func private @result_body(%arg0: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %0 = "tfl.logistic"(%arg0) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<1x4x4x4xf32>
  func.return %1 : tensor<1x4x4x4xf32>
}

// -----

// CHECK-LABEL: test_rfft2d
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x16xf32>
// CHECK: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 8, 9, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_1:.*]], %[[VAL_2:.*]] = tosa.rfft2d %[[VAL_0]] : (tensor<1x8x16xf32>) -> (tensor<1x8x9xf32>, tensor<1x8x9xf32>)
// CHECK: %[[VAL_3:.*]] = tosa.reshape %[[VAL_1]], %[[VAL_10]] : (tensor<1x8x9xf32>, !tosa.shape<4>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_4:.*]] = tosa.reshape %[[VAL_2]], %[[VAL_10]] : (tensor<1x8x9xf32>, !tosa.shape<4>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_5:.*]] = tosa.concat %[[VAL_3]], %[[VAL_4]] {axis = 3 : i32} : (tensor<1x8x9x1xf32>, tensor<1x8x9x1xf32>) -> tensor<1x8x9x2xf32>
// CHECK: return %[[VAL_5]] : tensor<1x8x9x2xf32>
func.func @test_rfft2d(%arg0: tensor<1x8x16xf32>) -> tensor<1x8x9xcomplex<f32>> {
  %0 = "tfl.pseudo_const"() {value = dense<[8, 16]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.rfft2d"(%arg0, %0) : (tensor<1x8x16xf32>, tensor<2xi32>) -> tensor<1x8x9xcomplex<f32>>
  return %1 : tensor<1x8x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: test_rfft2d_crop_input
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[13, 2, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[13, 2, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[VAL_4:.*]] = tosa.slice %arg0, %[[VAL_3]], %[[VAL_2]] : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x2x2xf32>
// CHECK: %[[VAL_5:.*]], %[[VAL_6:.*]] = tosa.rfft2d %[[VAL_4]] : (tensor<13x2x2xf32>) -> (tensor<13x2x2xf32>, tensor<13x2x2xf32>)
// CHECK: %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_1]] : (tensor<13x2x2xf32>, !tosa.shape<4>) -> tensor<13x2x2x1xf32>
// CHECK: %[[VAL_8:.*]] = tosa.reshape %[[VAL_6]], %[[VAL_1]] : (tensor<13x2x2xf32>, !tosa.shape<4>) -> tensor<13x2x2x1xf32>
// CHECK: %[[VAL_9:.*]] = tosa.concat %[[VAL_7]], %[[VAL_8]] {axis = 3 : i32} : (tensor<13x2x2x1xf32>, tensor<13x2x2x1xf32>) -> tensor<13x2x2x2xf32>
func.func @test_rfft2d_crop_input(%arg0: tensor<13x21x3xf32>) -> tensor<13x2x2xcomplex<f32>> {
  %0 = "tfl.pseudo_const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.rfft2d"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<13x2x2xcomplex<f32>>
  return %1 : tensor<13x2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: test_rfft2d_pad_input
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3xf32>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape {values = dense<[0, 0, 0, 11, 0, 5]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[13, 32, 5, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_3:.*]] = tosa.pad %[[VAL_0]], %[[VAL_2]], %[[VAL_1]] : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<13x32x8xf32>
// CHECK: %[[VAL_4:.*]], %[[VAL_5:.*]] = tosa.rfft2d %[[VAL_3]] : (tensor<13x32x8xf32>) -> (tensor<13x32x5xf32>, tensor<13x32x5xf32>)
// CHECK: %[[VAL_6:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_10]] : (tensor<13x32x5xf32>, !tosa.shape<4>) -> tensor<13x32x5x1xf32>
// CHECK: %[[VAL_7:.*]] = tosa.reshape %[[VAL_5]], %[[VAL_10]] : (tensor<13x32x5xf32>, !tosa.shape<4>) -> tensor<13x32x5x1xf32>
// CHECK: %[[VAL_8:.*]] = tosa.concat %[[VAL_6]], %[[VAL_7]] {axis = 3 : i32} : (tensor<13x32x5x1xf32>, tensor<13x32x5x1xf32>) -> tensor<13x32x5x2xf32>
// CHECK: return %[[VAL_8]] : tensor<13x32x5x2xf32>
func.func @test_rfft2d_pad_input(%arg0: tensor<13x21x3xf32>) -> (tensor<13x32x5xcomplex<f32>>) {
    %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 11], [0, 5]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
    %1 = "tfl.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x32x8xf32>
    %2 = "tfl.pseudo_const"() {value = dense<[32, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tfl.rfft2d"(%1, %2) : (tensor<13x32x8xf32>, tensor<2xi32>) -> tensor<13x32x5xcomplex<f32>>
    return %3 : tensor<13x32x5xcomplex<f32>>
}

// -----

// CHECK-LABEL: test_rfft2d_crop_height_pad_width
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[13, 2, 9, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[13, 2, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 0, 0, 13]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: %[[VAL_6:.*]] = tosa.pad %arg0, %[[VAL_4]], %[[VAL_5]] : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<13x21x16xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %[[VAL_6]], %[[VAL_2]], %[[VAL_3]] : (tensor<13x21x16xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<13x2x16xf32>
// CHECK: %[[VAL_8:.*]], %[[VAL_9:.*]] = tosa.rfft2d %[[VAL_7]] : (tensor<13x2x16xf32>) -> (tensor<13x2x9xf32>, tensor<13x2x9xf32>)
// CHECK: %[[VAL_10:.*]] = tosa.reshape %[[VAL_8]], %[[VAL_1]] : (tensor<13x2x9xf32>, !tosa.shape<4>) -> tensor<13x2x9x1xf32>
// CHECK: %[[VAL_11:.*]] = tosa.reshape %[[VAL_9]], %[[VAL_1]] : (tensor<13x2x9xf32>, !tosa.shape<4>) -> tensor<13x2x9x1xf32>
// CHECK: %[[VAL_12:.*]] = tosa.concat %[[VAL_10]], %[[VAL_11]] {axis = 3 : i32} : (tensor<13x2x9x1xf32>, tensor<13x2x9x1xf32>) -> tensor<13x2x9x2xf32>
func.func @test_rfft2d_crop_height_pad_width(%arg0: tensor<13x21x3xf32>) -> (tensor<13x2x9xcomplex<f32>>) {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [0, 0], [0, 13]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tfl.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x16xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[2, 16]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.rfft2d"(%1, %2) : (tensor<13x21x16xf32>, tensor<2xi32>) -> tensor<13x2x9xcomplex<f32>>
  return %3 : tensor<13x2x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: test_real
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[1, 8, 9]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 8, 9, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_4:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_3]] : (tensor<1x8x9x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_1]] : (tensor<1x8x9x1xf32>, !tosa.shape<3>) -> tensor<1x8x9xf32>
func.func @test_real(%arg0: tensor<1x8x9xcomplex<f32>>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.real"(%arg0) {} : (tensor<1x8x9xcomplex<f32>>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_real_non_complex
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9xf32>
// CHECK: %[[VAL_1:.*]] = tosa.identity %arg0 : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_1]]
func.func @test_real_non_complex(%arg0: tensor<1x8x9xf32>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.real"(%arg0) {} : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_imag
// CHECK-DAG: %[[VAL_1:.*]] = tosa.const_shape  {values = dense<[1, 8, 9]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[0, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 8, 9, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_4:.*]] = tosa.slice %arg0, %[[VAL_2]], %[[VAL_3]] : (tensor<1x8x9x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_5:.*]] = tosa.reshape %[[VAL_4]], %[[VAL_1]] : (tensor<1x8x9x1xf32>, !tosa.shape<3>) -> tensor<1x8x9xf32>
func.func @test_imag(%arg0: tensor<1x8x9xcomplex<f32>>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.imag"(%arg0) {} : (tensor<1x8x9xcomplex<f32>>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_imag_non_complex
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9xf32>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x8x9xf32>}> : () -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_1]] : tensor<1x8x9xf32>
func.func @test_imag_non_complex(%arg0: tensor<1x8x9xf32>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.imag"(%arg0) {} : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_squared_difference_qi8
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = tosa.rescale %arg0
// CHECK-DAG: %[[VAR1:.*]] = tosa.rescale %arg1
// CHECK-DAG: %[[VAR2:.*]] = tosa.rescale %[[VAR0]]
// CHECK-DAG: %[[VAR3:.*]] = tosa.rescale %[[VAR1]]
// CHECK-DAG: %[[VAR4:.*]] = tosa.sub %[[VAR2]], %[[VAR3]]
// CHECK-DAG: %[[VAR5:.*]] = tosa.mul %[[VAR4]], %[[VAR4]], %[[SHIFT]]
// CHECK-DAG: %[[VAR6:.*]] = tosa.rescale %[[VAR5]]
// CHECK: return %[[VAR6]]
func.func @test_squared_difference_qi8(%arg0: tensor<1x197x768x!quant.uniform<i8:f32, 0.13317519426345825:1>>, %arg1: tensor<1x197x1x!quant.uniform<i8:f32, 0.004602269735187292:-4>>) -> tensor<1x197x768x!quant.uniform<i8:f32, 0.9029696583747864:-128>> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1x197x768x!quant.uniform<i8:f32, 0.13317519426345825:1>>, tensor<1x197x1x!quant.uniform<i8:f32, 0.004602269735187292:-4>>) -> tensor<1x197x768x!quant.uniform<i8:f32, 0.9029696583747864:-128>>
  func.return %0 : tensor<1x197x768x!quant.uniform<i8:f32, 0.9029696583747864:-128>>
}

// -----

// CHECK-LABEL: test_squared_difference_f32
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK-DAG: %[[VAR0:.*]] = tosa.sub %arg0, %arg1
// CHECK-DAG: %[[VAR1:.*]] = tosa.mul %[[VAR0]], %[[VAR0]], %[[SHIFT]]
// CHECK: return %[[VAR1]]
func.func @test_squared_difference_f32(%arg0: tensor<1x197x768xf32>, %arg1: tensor<1x197x1xf32>) -> tensor<1x197x768xf32> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1x197x768xf32>, tensor<1x197x1xf32>) -> tensor<1x197x768xf32>
  func.return %0 : tensor<1x197x768xf32>
}

// -----

// CHECK-LABEL: test_squared_difference_with_unequal_ranks_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x304x1x44x!quant.uniform<i8:f32, 0.6395336389541626:-2>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<44x!quant.uniform<i8:f32, 0.0050808675587177277:-16>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<2132442608> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 44]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<1091903658> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_10:.*]] = "tosa.const"() <{values = dense<-16> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_11:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_12:.*]] = "tosa.const"() <{values = dense<23> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_13:.*]] = "tosa.const"() <{values = dense<-2> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_14:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_15:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x304x1x44x!quant.uniform<i8:f32, 0.6395336389541626:-2>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x304x1x44xi32>
// CHECK: %[[VAL_16:.*]] = tosa.rescale %[[VAL_1]], %[[VAL_11]], %[[VAL_12]], %[[VAL_10]], %[[VAL_14]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<44x!quant.uniform<i8:f32, 0.0050808675587177277:-16>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<44xi32>
// CHECK: %[[VAL_17:.*]] = tosa.rescale %[[VAL_15]], %[[VAL_11]], %[[VAL_9]], %[[VAL_14]], %[[VAL_14]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x304x1x44xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x304x1x44xi32>
// CHECK: %[[VAL_18:.*]] = tosa.rescale %[[VAL_16]], %[[VAL_8]], %[[VAL_7]], %[[VAL_14]], %[[VAL_14]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<44xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<44xi32>
// CHECK: %[[VAL_19:.*]] = tosa.reshape %[[VAL_18]], %[[VAL_6]] : (tensor<44xi32>, !tosa.shape<4>) -> tensor<1x1x1x44xi32>
// CHECK: %[[VAL_20:.*]] = tosa.sub %[[VAL_17]], %[[VAL_19]] : (tensor<1x304x1x44xi32>, tensor<1x1x1x44xi32>) -> tensor<1x304x1x44xi32>
// CHECK: %[[VAL_21:.*]] = tosa.mul %[[VAL_20]], %[[VAL_20]], %[[VAL_5]] : (tensor<1x304x1x44xi32>, tensor<1x304x1x44xi32>, tensor<1xi8>) -> tensor<1x304x1x44xi32>
// CHECK: %[[VAL_22:.*]] = tosa.rescale %[[VAL_21]], %[[VAL_4]], %[[VAL_3]], %[[VAL_14]], %[[VAL_2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x304x1x44xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x304x1x44x!quant.uniform<i8:f32, 26.360841751098633:-128>>
func.func @test_squared_difference_with_unequal_ranks_qi8(%arg0: tensor<1x304x1x44x!quant.uniform<i8:f32, 0.6395336389541626:-2>>, %arg1: tensor<44x!quant.uniform<i8:f32, 0.0050808675587177277:-16>>) -> tensor<1x304x1x44x!quant.uniform<i8:f32, 26.360841751098633:-128>> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1x304x1x44x!quant.uniform<i8:f32, 0.6395336389541626:-2>>, tensor<44x!quant.uniform<i8:f32, 0.0050808675587177277:-16>>) -> tensor<1x304x1x44x!quant.uniform<i8:f32, 26.360841751098633:-128>>
  func.return %0 : tensor<1x304x1x44x!quant.uniform<i8:f32, 26.360841751098633:-128>>
}

// -----

// CHECK-LABEL: test_squared_difference_with_unequal_ranks_f32
// CHECK: %[[C:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[CS:.*]] = tosa.const_shape  {values = dense<[1, 1, 1, 44]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[RH:.*]] = tosa.reshape %arg1, %[[CS]] : (tensor<44xf32>, !tosa.shape<4>) -> tensor<1x1x1x44xf32>
// CHECK: %[[SUB:.*]] = tosa.sub %arg0, %[[RH]]
// CHECK: %[[MUL:.*]] = tosa.mul %[[SUB]], %[[SUB]], %[[C]]
// CHECK: return %[[MUL]] : tensor<1x304x1x44xf32>
func.func @test_squared_difference_with_unequal_ranks_f32(%arg0: tensor<1x304x1x44xf32>, %arg1: tensor<44xf32>) -> tensor<1x304x1x44xf32> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1x304x1x44xf32>, tensor<44xf32>) -> tensor<1x304x1x44xf32>
  func.return %0 : tensor<1x304x1x44xf32>
}

// -----

// CHECK-LABEL: test_broadcast_to_f32
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 1, 13, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<-0.000000e+00> : tensor<3x3x13x7xf32>}
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_10]] : (tensor<13x1xf32>, !tosa.shape<4>)
// CHECK: %[[VAL_2:.*]] = tosa.add %[[VAL_1]], %[[VAL_0]] : (tensor<1x1x13x1xf32>, tensor<3x3x13x7xf32>) -> tensor<3x3x13x7xf32>
// CHECK: return %[[VAL_2]] : tensor<3x3x13x7xf32>
func.func @test_broadcast_to_f32(%arg0: tensor<13x1xf32>) -> (tensor<3x3x13x7xf32>) {
  %shape = arith.constant dense<[3, 3, 1, 7]> : tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<13x1xf32>, tensor<4xi32>) -> tensor<3x3x13x7xf32>
  return %1 : tensor<3x3x13x7xf32>
}

// -----

// CHECK-LABEL: test_broadcast_to_f16
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 1, 13, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<-0.000000e+00> : tensor<3x3x13x7xf16>}>
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_10]] : (tensor<13x1xf16>, !tosa.shape<4>)
// CHECK: %[[VAL_2:.*]] = tosa.add %[[VAL_1]], %[[VAL_0]] : (tensor<1x1x13x1xf16>, tensor<3x3x13x7xf16>) -> tensor<3x3x13x7xf16>
// CHECK: return %[[VAL_2]] : tensor<3x3x13x7xf16>
func.func @test_broadcast_to_f16(%arg0: tensor<13x1xf16>) -> (tensor<3x3x13x7xf16>) {
  %shape = arith.constant dense<[3, 3, 1, 7]> : tensor<4xi32>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<13x1xf16>, tensor<4xi32>) -> tensor<3x3x13x7xf16>
  return %1 : tensor<3x3x13x7xf16>
}

// -----

// CHECK-LABEL: test_broadcast_to_i32
// CHECK-DAG: %[[VAL_10]] = tosa.const_shape {values = dense<[1, 1, 13, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<0> : tensor<7x7x13x3xi32>}
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_10]] : (tensor<13x1xi32>, !tosa.shape<4>)
// CHECK: %[[VAL_2:.*]] = tosa.add %[[VAL_1]], %[[VAL_0]] : (tensor<1x1x13x1xi32>, tensor<7x7x13x3xi32>) -> tensor<7x7x13x3xi32>
// CHECK: return %[[VAL_2]] : tensor<7x7x13x3xi32>
func.func @test_broadcast_to_i32(%arg0: tensor<13x1xi32>) -> (tensor<3x3x13x3xi32>) {
  %shape = arith.constant dense<[7, 7, 13, 3]> : tensor<4xi64>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<13x1xi32>, tensor<4xi64>) -> tensor<3x3x13x3xi32>
  return %1 : tensor<3x3x13x3xi32>
}

// -----

// CHECK-LABEL: test_broadcast_to_i1
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 1, 13, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<false> : tensor<7x7x13x7xi1>}
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_10]] : (tensor<13x1xi1>, !tosa.shape<4>)
// CHECK: %[[VAL_2:.*]] = tosa.logical_or %[[VAL_1]], %[[VAL_0]] : (tensor<1x1x13x1xi1>, tensor<7x7x13x7xi1>) -> tensor<7x7x13x7xi1>
// CHECK: return %[[VAL_2]] : tensor<7x7x13x7xi1>
func.func @test_broadcast_to_i1(%arg0: tensor<13x1xi1>) -> (tensor<7x7x13x7xi1>) {
  %shape = arith.constant dense<[7, 7, 13, 7]> : tensor<4xi64>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<13x1xi1>, tensor<4xi64>) -> tensor<7x7x13x7xi1>
  return %1 : tensor<7x7x13x7xi1>
}

// -----

// CHECK-LABEL: test_broadcast_to_qi8
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 1, 13, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<0> : tensor<7x7x13x3xi32>}
// CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_10]]
// CHECK: %[[VAL_2:.*]] = tosa.cast %2 : (tensor<1x1x13x1x!quant.uniform<i16:f32, 1.000000e+00:-1>>) -> tensor<1x1x13x1xi32>
// CHECK: %[[VAL_3:.*]] = tosa.add %[[VAL_2]], %[[VAL_0]] : (tensor<1x1x13x1xi32>, tensor<7x7x13x3xi32>) -> tensor<7x7x13x3xi32>
// CHECK: %[[VAL_4:.*]] = tosa.cast %4 : (tensor<7x7x13x3xi32>) -> tensor<7x7x13x3x!quant.uniform<i16:f32, 1.000000e+00:-1>>
// CHECK: return %[[VAL_4]] : tensor<7x7x13x3x!quant.uniform<i16:f32, 1.000000e+00:-1>>
func.func @test_broadcast_to_qi8(%arg0: tensor<13x1x!quant.uniform<i16:f32, 1.0:-1>>) -> (tensor<7x7x13x3x!quant.uniform<i16:f32, 1.0:-1>>) {
  %shape = arith.constant dense<[7, 7, 1, 3]> : tensor<4xi64>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<13x1x!quant.uniform<i16:f32, 1.0:-1>>, tensor<4xi64>) -> tensor<7x7x13x3x!quant.uniform<i16:f32, 1.0:-1>>
  return %1 : tensor<7x7x13x3x!quant.uniform<i16:f32, 1.0:-1>>
}

// -----

// CHECK-LABEL: test_broadcast_to_smaller_rank
// CHECK: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<[13, 7]> : tensor<2xi48>}
// CHECK: %[[VAL_1:.*]] = "tfl.broadcast_to"(%arg0, %[[VAL_0]]) : (tensor<2x3x13x1xi32>, tensor<2xi48>) -> tensor<13x7xi32>
// CHECK: return %[[VAL_1]] : tensor<13x7xi32>
func.func @test_broadcast_to_smaller_rank(%arg0: tensor<2x3x13x1xi32>) -> (tensor<13x7xi32>) {
  %shape = arith.constant dense<[13, 7]> : tensor<2xi64>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<2x3x13x1xi32>, tensor<2xi64>) -> tensor<13x7xi32>
  return %1 : tensor<13x7xi32>
}

// -----

// CHECK-LABEL: test_broadcast_to_i48
// CHECK: %[[VAL_0:.*]] = "tosa.const"() <{values = dense<[7, 7, 1, 7]> : tensor<4xi48>}
// CHECK: %[[VAL_1:.*]] = "tfl.broadcast_to"(%arg0, %[[VAL_0]]) : (tensor<1x1x13x1xi48>, tensor<4xi48>) -> tensor<7x7x13x7xi48>
// CHECK: return %[[VAL_1]] : tensor<7x7x13x7xi48>
func.func @test_broadcast_to_i48(%arg0: tensor<1x1x13x1xi48>) -> (tensor<7x7x13x7xi48>) {
  %shape = arith.constant dense<[7, 7, 1, 7]> : tensor<4xi64>
  %1 = "tfl.broadcast_to"(%arg0, %shape) : (tensor<1x1x13x1xi48>, tensor<4xi64>) -> tensor<7x7x13x7xi48>
  return %1 : tensor<7x7x13x7xi48>
}

// -----

// CHECK-LABEL: test_transpose_conv2d_bias_f32
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<128x2x2x256xf32>}> : () -> tensor<128x2x2x256xf32>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.transpose_conv2d %arg0, %[[VAR1]], %[[VAR0]], %[[VAR2]], %[[VAR2]] {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}
func.func @test_transpose_conv2d_bias_f32(%arg0: tensor<1x64x64x256xf32>) -> tensor<1x128x128x128xf32> {
  %cst = arith.constant dense<[1, 128, 128, 128]> : tensor<4xi32>
  %0 = arith.constant dense<-1.000000e+00> : tensor<128x2x2x256xf32>
  %1 = arith.constant dense<1.000000e+00> : tensor<128xf32>
  %2 = "tfl.transpose_conv"(%cst, %0, %arg0, %1)  {padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"}  : (tensor<4xi32>, tensor<128x2x2x256xf32>, tensor<1x64x64x256xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
  return %2 : tensor<1x128x128x128xf32>
}

// -----

// CHECK-LABEL: test_mul_with_unequal_ranks
// CHECK-DAG: %[[VAL_10:.*]] = tosa.const_shape {values = dense<[1, 1, 1, 384]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: %[[VAR0:.*]] = tosa.reshape %arg1, %[[VAL_10]] : (tensor<384xf32>, !tosa.shape<4>) -> tensor<1x1x1x384xf32>
// CHECK: %[[VAR1:.*]] = tosa.mul %arg0, %[[VAR0]], %[[SHIFT]] : (tensor<?x135x240x384xf32>, tensor<1x1x1x384xf32>, tensor<1xi8>)
func.func @test_mul_with_unequal_ranks(%arg0: tensor<?x135x240x384xf32>, %arg1: tensor<384xf32>) -> tensor<?x135x240x384xf32> {
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<?x135x240x384xf32>, tensor<384xf32>) -> tensor<?x135x240x384xf32>
  func.return %0 : tensor<?x135x240x384xf32>
}

// -----

// CHECK-LABEL: test_mul_with_unequal_ranks_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x192x192x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1077952640> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<127> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
// CHECK: %[[VAL_6:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_7:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_8:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_9:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_10:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x192x192x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x192x192x3xi32>
// CHECK: %[[VAL_11:.*]] = tosa.rescale %[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<i32>
// CHECK: %[[VAL_12:.*]] = tosa.reshape %[[VAL_11]], %[[VAL_4]] : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK: %[[VAL_13:.*]] = tosa.mul %[[VAL_10]], %[[VAL_12]], %[[VAL_3]] : (tensor<1x192x192x3xi32>, tensor<1x1x1x1xi32>, tensor<1xi8>) -> tensor<1x192x192x3xi32>
// CHECK: %[[VAL_14:.*]] = tosa.rescale %[[VAL_13]], %[[VAL_2]], %[[VAL_1]], %[[VAL_9]], %[[VAL_8]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<1x192x192x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>>
func.func @test_mul_with_unequal_ranks_qi8(%arg0: tensor<1x192x192x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>) -> tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>> {
    %0 = "tfl.pseudo_qconst"() {qtype = tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
    %1 = tfl.mul(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<1x192x192x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>>
    func.return %1 : tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>>
}


// -----

// CHECK-LABEL: test_sub_with_unequal_ranks_qi8
// CHECK: %[[CS:.*]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK: %[[C:.*]] = "tosa.const"() <{values = dense<127> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 0.0039215688593685627:-128>>
// CHECK: %[[RS1:.*]] = tosa.rescale %arg0
// CHECK: %[[RS2:.*]] = tosa.rescale %[[C]]
// CHECK: %[[RS3:.*]] = tosa.rescale %[[RS2]]
// CHECK: %[[RH:.*]] = tosa.reshape %[[RS3]], %[[CS]] : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
// CHECK: %[[SUB:.*]] = tosa.sub %[[RS1]], %[[RH]] : (tensor<1x192x192x3xi32>, tensor<1x1x1x1xi32>) -> tensor<1x192x192x3xi32>
// CHECK: %[[RS4:.*]] = tosa.rescale %[[SUB]]
// CHECK: return %[[RS4]] : tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>>
func.func @test_sub_with_unequal_ranks_qi8(%arg0: tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>>) -> tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>> {
    %0 = "tfl.pseudo_qconst"() {qtype = tensor<!quant.uniform<i8:f32, 0.0039215688593685627:-128>>, value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 0.0039215688593685627:-128>>
    %1 = tfl.sub(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-128>>, tensor<!quant.uniform<i8:f32, 0.0039215688593685627:-128>>) -> tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>>
    func.return %1 : tensor<1x192x192x3x!quant.uniform<i8:f32, 0.0078431377187371254:-1>>
}

// -----

// CHECK-LABEL: test_add_with_unequal_ranks_qi8
// CHECK: %[[CS:.*]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: %[[C:.*]] = "tosa.const"() <{values = dense<127> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 0.0070588234812021255:-128>>
// CHECK: %[[RS1:.*]] = tosa.rescale %arg0
// CHECK: %[[RS2:.*]] = tosa.rescale %[[C]]
// CHECK: %[[RS3:.*]] = tosa.rescale %[[RS2]]
// CHECK: %[[RH:.*]] = tosa.reshape %[[RS3]], %[[CS]] : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
// CHECK: %[[ADD:.*]] = tosa.add %[[RS1]], %[[RH]] : (tensor<48x48x17xi32>, tensor<1x1x1xi32>) -> tensor<48x48x17xi32>
// CHECK: %[[RS4:.*]] = tosa.rescale %[[ADD]]
// CHECK: return %[[RS4]] : tensor<48x48x17x!quant.uniform<i8:f32, 0.24958714842796326:-128>>
func.func @test_add_with_unequal_ranks_qi8(%arg0: tensor<48x48x17x!quant.uniform<i8:f32, 0.24252831935882568:-128>>) -> tensor<48x48x17x!quant.uniform<i8:f32, 0.24958714842796326:-128>> {
    %0 = "tfl.pseudo_qconst"() {qtype = tensor<!quant.uniform<i8:f32, 0.0070588234812021255:-128>>, value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 0.0070588234812021255:-128>>
    %1 = tfl.add(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<48x48x17x!quant.uniform<i8:f32, 0.24252831935882568:-128>>, tensor<!quant.uniform<i8:f32, 0.0070588234812021255:-128>>) -> tensor<48x48x17x!quant.uniform<i8:f32, 0.24958714842796326:-128>>
    func.return %1 : tensor<48x48x17x!quant.uniform<i8:f32, 0.24958714842796326:-128>>
}

// -----

// CHECK-LABEL: test_cast_ui8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3xui8>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_6:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] {input_unsigned = true, output_unsigned = false, per_channel = false, rounding_mode = "SINGLE_ROUND", scale32 = true} : (tensor<13x21x3xui8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>
// CHECK: %[[VAL_7:.*]] = tosa.rescale %[[VAL_6]], %[[VAL_2]], %[[VAL_3]], %[[VAL_5]], %[[VAL_1]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_8:.*]] = tosa.cast %[[VAL_7]] : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAL_8]] : tensor<13x21x3xf32>
func.func @test_cast_ui8(%arg0: tensor<13x21x3xui8>) -> (tensor<13x21x3xf32>) {
  %0 = "tfl.cast"(%arg0) : (tensor<13x21x3xui8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_cast_qi8
// CHECK-SAME: %[[VAL_0:.*]]: tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-1>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_5:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<13x21x3x!quant.uniform<i8:f32, 1.000000e+00:-1>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_6:.*]] = tosa.cast %[[VAL_5]] : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
func.func @test_cast_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 1.0:-1>>) -> (tensor<13x21x3xf32>) {
  %0 = "tfl.cast"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 1.0:-1>>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d_bias_f32
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<128x2x2x256xf32>}> : () -> tensor<128x2x2x256xf32>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK-DAG: %[[VAR3:.*]] = tosa.transpose_conv2d %arg0, %[[VAR1]], %[[VAR0]], %[[VAR2]], %[[VAR2]] {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}
func.func @test_transpose_conv2d_bias_f32(%arg0: tensor<1x64x64x256xf32>) -> tensor<1x128x128x128xf32> {
  %cst = arith.constant dense<[1, 128, 128, 128]> : tensor<4xi32>
  %0 = arith.constant dense<-1.000000e+00> : tensor<128x2x2x256xf32>
  %1 = arith.constant dense<1.000000e+00> : tensor<128xf32>
  %2 = "tfl.transpose_conv"(%cst, %0, %arg0, %1)  {padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32, fused_activation_function = "NONE"}  : (tensor<4xi32>, tensor<128x2x2x256xf32>, tensor<1x64x64x256xf32>, tensor<128xf32>) -> tensor<1x128x128x128xf32>
  return %2 : tensor<1x128x128x128xf32>
}
