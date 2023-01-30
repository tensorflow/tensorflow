// RUN: tf-opt --split-input-file --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s
// RUN: tf-opt --split-input-file --tf-tfl-to-tosa-pipeline --verify-each %s | FileCheck %s

// Operations for testing tfl-to-tosa-pipeline

// TODO: For all quantized tests: compute and add checks on rescale attribute
// values
// TODO: These tests are fairly minimal. Expand the checks to be more robust.


// -----

// CHECK-LABEL: test_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.conv2d"(%arg0, %arg1, %[[VAR0]]) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
func.func @test_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x2x2x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv2d_dynamic
// CHECK: "tosa.conv2d"
// CHECK-SAME: tensor<?x32x32x16xf32>
func.func @test_conv2d_dynamic(%arg0: tensor<?x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_conv2d_bias
// CHECK: %[[VAR0:.*]] = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>}
// CHECK-SAME: tensor<1x32x32x16xf32>
func.func @test_conv2d_bias(%arg0: tensor<1x32x32x8xf32>, %cst: tensor<16x2x2x8xf32>, %cst_0: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "tfl.conv_2d"(%arg0, %cst, %cst_0)  {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>, tensor<16x2x2x8xf32>, tensor<16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose_conv2d"(%arg0, %arg1, %[[VAR0]]) {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>}
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = arith.constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_transpose_conv2d_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose_conv2d"(%arg0, %arg1, %[[VAR0]]) {out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR2:.*]] = "tosa.clamp"(%[[VAR1]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_transpose_conv2d_relu(%arg0: tensor<1x32x32x8xf32>, %cst_0: tensor<16x1x1x8xf32>) -> tensor<1x32x32x16xf32> {
  %cst = arith.constant dense<[1, 32, 32, 16]> : tensor<4xi32>
  %cst_1 = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%cst, %cst_0, %arg0, %cst_1)  {padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU"}  : (tensor<4xi32>, tensor<16x1x1x8xf32>, tensor<1x32x32x8xf32>, none) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: test_conv2d_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<16x2x2x8xi8>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0> : tensor<16xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.conv2d"(%arg0, %[[VAR0]], %[[VAR1]]) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func.func @test_conv2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<42> : tensor<16x2x2x8xi8>} : () -> tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0,  {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4}>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684768557548523>>, tensor<16x2x2x8x!quant.uniform<i8<-127:127>:f32:0, {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0, {2.0,2.0,1.0,1.0,1.0,2.0,2.4,1.7,2.3,2.4,2.4,2.3,2.1,2.4,2.1,2.4} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
}

// -----

// CHECK-LABEL: test_conv2d_qi16
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0> : tensor<16xi48>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<16x1x1x8xi8>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.conv2d"(%arg0, %[[VAR1]], %[[VAR0]]) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>, stride = array<i64: 1, 1>}
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func.func @test_conv2d_qi16(%arg0: tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "arith.constant"() {value = dense<0> : tensor<16xi64>} : () -> tensor<16xi64>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16xi64>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_depthwise_conv2d_bias_qi8
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>
// CHECK-DAG:     %[[CONST:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<16xi32>}
// CHECK-DAG:     %[[CONST_0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<1x2x2x16xi8>}
// CHECK-DAG:     %[[RESHAPE:.*]] = "tosa.reshape"(%[[CONST_0]]) {new_shape = array<i64: 2, 2, 8, 2>}
// CHECK-DAG:     %[[DEPTHWISE:.*]] = "tosa.depthwise_conv2d"(%[[ARG0]], %[[RESHAPE]], %[[CONST]]) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, quantization_info = #tosa.conv_quant<input_zp = -1, weight_zp = 0>, stride = array<i64: 1, 1>}
// CHECK:         %[[RESCALE:.*]] = "tosa.rescale"(%[[DEPTHWISE]])
// CHECK-SAME:        multiplier = array<i32: 1373724854, 1373724854, 1373724854, 1373724854, 1803013871, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854, 1373724854>
// CHECK-SAME:        shift = array<i32: 36, 36, 36, 36, 32, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36>
// CHECK:         return %[[RESCALE]]
func.func @test_depthwise_conv2d_bias_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127], [-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]], [[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127], [-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x2x2x16xi8>} : () -> tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3,  {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1} >>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32:0, {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5}>>, value = dense<[-2879, 6636, 3531, 23376, -79787, -6142, 5582, -30384, 17330, -4549, -3518, 16215, 2695, -2670, 8399, -12223]> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32:0,  {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %0, %1) {depth_multiplier = 2 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015678688883781433:-1>>, tensor<1x2x2x16x!quant.uniform<i8<-127:127>:f32:3, {0.1,0.1,0.1,0.1,2.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}>>, tensor<16x!quant.uniform<i32:f32:0,   {9.1E-5,1.9E-4,2.3E-4,4.5E-5,3.6E-6,2.3E-4,2.3E-4,5.6E-5,5.8E-5,1.7E-4,7.1E-5,7.3E-5,2.2E-4,1.5E-4,1.7E-4,7.3E-5} >>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 0.078431375324726104>>
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

// CHECK-LABEL: test_conv3d
// CHECK-SAME: %[[VAL_0:.*]]: tensor<2x2x7x7x2xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x2x4xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<4xf32>}
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]> : tensor<5xi32>}
// CHECK: %[[VAL_4:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.conv3d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_2]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d(%arg0: tensor<2x2x7x7x2xf32>, %arg1: tensor<2x3x3x2x4xf32>) -> tensor<2x2x7x7x4xf32> {
  %cst = "tfl.no_value"() {value} : () -> none
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<2x2x7x7x2xf32>, tensor<2x3x3x2x4xf32>, none) -> tensor<2x2x7x7x4xf32>
  func.return %0 : tensor<2x2x7x7x4xf32>
}

// -----

// CHECK-LABEL: test_conv3d_dynamic
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x11x32x32x8xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<3x1x1x8x16xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<16xf32>}
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]> : tensor<5xi32>}
// CHECK: %[[VAL_4:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.conv3d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_2]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 1, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>}
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
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]>
// CHECK: %[[VAL_4:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.conv3d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_2]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 7, 8, 0, 1, 0, 1>, stride = array<i64: 1, 1, 1>}
func.func @test_conv3d_bias(%arg0: tensor<10x3x64x64x12xf32>, %arg1: tensor<16x2x2x12x8xf32>, %cst: tensor<8xf32>) -> tensor<10x3x64x64x8xf32> {
  %0 = "tfl.conv_3d"(%arg0, %arg1, %cst) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<10x3x64x64x12xf32>, tensor<16x2x2x12x8xf32>, tensor<8xf32>) -> tensor<10x3x64x64x8xf32>
  func.return %0 : tensor<10x3x64x64x8xf32>
}

// -----

// CHECK-LABEL: test_conv3d_qi8(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<2x3x3x17x34xf32>) -> tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<0.0156862643> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() {value = dense<1.11982894> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() {value = dense<-4.000000e+00> : tensor<1x1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<34xf32>}
// CHECK-DAG: %[[VAL_7:.*]] = "tosa.const"() {value = dense<[4, 0, 1, 2, 3]> : tensor<5xi32>}
// CHECK: %[[VAL_8:.*]] = "tosa.cast"(%[[VAL_0]])
// CHECK: %[[VAL_9:.*]] = "tosa.sub"(%[[VAL_8]], %[[VAL_2]])
// CHECK: %[[VAL_10:.*]] = "tosa.mul"(%[[VAL_9]], %[[VAL_3]]) {shift = 0 : i32}
// CHECK: %[[VAL_11:.*]] = "tosa.transpose"(%[[VAL_1]], %[[VAL_7]])
// CHECK: %[[VAL_12:.*]] = "tosa.conv3d"(%[[VAL_10]], %[[VAL_11]], %[[VAL_6]]) {dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 1, 1, 1, 1, 1>, stride = array<i64: 1, 1, 2>}
// CHECK: %[[VAL_13:.*]] = "tosa.mul"(%[[VAL_12]], %[[VAL_4]]) {shift = 0 : i32}
// CHECK: %[[VAL_14:.*]] = "tosa.add"(%[[VAL_13]], %[[VAL_5]])
// CHECK: %[[VAL_15:.*]] = "tosa.cast"(%[[VAL_14]])
func.func @test_conv3d_qi8(%arg0: tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>, %arg1: tensor<2x3x3x17x34xf32>) -> (tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x4x8x21x17x!quant.uniform<i8:f32, 0.015686264261603355>>) -> tensor<1x4x8x21x17xf32>
  %2 = "tfl.no_value"() {value} : () -> none
  %3 = "tfl.conv_3d"(%0, %arg1, %2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 2 : i32} : (tensor<1x4x8x21x17xf32>, tensor<2x3x3x17x34xf32>, none) -> tensor<1x4x8x11x34xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>} : (tensor<1x4x8x11x34xf32>) -> tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
  return %4 : tensor<1x4x8x11x34x!quant.uniform<i8:f32, 0.8929935097694397:-4>>
}

// -----

// CHECK-LABEL: test_add
// CHECK: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_add_unranked
// CHECK: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
func.func @test_add_unranked(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_sub
// CHECK: %[[VAR0:.*]] = "tosa.sub"(%arg0, %arg1)
func.func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_sub_unranked
// CHECK: %[[VAR0:.*]] = "tosa.sub"(%arg0, %arg1)
func.func @test_sub_unranked(%arg0: tensor<1x21x3xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<1x21x3xf32>, tensor<1x1x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_mul
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32}
func.func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_mul_unranked
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32}
func.func @test_mul_unranked(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_exp
// CHECK: %[[VAR0:.*]] = "tosa.exp"(%arg0)
func.func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_rcp
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reciprocal"(%arg0)
func.func @test_rcp(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.div"(%cst, %arg0)  {fused_activation_function = "NONE"}  : (tensor<f32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_div
// CHECK-DAG: %[[RESHAPE:.*]] = "tosa.reshape"(%arg1)
// CHECK: %[[VAR0:.*]] = "tosa.div"(%arg0, %[[RESHAPE]])
func.func @test_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "tfl.div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: test_floor_div
// CHECK-DAG: %[[RESHAPE:.*]] = "tosa.reshape"(%arg1)
// CHECK: %[[VAR0:.*]] = "tosa.div"(%arg0, %[[RESHAPE]])
func.func @test_floor_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "tfl.floor_div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: test_relu1
// CHECK: %[[VAL0:.*]] = "tosa.clamp"(%arg0) {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = -1.000000e+00 : f32, min_int = -1 : i64}
func.func @test_relu1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu_n1_to_1"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu0To1
// CHECK: %[[VAL0:.*]] = "tosa.clamp"(%arg0) {max_fp = 1.000000e+00 : f32, max_int = 1 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu0To1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.relu_0_to_1"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_relu6
// CHECK: %[[VAR0:.*]] = "tosa.clamp"(%arg0) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu6(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_relu6_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.clamp"(%arg0) {max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
// CHECK-SAME: -> tensor<?x21x3xf32>
func.func @test_relu6_dynamic(%arg0: tensor<?x21x3xf32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.relu6"(%arg0) : (tensor<?x21x3xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.707330704> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR5]], %arg0, %[[VAR3]])
func.func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.leaky_relu"(%arg0)  {alpha = 0.707330704 : f32}  : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_prelu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 2, 3>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR4:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
func.func @test_prelu(%arg0: tensor<4x2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x2x3xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<4x2x3xf32>, tensor<2x3xf32>) -> tensor<4x2x3xf32>
  func.return %0 : tensor<4x2x3xf32>
}

// -----

// CHECK-LABEL: test_prelu_qu8
// CHECK-DAG: %[[VAL_0:.*]]: tensor<1x7x7x9x!quant.uniform<u8:f32, 0.015686038881540298:128>>
// CHECK-DAG: %[[VAL_1:.*]]: tensor<7x7x9x!quant.uniform<i8<-127:127>:f32, 0.031982593238353729>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0> : tensor<1x1x1x1xi32>}
// CHECK: %[[VAL_3:.*]] = "tosa.rescale"(%[[VAL_0]]) {double_round = false, input_zp = 128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_4:.*]] = "tosa.rescale"(%[[VAL_3]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_5:.*]] = "tosa.rescale"(%[[VAL_4]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_6:.*]] = "tosa.greater_equal"(%[[VAL_5]], %[[VAL_2]]) : (tensor<1x7x7x9xi32>, tensor<1x1x1x1xi32>) -> tensor<1x7x7x9xi1>
// CHECK: %[[VAL_7:.*]] = "tosa.rescale"(%[[VAL_1]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_8:.*]] = "tosa.reshape"(%[[VAL_7]]) {new_shape = array<i64: 1, 7, 7, 9>}
// CHECK: %[[VAL_9:.*]] = "tosa.mul"(%[[VAL_5]], %[[VAL_8]]) {shift = 0 : i32}
// CHECK: %[[VAL_10:.*]] = "tosa.rescale"(%[[VAL_9]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1285552818>, output_zp = -24 : i32, per_channel = false, scale32 = true, shift = array<i32: 37>}
// CHECK: %[[VAL_11:.*]] = "tosa.rescale"(%[[VAL_4]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1256105947>, output_zp = -24 : i32, per_channel = false, scale32 = true, shift = array<i32: 32>}
// CHECK: %[[VAL_12:.*]] = "tosa.select"(%[[VAL_6]], %[[VAL_11]], %[[VAL_10]]) : (tensor<1x7x7x9xi1>, tensor<1x7x7x9x!quant.uniform<i8:f32, 0.053634826093912125:-24>>, tensor<1x7x7x9x!quant.uniform<i8:f32, 0.053634826093912125:-24>>)
// CHECK: %[[VAL_13:.*]] = "tosa.rescale"(%[[VAL_12]]) {double_round = true, input_zp = -24 : i32, multiplier = array<i32: 1073741824>, output_zp = -24 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_14:.*]] = "tosa.rescale"(%[[VAL_13]]) {double_round = false, input_zp = -24 : i32, multiplier = array<i32: 1073741824>, output_zp = 104 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
func.func @test_prelu_qu8(%arg0: tensor<1x7x7x9x!quant.uniform<u8:f32, 0.015686038881540298:128>>,  %arg1: tensor<7x7x9x!quant.uniform<i8<-127:127>:f32, 0.031982593238353729>>) -> tensor<1x7x7x9x!quant.uniform<u8:f32, 0.053634826093912125:104>> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x7x7x9x!quant.uniform<i8:f32, 0.015686038881540298>>} : (tensor<1x7x7x9x!quant.uniform<u8:f32, 0.015686038881540298:128>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.015686038881540298>>
  %2 = "tfl.prelu"(%0, %arg1) : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<7x7x9x!quant.uniform<i8<-127:127>:f32, 0.031982593238353729>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.053634826093912125:-24>>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x7x7x9x!quant.uniform<u8:f32, 0.053634826093912125:104>>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.053634826093912125:-24>>) -> tensor<1x7x7x9x!quant.uniform<u8:f32, 0.053634826093912125:104>>
  func.return %3 : tensor<1x7x7x9x!quant.uniform<u8:f32, 0.053634826093912125:104>>
}

// -----

// CHECK-LABEL: test_prelu_qi8
// CHECK-DAG: %[[VAL_0:.*]]: tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>
// CHECK-DAG: %[[VAL_1:.*]]: tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.025940980762243271>>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() {value = dense<0> : tensor<1x1x1x1xi32>}
// CHECK: %[[VAL_3:.*]] = "tosa.rescale"(%[[VAL_0]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_4:.*]] = "tosa.greater_equal"(%[[VAL_3]], %[[VAL_2]]) : (tensor<1x8x4x17xi32>, tensor<1x1x1x1xi32>) -> tensor<1x8x4x17xi1>
// CHECK: %[[VAL_5:.*]] = "tosa.rescale"(%[[VAL_1]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[VAL_6:.*]] = "tosa.reshape"(%[[VAL_5]]) {new_shape = array<i64: 1, 8, 4, 17>}
// CHECK: %[[VAL_7:.*]] = "tosa.mul"(%[[VAL_3]], %[[VAL_6]]) {shift = 0 : i32}
// CHECK: %[[VAL_8:.*]] = "tosa.rescale"(%[[VAL_7]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1125394528>, output_zp = -6 : i32, per_channel = false, scale32 = true, shift = array<i32: 37>}
// CHECK: %[[VAL_9:.*]] = "tosa.rescale"(%[[VAL_0]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1355715088>, output_zp = -6 : i32, per_channel = false, scale32 = true, shift = array<i32: 32>}
// CHECK: %[[VAL_10:.*]] = "tosa.select"(%[[VAL_4]], %[[VAL_9]], %[[VAL_8]]) : (tensor<1x8x4x17xi1>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>>, tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>>
func.func @test_prelu_qi8(%arg0: tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, %arg1: tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.025940980762243271>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<1x8x4x17x!quant.uniform<i8:f32, 0.015686038881540298>>, tensor<8x4x17x!quant.uniform<i8<-127:127>:f32, 0.025940980762243271>>) -> tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>>
  func.return %0 : tensor<1x8x4x17x!quant.uniform<i8:f32, 0.049694087356328964:-6>>
}

// -----

// CHECK-LABEL: test_logical_and
// CHECK: %[[VAR0:.*]] = "tosa.logical_and"(%arg0, %arg1)
func.func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<*xi1> {
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_logical_or
// CHECK: %[[VAR0:.*]] = "tosa.logical_or"(%arg0, %arg1)
func.func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<*xi1> {
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_logical_not
// CHECK: %[[VAR0:.*]] = "tosa.logical_not"(%arg0)
func.func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<*xi1> {
  %0 = "tfl.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// CHECK-LABEL: test_reduce_all_axis_1_keep_true
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = "tosa.reduce_all"(%[[VAL_0]]) {axis = 1 : i64} : (tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1>
func.func @test_reduce_all_axis_1_keep_true(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = true}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x1x8x19xi1>
  func.return %0 : tensor<1x1x8x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_1_keep_false
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = "tosa.reduce_all"(%[[VAL_0]]) {axis = 1 : i64} : (tensor<1x4x8x19xi1>) -> tensor<1x1x8x19xi1>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 8, 19>} : (tensor<1x1x8x19xi1>) -> tensor<1x8x19xi1>
func.func @test_reduce_all_axis_1_keep_false(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x8x19xi1> {
  %cst = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = false}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x8x19xi1>
  func.return %0 : tensor<1x8x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_2_keep_true
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = "tosa.reduce_all"(%[[VAL_0]]) {axis = 2 : i64} : (tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1>
func.func @test_reduce_all_axis_2_keep_true(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = true}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x4x1x19xi1>
  func.return %0 : tensor<1x4x1x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_all_axis_2_keep_false
// CHECK-SAME: %[[VAL_0:.+]]: tensor<1x4x8x19xi1>
// CHECK: %[[VAL_1:.*]] = "tosa.reduce_all"(%[[VAL_0]]) {axis = 2 : i64} : (tensor<1x4x8x19xi1>) -> tensor<1x4x1x19xi1>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 4, 19>} : (tensor<1x4x1x19xi1>) -> tensor<1x4x19xi1>
func.func @test_reduce_all_axis_2_keep_false(%arg0: tensor<1x4x8x19xi1>) -> tensor<1x4x19xi1> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %0 = "tfl.reduce_all"(%arg0, %cst)  {keep_dims = false}  : (tensor<1x4x8x19xi1>, tensor<1xi32>) -> tensor<1x4x19xi1>
  func.return %0 : tensor<1x4x19xi1>
}

// -----

// CHECK-LABEL: test_reduce_any
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_any"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xi1>, tensor<1xi32>) -> tensor<21x3xi1>
  func.return %0 : tensor<21x3xi1>
}

// -----

// CHECK-LABEL: test_reduce_min
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_min"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_min"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_max
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_max"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_sum
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// CHECK-LABEL: test_reduce_sum_nonzero_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<10x20x30x40x50xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.transpose"(%[[VAL_0]], %[[VAL_1]]) : (tensor<10x20x30x40x50xf32>, tensor<5xi32>) -> tensor<10x20x30x50x40xf32>
// CHECK: %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_2]]) {new_shape = array<i64: 300000, 40>} : (tensor<10x20x30x50x40xf32>) -> tensor<300000x40xf32>
// CHECK: %[[VAL_4:.*]] = "tosa.reduce_sum"(%[[VAL_3]]) {axis = 1 : i64} : (tensor<300000x40xf32>) -> tensor<300000x1xf32>
// CHECK: %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_4]]) {new_shape = array<i64: 10, 20, 30, 50>} : (tensor<300000x1xf32>) -> tensor<10x20x30x50xf32>
// CHECK: return %[[VAL_5]] : tensor<10x20x30x50xf32>
func.func @test_reduce_sum_nonzero_axis(%arg0: tensor<10x20x30x40x50xf32> {tf._user_specified_name = "inp_list"}) -> tensor<10x20x30x50xf32> {
  %cst = arith.constant dense<3> : tensor<i32>
  %0 = "tfl.sum"(%arg0, %cst) {device = "", keep_dims = false} : (tensor<10x20x30x40x50xf32>, tensor<i32>) -> tensor<10x20x30x50xf32>
  func.return %0 : tensor<10x20x30x50xf32>
}

// -----

// -----

// CHECK-LABEL: test_reduce_sum_5D
func.func @test_reduce_sum_5D(%arg0: tensor<4x5x6x7x8xf32>) -> tensor<6x8xf32> {
  %cst = arith.constant dense<[0, 1, 3]> : tensor<3xi32>
  // CHECK-DAG: %[[PERM:.+]] = "tosa.const"() {value = dense<[2, 4, 0, 1, 3]> : tensor<5xi32>}
  // CHECK-DAG: %[[TRANSPOSE:.+]] = "tosa.transpose"(%arg0, %[[PERM]])
  // CHECK-DAG: %[[RESHAPE0:.+]] = "tosa.reshape"(%[[TRANSPOSE:.+]]) {new_shape = array<i64: 48, 140>}
  // CHECK-DAG: %[[REDUCE:.+]] = "tosa.reduce_sum"(%[[RESHAPE0]]) {axis = 1 : i64}
  // CHECK: %[[RESHAPE1:.+]] = "tosa.reshape"(%[[REDUCE]]) {new_shape = array<i64: 6, 8>}
  %0 = "tfl.sum"(%arg0, %cst)  {keep_dims = false}  : (tensor<4x5x6x7x8xf32>, tensor<3xi32>) -> tensor<6x8xf32>
  func.return %0 : tensor<6x8xf32>
}

// -----

// CHECK-LABEL: test_reduce_mean
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0.0769230798> : tensor<1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = array<i64: 21, 3>}
// CHECK: %[[VAR4:.*]] = "tosa.mul"(%[[VAR2]], %[[VAR0]]) {shift = 0 : i32}
func.func @test_reduce_mean(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.mean"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_reduce_product
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_prod"(%arg0) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 21, 3>}
func.func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %cst = arith.constant dense<0> : tensor<1xi32>
  %0 = "tfl.reduce_prod"(%arg0, %cst)  {keep_dims = false}  : (tensor<13x21x3xf32>, tensor<1xi32>) -> tensor<21x3xf32>
  func.return %0 : tensor<21x3xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: %[[VAR0:.*]] = "tosa.minimum"(%arg0, %arg1)
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: %[[VAR0:.*]] = "tosa.maximum"(%arg0, %arg1)
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.maximum"(%arg0, %arg1)
// CHECK-SAME: -> tensor<13x21x?xf32>
func.func @test_max_dynamic(%arg0: tensor<13x1x?xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x1x?xf32>, tensor<13x21x1xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: test_pow
// CHECK: %[[VAR0:.*]] = "tosa.pow"(%arg0, %arg1)
func.func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_pow_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.pow"(%arg0, %arg1)
// CHECK-SAME: -> tensor<13x21x3xf32>
func.func @test_pow_dynamic(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_abs
// CHECK: %[[VAR0:.*]] = "tosa.abs"(%arg0)
func.func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_ceil
// CHECK: %[[VAR0:.*]] = "tosa.ceil"(%arg0)
func.func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_floor
// CHECK: %[[VAR0:.*]] = "tosa.floor"(%arg0)
func.func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_log
// CHECK: %[[VAR0:.*]] = "tosa.log"(%arg0)
func.func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_negate
// CHECK: %[[VAR0:.*]] = "tosa.negate"(%arg0)
func.func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.neg"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_rsqrt
// CHECK: %[[VAR0:.*]] = "tosa.rsqrt"(%arg0)
func.func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
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
  %0 = "tfl.sin"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>

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
  %0 = "tfl.cos"(%arg0) : (tensor<10xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_atan2
// CHECK-SAME: -> tensor<13x21x3xf32>
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<2.000000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() {value = dense<3.276700e+04> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() {value = dense<2.38418579E-7> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_6:.*]] = "tosa.const"() {value = dense<1.57079637> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_7:.*]] = "tosa.const"() {value = dense<3.14159274> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_8:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
// CHECK-DAG: %[[VAL_9:.*]] = "tosa.const"() {value = dense<{{.+}}> : tensor<513xi16>} : () -> tensor<513xi16>
// CHECK: %[[VAL_10:.*]] = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_11:.*]] = "tosa.abs"(%arg1) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_12:.*]] = "tosa.minimum"(%[[VAL_10]], %[[VAL_11]]) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_13:.*]] = "tosa.maximum"(%[[VAL_10]], %[[VAL_11]]) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_14:.*]] = "tosa.reciprocal"(%[[VAL_13]]) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_15:.*]] = "tosa.mul"(%[[VAL_14]], %[[VAL_12]]) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_16:.*]] = "tosa.mul"(%[[VAL_15]], %[[VAL_2]]) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_17:.*]] = "tosa.sub"(%[[VAL_16]], %[[VAL_3]]) : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_18:.*]] = "tosa.mul"(%[[VAL_17]], %[[VAL_4]]) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_19:.*]] = "tosa.cast"(%[[VAL_18]]) : (tensor<13x21x3xf32>) -> tensor<13x21x3xi16>
// CHECK: %[[VAL_20:.*]] = "tosa.table"(%[[VAL_19]], %[[VAL_9]]) : (tensor<13x21x3xi16>, tensor<513xi16>) -> tensor<13x21x3xi32>
// CHECK: %[[VAL_21:.*]] = "tosa.cast"(%[[VAL_20]]) : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_22:.*]] = "tosa.mul"(%[[VAL_21]], %[[VAL_5]]) {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_23:.*]] = "tosa.sub"(%[[VAL_6]], %[[VAL_22]]) : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_24:.*]] = "tosa.greater"(%[[VAL_10]], %[[VAL_11]]) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_25:.*]] = "tosa.select"(%[[VAL_24]], %[[VAL_23]], %[[VAL_22]]) : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_26:.*]] = "tosa.sub"(%[[VAL_7]], %[[VAL_25]]) : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_27:.*]] = "tosa.greater"(%[[VAL_8]], %arg1) : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_28:.*]] = "tosa.select"(%[[VAL_27]], %[[VAL_26]], %[[VAL_25]]) : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_29:.*]] = "tosa.negate"(%[[VAL_28]]) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: %[[VAL_30:.*]] = "tosa.greater"(%[[VAL_8]], %arg0) : (tensor<1x1x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
// CHECK: %[[VAL_31:.*]] = "tosa.select"(%[[VAL_30]], %[[VAL_29]], %[[VAL_28]]) : (tensor<13x21x3xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK: return %[[VAL_31]] : tensor<13x21x3xf32>
func.func @test_atan2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.atan2"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----


// CHECK-LABEL: test_sigmoid
// CHECK: %[[VAR0:.*]] = "tosa.sigmoid"(%arg0)
func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_square
// CHECK: %[[VAR0:.*]] = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32}
func.func @test_square(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.square"(%arg0) : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: %[[VAR0:.*]] = "tosa.equal"(%arg0, %arg1)
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: %[[VAR0:.*]] = "tosa.greater_equal"(%arg0, %arg1)
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: %[[VAR0:.*]] = "tosa.greater"(%arg0, %arg1)
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: %[[VAR0:.*]] = "tosa.greater"(%arg1, %arg0)
func.func @test_less(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.greater"(%arg1, %arg0)
// CHECK-SAME: -> tensor<13x?x3xi1>
func.func @test_less_dynamic(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x?x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x?x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK: %[[VAR0:.*]] = "tosa.greater_equal"(%arg1, %arg0)
func.func @test_less_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_less_equal_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.greater_equal"(%arg1, %arg0)
// CHECK-SAME: -> tensor<13x?x3xi1>
func.func @test_less_equal_dynamic(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x?x3xf32>) -> tensor<*xi1> {
  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x?x3xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: test_avg_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_avg_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_avg_pool2d_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_avg_pool2d_dynamic(%arg0: tensor<?x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_max_pool2d_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d_dynamic(%arg0: tensor<?x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<?x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 819>}
func.func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 819]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_unknown
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 9, -1>}
// CHECK-SAME: -> tensor<9x91xf32>
func.func @test_reshape_unknown(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[9, -1]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_dynamic
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 3, -1>}
// CHECK-SAME: -> tensor<3x?xf32>
func.func @test_reshape_dynamic(%arg0: tensor<13x21x?xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[3, -1]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x?xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
func.func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_transpose
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>}
// CHECK: %[[VAR1:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
func.func @test_transpose(%arg0: tensor<13x?x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<13x?x3xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_slice
// CHECK: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 4, 11, 1>, start = array<i64: 6, 8, 0>}
func.func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[6, 8, 0]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[4, 11, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_simple
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 21, 2>, start = array<i64: 4, 0, 1>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 9, 1, 7, 3, 2, 1>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.slice"(%[[VAR1]]) {size = array<i64: 9, 1, 7, 1, 2, 1>, start = array<i64: 0, 0, 0, 0, 0, 0>}
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 9, 7, 2>}
func.func @test_strided_slice_simple(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_simple_negative
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 18, 2>, start = array<i64: 4, 0, 1>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 9, 1, 6, 3, 2, 1>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.slice"(%[[VAR1]]) {size = array<i64: 9, 1, 6, 1, 2, 1>, start = array<i64: 0, 0, 0, 0, 0, 0>}
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 9, 6, 2>}
func.func @test_strided_slice_simple_negative(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, -3, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 1 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_strideless
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 1, 2>, start = array<i64: 4, 0, 1>}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 9, 2>}
func.func @test_strided_slice_strideless(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 1, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 2 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_shrink
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 1, 21, 1>, start = array<i64: 4, 0, 1>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 1, 1, 7, 3, 1, 1>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.slice"(%[[VAR1]]) {size = array<i64: 1, 1, 7, 1, 1, 1>, start = array<i64: 0, 0, 0, 0, 0, 0>}
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 7>}
func.func @test_strided_slice_shrink(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 5 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_shrink_ignore_stride
// CHECK-DAG: %[[VAR0:.*]] =  "tosa.slice"(%arg0) {size = array<i64: 1, 1, 2>, start = array<i64: 4, 0, 1>}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 2>}
func.func @test_strided_slice_shrink_ignore_stride(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 3, 1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 3 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_unstrided
// CHECK-SAME: -> tensor<9x21x2xf32>
// CHECK: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 21, 2>, start = array<i64: 4, 0, 1>}
// CHECK: %[[VAR1:.*]] = "tosa.reverse"(%[[VAR0]]) {axis = 2 : i64}
// CHECK: return %[[VAR1]]
func.func @test_strided_slice_unstrided(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0, 1]> : tensor<3xi32>
  %cst_0 = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, 1, -1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_unstrided_shorter
// CHECK: -> tensor<9x21x3xf32>
// CHECK: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 9, 21, 3>, start = array<i64: 4, 0, 0>}
// CHECK: %[[VAR1:.*]] = "tosa.reverse"(%[[VAR0]]) {axis = 1 : i64}
// CHECK: return %[[VAR1]]
func.func @test_strided_slice_unstrided_shorter(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[4, 0]> : tensor<2xi32>
  %cst_0 = arith.constant dense<[13, 21]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[1, -1]> : tensor<2xi32>
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 3 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<13x21x3xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_strided_slice_dynamic_masked
// CHECK-SAME: -> tensor<10x?x?xf32>
// CHECK: %[[VAR0:.*]] = "tosa.reverse"(%arg0) {axis = 1 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reverse"(%[[VAR0]]) {axis = 2 : i64}
// CHECK: return %[[VAR1]]
func.func @test_strided_slice_dynamic_masked(%arg0: tensor<10x?x?xf32>, %arg1: tensor<3xi32>) -> tensor<*xf32> {
  %cst_0 = arith.constant dense<[13, -1, 3]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[1, -1, -1]> : tensor<3xi32>
  %0 = "tfl.strided_slice"(%arg0, %arg1, %cst_0, %cst_1)  {begin_mask = 7 : i32, ellipsis_mask = 0 : i32, end_mask = 7 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
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
  // CHECK: %[[VAR0:.*]] = "tosa.reverse"(%arg0) {axis = 1 : i64}
  // CHECK: %[[VAR1:.*]] = "tosa.reverse"(%[[VAR0]]) {axis = 2 : i64}
  // CHECK: return %[[VAR1]]
  %0 = "tfl.strided_slice"(%arg0, %cst, %cst_0, %cst_1)  {begin_mask = 2 : i32, ellipsis_mask = 0 : i32, end_mask = 7 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// -----

// CHECK-LABEL: test_strided_slice_dynamic_end
// CHECK-SAME: 10x?x?xf32>
func.func @test_strided_slice_dynamic_end(%arg0: tensor<10x?x?xf32>) -> tensor<*xf32> {
  %begin = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %end = arith.constant dense<[7, -1, 6]> : tensor<3xi32>
  %stride = arith.constant dense<[1, 2, -1]> : tensor<3xi32>

  // CHECK: %[[SLICE1:.+]] = "tosa.slice"(%arg0) {size = array<i64: 7, -1, 1>, start = array<i64: 0, 1, 2>}
  // CHECK: %[[RESHAPE1:.+]] = "tosa.reshape"(%[[SLICE1]]) {new_shape = array<i64: 7, 1, -1, 2, 1, 1>}
  // CHECK: %[[SLICE2:.+]] = "tosa.slice"(%[[RESHAPE1]]) {size = array<i64: 7, 1, -1, 1, 1, 1>, start = array<i64: 0, 0, 0, 0, 0, 0>}
  // CHECK: %[[RESHAPE2:.+]] = "tosa.reshape"(%[[SLICE2]]) {new_shape = array<i64: 7, -1>}
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %stride)  {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 2 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 4 : i32}  : (tensor<10x?x?xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<*xf32>
  // CHECK: return %[[RESHAPE2]]
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_select
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg2) {new_shape = array<i64: 1, 1, 1>} : (tensor<1xi1>) -> tensor<1x1x1xi1>
// CHECK: %[[VAR2:.*]] = "tosa.select"(%[[VAR1]], %arg0, %arg1)
func.func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<1xi1>) -> tensor<13x21x3xf32> {
  %0 = "tfl.select_v2"(%arg2, %arg0, %arg1) : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_select_with_unranked
func.func @test_select_with_unranked(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> {
  // CHECK: "tosa.select"
  // CHECK-SAME: (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %57 = "tfl.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  return %57 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_addn
// CHECK-DAG: %[[VAR0:.*]] = "tosa.add"(%arg0, %arg1)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.add"(%arg2, %[[VAR0]])
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg3, %[[VAR1]])
func.func @test_addn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2, %arg3) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_concatv2
// CHECK: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
func.func @test_concatv2(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<52x21x3xf32> {
  %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<52x21x3xf32>
  func.return %0 : tensor<52x21x3xf32>
}

// -----

// CHECK-LABEL: test_stack
// CHECK-DAG: %[[VAR0:.*]] = "tosa.concat"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i64}
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%[[VAR0]]) {new_shape = array<i64: 4, 13, 21, 3>}
func.func @test_stack(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>, %arg3: tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32> {
  %0 = "tfl.pack"(%arg0, %arg1, %arg2, %arg3)  {axis = 0 : i32, values_count = 4 : i32}  : (tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<4x13x21x3xf32>
  func.return %0 : tensor<4x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_unstack
// CHECK: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 32, 32, 8>}
func.func @test_unstack(%arg0: tensor<1x32x32x8xf32>) -> tensor<*xf32> {
  %0 = "tfl.unpack"(%arg0)  {axis = 0 : i32, num = 1 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_pad
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[\[}}1, 1], {{\[}}2, 2]]> : tensor<2x2xi32>}
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK: %[[VAR1:.*]] = "tosa.pad"(%arg0, %[[VAR0]], %[[PVAL]])
func.func @test_pad(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>
  %0 = "tfl.pad"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}


// -----

// CHECK-LABEL: test_pad_v2
// CHECK-SAME: -> tensor<1x257x9x28xf32>
func.func @test_pad_v2(%arg0: tensor<1x256x8x25xf32>) -> (tensor<*xf32>) {
  // CHECK-DAG: %[[PADDING:.+]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [1, 0], [0, 1], [1, 2]]> : tensor<4x2xi32>}
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 0], [1, 0], [0, 1], [1, 2]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>

  // CHECK-DAG: %[[VAL:.+]] = "tosa.const"() {value = dense<-3.40282347E+38> : tensor<f32>}
  %1 = "tfl.pseudo_const"() {value = dense<-3.40282347E+38> : tensor<f32>} : () -> tensor<f32>

  // CHECK-DAG: %[[PAD:.+]] = "tosa.pad"(%arg0, %[[PADDING]], %[[VAL]]) : (tensor<1x256x8x25xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<1x257x9x28xf32>
  %2 = "tfl.padv2"(%arg0, %0, %1) : (tensor<1x256x8x25xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: return %[[PAD]]
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_expand_dims
// CHECK: %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 21, 3>}
func.func @test_expand_dims(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 13, 21, 3]> : tensor<4xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<4xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_shape
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<[13, 21, 3]> : tensor<3xi32>}
func.func @test_shape() -> tensor<3xi32> {
  %cst = arith.constant dense<[13, 21, 3]> : tensor<3xi32>
  func.return %cst : tensor<3xi32>
}

// -----

// CHECK-LABEL: test_rank
// CHECK: %[[VAR0:.*]] = "tosa.const"() {value = dense<3> : tensor<i32>}
func.func @test_rank() -> tensor<i32> {
  %cst = arith.constant dense<3> : tensor<i32>
  func.return %cst : tensor<i32>
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
  %0 = "tfl.elu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
func.func @test_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.softmax"(%arg0)  {beta = 1.000000e+00 : f32}  : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_l2normalization
func.func @test_l2normalization(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
  // CHECK-DAG: %[[MIN:.+]] = "tosa.const"() {value = dense<1.08420217E-19> : tensor<1x1xf32>}
  // CHECK-DAG: %[[SQR:.+]] = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32}
  // CHECK-DAG: %[[SUM:.+]] = "tosa.reduce_sum"(%[[SQR]]) {axis = 1 : i64}
  // CHECK-DAG: %[[MAX:.+]] = "tosa.maximum"(%[[SUM]], %[[MIN]])
  // CHECK-DAG: %[[RSQRT:.+]] = "tosa.rsqrt"(%[[MAX]])
  // CHECK-DAG: %[[MUL:.+]] = "tosa.mul"(%[[RSQRT]], %arg0)
  // CHECK: %[[CLAMP:.+]] = "tosa.clamp"(%[[MUL]]) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
  %0 = "tfl.l2_normalization"(%arg0) {fused_activation_function = "RELU"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: test_log_softmax
// CHECK-DAG: %[[VAR0:.*]] = "tosa.exp"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reciprocal"(%[[VAR1]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR2]]) {shift = 0 : i32}
// CHECK: %[[VAR4:.*]] = "tosa.log"(%[[VAR3]])
func.func @test_log_softmax(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.log_softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_matmul
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>}
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

// CHECK-LABEL: @test_fullyconnected
func.func @test_fullyconnected(%arg0: tensor<14x19xf32>, %arg1: tensor<28x19xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  // CHECK: "tosa.fully_connected"
  // CHECK-SAME: tensor<14x19xf32>
  // CHECK-SAME: tensor<28x19xf32>
  // CHECK-SAME: tensor<28xf32>
  // CHECK-SAME: tensor<14x28xf32>
  %2 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<14x19xf32>, tensor<28x19xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  func.return %2 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: @test_fullyconnected_in_batch_dim
func.func @test_fullyconnected_in_batch_dim(%arg0: tensor<1x14x19xf32>, %arg1: tensor<28x19xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  // CHECK: "tosa.reshape"
  // CHECK-SAME: tensor<1x14x19xf32>
  // CHECK-SAME: tensor<14x19xf32>
  // CHECK: "tosa.fully_connected"
  // CHECK-SAME: tensor<14x19xf32>
  // CHECK-SAME: tensor<28x19xf32>
  // CHECK-SAME: tensor<28xf32>
  // CHECK-SAME: tensor<14x28xf32>
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x14x19xf32>, tensor<28x19xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  func.return %0 : tensor<14x28xf32>
}

// -----

// CHECK-LABEL: @test_fullyconnected_extra_dim
func.func @test_fullyconnected_extra_dim(%arg0: tensor<1x14x19xf32>, %arg1: tensor<28x19xf32>, %arg2: tensor<28xf32>) -> tensor<1x14x28xf32> {
  // CHECK: "tosa.reshape"
  // CHECK-SAME: tensor<1x14x19xf32>
  // CHECK-SAME: tensor<14x19xf32>
  // CHECK: "tosa.fully_connected"
  // CHECK-SAME: tensor<14x19xf32>
  // CHECK-SAME: tensor<28x19xf32>
  // CHECK-SAME: tensor<28xf32>
  // CHECK-SAME: tensor<14x28xf32>
  // CHECK: "tosa.reshape"
  // CHECK-SAME: tensor<14x28xf32>
  // CHECK-SAME: tensor<1x14x28xf32>
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x14x19xf32>, tensor<28x19xf32>, tensor<28xf32>) -> tensor<1x14x28xf32>
  func.return %0 : tensor<1x14x28xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul
func.func @test_batch_matmul(%arg0: tensor<1x16x128xf32>, %arg1: tensor<1x128x32xf32>) -> (tensor<1x16x32xf32> ) {
  // CHECK: "tosa.matmul"(%arg0, %arg1)
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<1x16x128xf32>, tensor<1x128x32xf32>) -> tensor<1x16x32xf32>
  func.return %0 : tensor<1x16x32xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul_4d
func.func @test_batch_matmul_4d(%arg0: tensor<4x5x16x128xf32>, %arg1: tensor<4x5x128x32xf32>) -> (tensor<4x5x16x32xf32> ) {
  // CHECK: %[[R0:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 20, 16, 128>}
  // CHECK: %[[R1:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 20, 128, 32>}
  // CHECK: %[[MM:.*]] = "tosa.matmul"(%[[R0]], %[[R1]])
  // CHECK: "tosa.reshape"(%[[MM]]) {new_shape = array<i64: 4, 5, 16, 32>}
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<4x5x16x128xf32>, tensor<4x5x128x32xf32>) -> tensor<4x5x16x32xf32>
  func.return %0 : tensor<4x5x16x32xf32>
}

// -----

// CHECK-LABEL: @test_batch_matmul_transpose
func.func @test_batch_matmul_transpose(%arg0: tensor<1x16x128xf32>, %arg1: tensor<1x128x32xf32>) -> (tensor<1x32x16xf32> ) {
  // CHECK-DAG: %[[PERM:.+]] = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>}
  // CHECK-DAG: %[[TP0:.+]] = "tosa.transpose"(%arg0, %[[PERM]])
  // CHECK-DAG: %[[TP1:.+]] = "tosa.transpose"(%arg1, %[[PERM]])
  // CHECK: "tosa.matmul"(%[[TP1]], %[[TP0]])
  %0 = "tfl.batch_matmul"(%arg1, %arg0) {adj_x = true, adj_y = true} : (tensor<1x128x32xf32>, tensor<1x16x128xf32>) -> tensor<1x32x16xf32>
  func.return %0 : tensor<1x32x16xf32>
}

// -----

// CHECK-LABEL: test_add_scalar
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1xf32>}
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg0, %[[VAR0]])
func.func @test_add_scalar(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = "tfl.add"(%arg0, %cst)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xf32>, tensor<f32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_add_1d
// CHECK-DAG: %[[VAR0:.*]] = "tosa.reduce_sum"(%arg1) {axis = 0 : i64}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reduce_sum"(%[[VAR0]]) {axis = 1 : i64}
// CHECK: %[[VAR2:.*]] = "tosa.add"(%arg0, %[[VAR1]])
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
  // CHECK: "tosa.clamp"(%{{.+}}) {max_fp = 0.000000e+00 : f32, max_int = -67 : i64, min_fp = 0.000000e+00 : f32, min_int = -127 : i64}
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
  // CHECK: "tosa.clamp"(%{{.+}}) {max_fp = 0.000000e+00 : f32, max_int = 127 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU6"} : (tensor<10x!quant.uniform<i8:f32, 0.01:-129>>, tensor<10x!quant.uniform<i8:f32, 0.01:-129>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-129>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.01:-129>>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun1to1_noclamp
func.func @test_fused_activation_relun1to1_noclamp(
                         %arg0: tensor<10x!quant.uniform<i8:f32, 0.001:-120>>,
                         %arg1: tensor<10x!quant.uniform<i8:f32, 0.001:-120>>) -> tensor<10x!quant.uniform<i8:f32, 0.001:-120>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK:  "tosa.clamp"(%{{.}}) {max_fp = 0.000000e+00 : f32, max_int = 127 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU_N1_TO_1"}  : (tensor<10x!quant.uniform<i8:f32, 0.001:-120>>, tensor<10x!quant.uniform<i8:f32, 0.001:-120>>) -> tensor<10x!quant.uniform<i8:f32, 0.001:-120>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.001:-120>>
}

// -----

// CHECK-LABEL: func @test_fused_activation_relun1to1_clamp
func.func @test_fused_activation_relun1to1_clamp(
                         %arg0: tensor<10x!quant.uniform<i8:f32, 0.01:-10>>,
                         %arg1: tensor<10x!quant.uniform<i8:f32, 0.01:-10>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-10>> {
  %cst = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK:  "tosa.clamp"(%{{.}}) {max_fp = 0.000000e+00 : f32, max_int = 90 : i64, min_fp = 0.000000e+00 : f32, min_int = -110 : i64}
  %0 = "tfl.add"(%arg0, %arg0)  {fused_activation_function = "RELU_N1_TO_1"}  : (tensor<10x!quant.uniform<i8:f32, 0.01:-10>>, tensor<10x!quant.uniform<i8:f32, 0.01:-10>>) -> tensor<10x!quant.uniform<i8:f32, 0.01:-10>>
  func.return %0 : tensor<10x!quant.uniform<i8:f32, 0.01:-10>>
}

// -----

// CHECK-LABEL: test_split
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 0, 0>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 7, 0>}
// CHECK: %[[VAR2:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 14, 0>}
func.func @test_split(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_split_dynamic
func.func @test_split_dynamic(%arg0: tensor<13x?x3xf32>) -> (tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>) {
  %cst_0 = arith.constant dense<1> : tensor<i32>
  // CHECK-DAG: %[[VAR0:.+]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 13, 3, -1, 3>}
  // CHECK-DAG: %[[VAR1:.+]] = "tosa.slice"(%[[VAR0]]) {size = array<i64: 13, 1, -1, 3>, start = array<i64: 0, 0, 0, 0>}
  // CHECK-DAG: %[[VAR2:.+]] = "tosa.slice"(%[[VAR0]]) {size = array<i64: 13, 1, -1, 3>, start = array<i64: 0, 1, 0, 0>}
  // CHECK-DAG: %[[VAR3:.+]] = "tosa.slice"(%[[VAR0]]) {size = array<i64: 13, 1, -1, 3>, start = array<i64: 0, 2, 0, 0>}
  // CHECK-DAG: %[[VAR4:.+]] = "tosa.reshape"(%[[VAR1]]) {new_shape = array<i64: 13, -1, 3>}
  // CHECK-DAG: %[[VAR5:.+]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 13, -1, 3>}
  // CHECK-DAG: %[[VAR6:.+]] = "tosa.reshape"(%[[VAR3]]) {new_shape = array<i64: 13, -1, 3>}
  // CHECK: return %[[VAR4]], %[[VAR5]], %[[VAR6]]
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x?x3xf32>) -> (tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x?x3xf32>, tensor<13x?x3xf32>, tensor<13x?x3xf32>
}

// -----

// CHECK-LABEL: test_split_neg
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 0, 0>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 7, 0>}
// CHECK: %[[VAR2:.*]] = "tosa.slice"(%arg0) {size = array<i64: 13, 7, 3>, start = array<i64: 0, 14, 0>}
func.func @test_split_neg(%arg0: tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>) {
  %cst_0 = arith.constant dense<-2> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<13x21x3xf32>) -> (tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<13x7x3xf32>, tensor<13x7x3xf32>, tensor<13x7x3xf32>
}

// -----

// CHECK-LABEL: test_split_axis_0
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 7, 13, 3>, start = array<i64: 0, 0, 0>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = array<i64: 7, 13, 3>, start = array<i64: 7, 0, 0>}
// CHECK: %[[VAR2:.*]] = "tosa.slice"(%arg0) {size = array<i64: 7, 13, 3>, start = array<i64: 14, 0, 0>}
func.func @test_split_axis_0(%arg0: tensor<21x13x3xf32>) -> (tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>) {
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %0:3 = "tfl.split"(%cst_0, %arg0)  {num_splits = 3 : i32}  : (tensor<i32>, tensor<21x13x3xf32>) -> (tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<7x13x3xf32>, tensor<7x13x3xf32>, tensor<7x13x3xf32>
}

// -----

// CHECK-LABEL: test_split_v_neg_axis
// CHECK-DAG: %[[VAR0:.*]] = "tosa.slice"(%arg0) {size = array<i64: 2, 3, 3, 3>, start = array<i64: 0, 0, 0, 0>}
// CHECK: %[[VAR1:.*]] = "tosa.slice"(%arg0) {size = array<i64: 2, 3, 3, 5>, start = array<i64: 0, 0, 0, 3>}
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
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[}}[0, 0], [0, 1], [0, 0]]> : tensor<3x2xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>}
// CHECK-DAG: %[[PVAL:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.pad"(%arg0, %[[VAR0]], %[[PVAL]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 13, 11, 2, 3>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = array<i64: 26, 11, 3>}
func.func @test_space_to_batch(%arg0: tensor<13x21x3xf32>) -> tensor<26x11x3xf32> {
  %cst = arith.constant dense<2> : tensor<1xi32>
  %cst_0 = arith.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %0 = "tfl.space_to_batch_nd"(%arg0, %cst, %cst_0) : (tensor<13x21x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<26x11x3xf32>
  func.return %0 : tensor<26x11x3xf32>
}

// -----

// CHECK-LABEL: test_space_to_batch_dyn
// CHECK-DAG: %[[C0:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[C1:.+]] = "tosa.const"() {value = dense<{{\[\[}}0, 0], [0, 2], [0, 0], [0, 0]]> : tensor<4x2xi32>}
// CHECK-DAG: %[[C2:.+]] = "tosa.const"() {value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[PAD:.+]] = "tosa.pad"(%arg0, %[[C1]], %[[C0]]) : (tensor<?x241x1x80xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<?x243x1x80xf32>
// CHECK-DAG: %[[R0:.+]] = "tosa.reshape"(%[[PAD]]) {new_shape = array<i64: -1, 81, 3, 1, 1, 80>}
// CHECK-DAG: %[[T:.+]] = "tosa.transpose"(%[[R0]], %[[C2]])
// CHECK-DAG: %[[R1:.+]] = "tosa.reshape"(%[[T]]) {new_shape = array<i64: -1, 81, 1, 80>}
// CHECK: return %[[R1]] : tensor<?x81x1x80xf32>
func.func @test_space_to_batch_dyn(%arg0 : tensor<?x241x1x80xf32>) -> (tensor<?x81x1x80xf32>) {
    %0 = "tfl.pseudo_const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() {value = dense<[[0, 2], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tfl.space_to_batch_nd"(%arg0, %0, %1) : (tensor<?x241x1x80xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x81x1x80xf32>
    func.return %2 : tensor<?x81x1x80xf32>
}

// -----

// CHECK-LABEL: test_batch_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[3, 1, 2, 0]> : tensor<4xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%arg0, %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 2, 2, 2, 32, 32, 1>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.transpose"(%[[VAR3]], %[[VAR1]])
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = array<i64: 2, 64, 64, 1>}
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
// CHECK-DAG: %[[C0:.+]] = "tosa.const"() {value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[R0:.+]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 3, 1, -1, 79, 1, 80>}
// CHECK-DAG: %[[T:.+]] = "tosa.transpose"(%[[R0]], %[[C0]])
// CHECK-DAG: %[[R1:.+]] = "tosa.reshape"(%[[T]]) {new_shape = array<i64: -1, 237, 1, 80>}
// CHECK-DAG: %[[SLICE:.+]] = "tosa.slice"(%[[R1]]) {size = array<i64: -1, 235, 1, 80>, start = array<i64: 0, 0, 0, 0>}
// CHECK: return %[[SLICE]]
func.func @test_batch_to_space_dyn(%arg0 : tensor<?x79x1x80xf32>) -> (tensor<?x235x1x80xf32>) {
    %0 = "tfl.pseudo_const"() {value = dense<[3, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() {value = dense<[[0, 2], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tfl.batch_to_space_nd"(%arg0, %0, %1) : (tensor<?x79x1x80xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x235x1x80xf32>
    func.return %2 : tensor<?x235x1x80xf32>
}

// -----

// CHECK-LABEL: test_space_to_depth
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 16, 2, 16, 2, 8>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 1, 16, 16, 32>}
func.func @test_space_to_depth(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32> {
  %0 = "tfl.space_to_depth"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x16x16x32xf32>
  func.return %0 : tensor<1x16x16x32xf32>
}

// -----

// CHECK-LABEL: test_depth_to_space
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[0, 1, 3, 2, 4, 5]> : tensor<6xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 32, 32, 2, 2, 2>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%[[VAR1]], %[[VAR0]])
// CHECK: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 1, 64, 64, 2>}
func.func @test_depth_to_space(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32> {
  %0 = "tfl.depth_to_space"(%arg0)  {block_size = 2 : i32}  : (tensor<1x32x32x8xf32>) -> tensor<1x64x64x2xf32>
  func.return %0 : tensor<1x64x64x2xf32>
}

// -----

// CHECK-LABEL: @test_bucketize
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() {value = dense<{{\[\[\[}}0.000000e+00, 3.000000e+00, 8.000000e+00, 1.100000e+01]]]> : tensor<1x1x4xf32>}
// CHECK: %[[VAL_1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 5, 1>}
// CHECK: %[[VAL_2:.*]] = "tosa.greater_equal"(%[[VAL_1]], %[[VAL_0]])
// CHECK: %[[VAL_3:.*]] = "tosa.cast"(%[[VAL_2]]) : (tensor<2x5x4xi1>) -> tensor<2x5x4xi32>
// CHECK: %[[VAL_4:.*]] = "tosa.reduce_sum"(%[[VAL_3]]) {axis = 2 : i64}
// CHECK: %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_4]]) {new_shape = array<i64: 2, 5>}
func.func @test_bucketize(%arg0: tensor<2x5xf32>) -> tensor<2x5xi32> {
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x5xf32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @test_bucketize_cast_boundaries
// CHECK-DAG: %[[VAL_0:.*]] = "tosa.const"() {value = dense<{{\[}}0.000000e+00, 3.000000e+00, 8.000000e+00, 1.100000e+01]> : tensor<4xf32>}
// CHECK: %[[VAL_1:.*]] = "tosa.cast"(%[[VAL_0]]) : (tensor<4xf32>) -> tensor<4xi32>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 5, 1>}
// CHECK: %[[VAL_3:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 1, 4>}
// CHECK: %[[VAL_4:.*]] = "tosa.greater_equal"(%[[VAL_2]], %[[VAL_3]])
// CHECK: %[[VAL_5:.*]] = "tosa.cast"(%[[VAL_4]]) : (tensor<2x5x4xi1>) -> tensor<2x5x4xi32>
// CHECK: %[[VAL_6:.*]] = "tosa.reduce_sum"(%[[VAL_5]]) {axis = 2 : i64}
// CHECK: %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) {new_shape = array<i64: 2, 5>}
func.func @test_bucketize_cast_boundaries(%arg0: tensor<2x5xi32>) -> tensor<2x5xi32> {
  %0 = "tfl.bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x5xi32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @test_one_hot
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x4xi32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<f32>
// CHECK-DAG:     %[[RESHAPE:.*]] = "tosa.reshape"(%[[ARG1]]) {new_shape = array<i64: 1, 1, 1>}
// CHECK-DAG:     %[[TILE:.*]] = "tosa.tile"(%[[RESHAPE]]) {multiples = array<i64: 16, 1, 1>}
// CHECK-DAG:     %[[RESHAPE_0:.*]] = "tosa.reshape"(%[[ARG2]]) {new_shape = array<i64: 1, 1, 1>}
// CHECK-DAG:     %[[TILE_0:.*]] = "tosa.tile"(%[[RESHAPE_0]]) {multiples = array<i64: 16, 2, 1>}
// CHECK-DAG:     %[[RESHAPE_1:.*]] = "tosa.reshape"(%[[ARG0]]) {new_shape = array<i64: 16, 1>}
// CHECK-DAG:     %[[SCATTER:.*]] = "tosa.scatter"(%[[TILE_0]], %[[RESHAPE_1]], %[[TILE]])
// CHECK-DAG:     %[[RESHAPE_2:.*]] = "tosa.reshape"(%[[SCATTER]]) {new_shape = array<i64: 4, 4, 2>}
// CHECK:         return %[[RESHAPE_2]]
func.func @test_one_hot(%arg0: tensor<4x4xi32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<4x4x2xf32> {
  %0 = arith.constant dense<2> : tensor<i32>
  %1 = "tfl.one_hot"(%arg0, %0, %arg1, %arg2) {axis = -1 : i32} : (tensor<4x4xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<4x4x2xf32>
  func.return %1 : tensor<4x4x2xf32>
}

// -----

// CHECK-LABEL: test_fakequant_with_min_max_args
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<16383.75> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<6.10360876E-5> : tensor<1x1x1xf32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.mul"(%arg0, %[[VAR0]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.add"(%[[VAR4]], %[[VAR1]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.cast"(%[[VAR5]])
// CHECK-DAG: %[[VAR8:.*]] = "tosa.cast"(%[[VAR7]])
// CHECK-DAG: %[[VAR10:.*]] = "tosa.sub"(%[[VAR8]], %[[VAR1]])
// CHECK-DAG: %[[VAR12:.*]] = "tosa.mul"(%[[VAR10]], %[[VAR2]]) {shift = 0 : i32}
func.func @test_fakequant_with_min_max_args(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tfl.quantize"(%arg0)  {qtype = tensor<13x21x3x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>}  : (tensor<13x21x3xf32>) -> tensor<*x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>
  %1 = "tfl.dequantize"(%0) : (tensor<*x!quant.uniform<u16:f32, 6.1036087586785687E-5:32768>>) -> tensor<13x21x3xf32>
  func.return %1 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: @test_dequantize_float
// CHECK-SAME: -> tensor<10xf32>
func.func @test_dequantize_float(%arg0: tensor<10xf16>) -> tensor<*xf32> {
  // CHECK: %[[VAR0:.+]] = "tosa.cast"(%arg0) : (tensor<10xf16>) -> tensor<10xf32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<10xf16>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_dequantize_quant_uniform
func.func @test_dequantize_quant_uniform(%arg0: tensor<4x!quant.uniform<i8:f32, 1.0:-1>>) -> tensor<*xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tosa.const"() {value = dense<-1.000000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[VAL1:.+]] = "tosa.cast"(%arg0)
  // CHECK-DAG: %[[VAL2:.+]] = "tosa.sub"(%[[VAL1]], %[[VAL0]])
  %0 = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<i8:f32, 1.0:-1>>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// -----

// CHECK-LABEL: @test_dequantize_quant_per_axis
func.func @test_dequantize_quant_per_axis(%arg0: tensor<1x4x!quant.uniform<i8:f32:1, {1.0:5, 2.0:6, 3.0:7, 4.0:8}>>) -> tensor<*xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tosa.const"() {value = dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>}
  // CHECK-DAG: %[[VAL1:.+]] = "tosa.const"() {value = dense<{{\[}}[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]]> : tensor<1x4xf32>}
  // CHECK-DAG: %[[VAL2:.+]] = "tosa.cast"(%arg0) : (tensor<1x4x!quant.uniform<i8:f32:1, {1.000000e+00:5,2.000000e+00:6,3.000000e+00:7,4.000000e+00:8}>>) -> tensor<1x4xf32>
  // CHECK-DAG: %[[VAL3:.+]] = "tosa.sub"(%[[VAL2]], %[[VAL1]]) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  // CHECK: %[[VAL4:.+]] = "tosa.mul"(%[[VAL3]], %[[VAL0]]) {shift = 0 : i32} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
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
// CHECK-SAME: %arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>
// CHECK-SAME: %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.add"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func.func @test_add_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.028171317651867867:-1>> {
  %0 = "tfl.add"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.01564602367579937:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.028171317651867867:-1>>
}

// -----

// CHECK-LABEL: test_sub_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.sub"(%[[VAR0]], %[[VAR1]])
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func.func @test_sub_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<*x!quant.uniform<i8:f32, 0.028217222541570663:-1>> {
  %0 = "tfl.sub"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015683440491557121:-1>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015669029206037521>>) -> tensor<*x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.028217222541570663:-1>>
}

// -----

// CHECK-LABEL: test_mul_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%[[VAR0]], %[[VAR1]]) {shift = 0 : i32}
// CHECK: %[[VAR3:.*]] = "tosa.rescale"(%[[VAR2]])
func.func @test_mul_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, %arg1: tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, tensor<13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
}

// -----

// CHECK-LABEL: test_avg_pool2d_qi8
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>, stride = array<i64: 1, 1>}
// CHECK-SAME: -> tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
func.func @test_avg_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.015684349462389946:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015684349462389946:-1>>
}

// -----

// CHECK-LABEL: test_avg_pool2d_i16
// CHECK: %[[VAR0:.*]] = "tosa.avg_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
// CHECK-SAME: -> tensor<1x32x32x8xi16>
func.func @test_avg_pool2d_i16(%arg0: tensor<1x32x32x8xi16>) -> tensor<*xi16> {
  %0 = "tfl.average_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8xi16>) -> tensor<*xi16>
  func.return %0 : tensor<*xi16>
}

// -----

// CHECK-LABEL: test_max_pool2d_qi8
// CHECK: %[[VAR0:.*]] = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
func.func @test_max_pool2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>> {
  %0 = "tfl.max_pool_2d"(%arg0)  {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<1x32x32x8x!quant.uniform<i8:f32, 0.01568342000246048:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.01568342000246048:-1>>
}

// -----

// CHECK-LABEL: test_softmax_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<35> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<4> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() {value = dense<536870912> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() {value = dense<1515870810> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.const"() {value = dense<-1010580540> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.const"() {value = dense<1> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.const"() {value = dense<12> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.const"() {value = dense<7> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR9:.*]] = "tosa.const"() {value = dense<9> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR10:.*]] = "tosa.const"() {value = dense<17> : tensor<1x1x1xi32>}
// CHECK-DAG: %[[VAR11:.*]] = "tosa.const"() {value = dense<"0x5{{.*}}"> : tensor<513xi16>}
// CHECK-DAG: %[[VAR12:.*]] = "tosa.const"() {value = dense<"0xE{{.*}}"> : tensor<513xi16>}
// CHECK-DAG: %[[VAR13:.*]] = "tosa.const"() {value = dense<"0x4{{.*}}"> : tensor<513xi16>}
// CHECK-DAG: %[[VAR14:.*]] = "tosa.const"() {value = dense<"0x0{{.*}}"> : tensor<513xi16>}
// CHECK-DAG: %[[VAR15:.*]] = "tosa.rescale"(%arg0) {double_round = false, input_zp = -1 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK-DAG: %[[VAR16:.*]] = "tosa.reduce_max"(%[[VAR15]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR17:.*]] = "tosa.sub"(%[[VAR15]], %[[VAR16]])
// CHECK-DAG: %[[VAR18:.*]] = "tosa.rescale"(%[[VAR17]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 23>}
// CHECK-DAG: %[[VAR19:.*]] = "tosa.table"(%[[VAR18]], %[[VAR14]])
// CHECK-DAG: %[[VAR20:.*]] = "tosa.table"(%[[VAR18]], %[[VAR13]])
// CHECK-DAG: %[[VAR21:.*]] = "tosa.table"(%[[VAR18]], %[[VAR12]])
// CHECK-DAG: %[[VAR22:.*]] = "tosa.table"(%[[VAR18]], %[[VAR11]])
// CHECK-DAG: %[[VAR23:.*]] = "tosa.logical_left_shift"(%[[VAR19]], %[[VAR10]])
// CHECK-DAG: %[[VAR24:.*]] = "tosa.logical_left_shift"(%[[VAR20]], %[[VAR9]])
// CHECK-DAG: %[[VAR25:.*]] = "tosa.logical_left_shift"(%[[VAR21]], %[[VAR6]])
// CHECK-DAG: %[[VAR26:.*]] = "tosa.arithmetic_right_shift"(%[[VAR22]], %[[VAR8]]) {round = true}
// CHECK-DAG: %[[VAR27:.*]] = "tosa.add"(%[[VAR23]], %[[VAR24]])
// CHECK-DAG: %[[VAR28:.*]] = "tosa.add"(%[[VAR27]], %[[VAR25]])
// CHECK-DAG: %[[VAR29:.*]] = "tosa.add"(%[[VAR28]], %[[VAR26]])
// CHECK-DAG: %[[VAR30:.*]] = "tosa.arithmetic_right_shift"(%[[VAR29]], %[[VAR7]]) {round = true}
// CHECK-DAG: %[[VAR31:.*]] = "tosa.reduce_sum"(%[[VAR30]]) {axis = 2 : i64}
// CHECK-DAG: %[[VAR32:.*]] = "tosa.clz"(%[[VAR31]])
// CHECK-DAG: %[[VAR33:.*]] = "tosa.sub"(%[[VAR32]], %[[VAR6]])
// CHECK-DAG: %[[VAR34:.*]] = "tosa.logical_left_shift"(%[[VAR31]], %[[VAR33]])
// CHECK-DAG: %[[VAR35:.*]] = "tosa.mul"(%[[VAR34]], %[[VAR5]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR36:.*]] = "tosa.add"(%[[VAR35]], %[[VAR4]])
// CHECK-DAG: %[[VAR37:.*]] = "tosa.mul"(%[[VAR36]], %[[VAR34]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR38:.*]] = "tosa.sub"(%[[VAR3]], %[[VAR37]])
// CHECK-DAG: %[[VAR39:.*]] = "tosa.mul"(%[[VAR36]], %[[VAR38]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR40:.*]] = "tosa.mul"(%[[VAR39]], %[[VAR2]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR41:.*]] = "tosa.add"(%[[VAR36]], %[[VAR40]])
// CHECK-DAG: %[[VAR42:.*]] = "tosa.mul"(%[[VAR41]], %[[VAR34]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR43:.*]] = "tosa.sub"(%[[VAR3]], %[[VAR42]])
// CHECK-DAG: %[[VAR44:.*]] = "tosa.mul"(%[[VAR41]], %[[VAR43]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR45:.*]] = "tosa.mul"(%[[VAR44]], %[[VAR2]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR46:.*]] = "tosa.add"(%[[VAR41]], %[[VAR45]])
// CHECK-DAG: %[[VAR47:.*]] = "tosa.mul"(%[[VAR46]], %[[VAR34]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR48:.*]] = "tosa.sub"(%[[VAR3]], %[[VAR47]])
// CHECK-DAG: %[[VAR49:.*]] = "tosa.mul"(%[[VAR46]], %[[VAR48]]) {shift = 31 : i32}
// CHECK-DAG: %[[VAR50:.*]] = "tosa.mul"(%[[VAR49]], %[[VAR2]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR51:.*]] = "tosa.add"(%[[VAR46]], %[[VAR50]])
// CHECK-DAG: %[[VAR52:.*]] = "tosa.mul"(%[[VAR29]], %[[VAR51]]) {shift = 30 : i32}
// CHECK-DAG: %[[VAR53:.*]] = "tosa.sub"(%[[VAR1]], %[[VAR32]])
// CHECK-DAG: %[[VAR54:.*]] = "tosa.arithmetic_right_shift"(%[[VAR52]], %[[VAR53]]) {round = true}
// CHECK: %[[VAR55:.*]] = "tosa.rescale"(%[[VAR54]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
func.func @test_softmax_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015685837715864182:-1>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  func.return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----


// CHECK-LABEL: test_softmax_qi16
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<31> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<7> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.const"() {value = dense<32768> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.const"() {value = dense<14> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.const"() {value = dense<1073741824> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.const"() {value = dense<1> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.const"() {value = dense<32767> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.const"() {value = dense<"0xF{{.*}}>
// CHECK-DAG: %[[VAR8:.*]] = "tosa.const"() {value = dense<"0x0{{.*}}> : tensor<513xi16>}
// CHECK-DAG: %[[VAR9:.*]] = "tosa.rescale"(%arg0) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK-DAG: %[[VAR10:.*]] = "tosa.reduce_max"(%[[VAR9]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR11:.*]] = "tosa.sub"(%[[VAR9]], %[[VAR10]])
// CHECK-DAG: %[[VAR12:.*]] = "tosa.rescale"(%[[VAR11]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1717965619>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 32>}
// CHECK-DAG: %[[VAR13:.*]] = "tosa.add"(%[[VAR12]], %[[VAR6]])
// CHECK-DAG: %[[VAR14:.*]] = "tosa.cast"(%[[VAR13]])
// CHECK-DAG: %[[VAR15:.*]] = "tosa.table"(%[[VAR14]], %[[VAR8]])
// CHECK-DAG: %[[VAR16:.*]] = "tosa.arithmetic_right_shift"(%[[VAR15]], %[[VAR1]]) {round = true}
// CHECK-DAG: %[[VAR17:.*]] = "tosa.reduce_sum"(%[[VAR16]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR18:.*]] = "tosa.clz"(%[[VAR17]])
// CHECK-DAG: %[[VAR19:.*]] = "tosa.sub"(%[[VAR18]], %[[VAR5]])
// CHECK-DAG: %[[VAR20:.*]] = "tosa.logical_left_shift"(%[[VAR17]], %[[VAR19]])
// CHECK-DAG: %[[VAR21:.*]] = "tosa.sub"(%[[VAR20]], %[[VAR4]])
// CHECK-DAG: %[[VAR22:.*]] = "tosa.arithmetic_right_shift"(%[[VAR21]], %[[VAR3]]) {round = true}
// CHECK-DAG: %[[VAR23:.*]] = "tosa.sub"(%[[VAR22]], %[[VAR2]])
// CHECK-DAG: %[[VAR24:.*]] = "tosa.cast"(%[[VAR23]])
// CHECK-DAG: %[[VAR25:.*]] = "tosa.table"(%[[VAR24]], %[[VAR7]])
// CHECK-DAG: %[[VAR26:.*]] = "tosa.arithmetic_right_shift"(%[[VAR25]], %[[VAR1]]) {round = true}
// CHECK-DAG: %[[VAR27:.*]] = "tosa.mul"(%[[VAR26]], %[[VAR16]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR28:.*]] = "tosa.sub"(%[[VAR0]], %[[VAR18]])
// CHECK-DAG: %[[VAR29:.*]] = "tosa.arithmetic_right_shift"(%[[VAR27]], %[[VAR28]]) {round = true}
// CHECK: %[[VAR30:.*]] = "tosa.rescale"(%[[VAR29]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
func.func @test_softmax_qi16(%arg0: tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>) -> tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>> {
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<14x19x!quant.uniform<i16:f32, 6.103533087298274E-5>>) -> tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>>
  func.return %0 : tensor<14x19x!quant.uniform<i16:f32, 3.0517578125E-5>>
}

// -----

// CHECK-LABEL: test_sigmoid_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<256xi8>}
// CHECK: %[[VAR1:.*]] = "tosa.table"(%arg0, %[[VAR0]])
func.func @test_sigmoid_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tfl.logistic"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015667613595724106>>) -> tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}

// -----

// CHECK-LABEL: test_tanh_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<256xi8>}
// CHECK: %[[VAR1:.*]] = "tosa.table"(%arg0, %[[VAR0]])
func.func @test_tanh_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<*x!quant.uniform<i8:f32, 7.812500e-03>> {
  %0 = "tfl.tanh"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015673128888010979:-1>>) -> tensor<*x!quant.uniform<i8:f32, 7.812500e-03>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 7.812500e-03>>
}

// -----

// CHECK-LABEL: test_relu_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK: %[[VAR1:.*]] = "tosa.clamp"(%0) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = -1 : i64}
func.func @test_relu_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015671534463763237:-1>> {
  %0 = "tfl.relu"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015671534463763237:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015671534463763237:-1>>
}

// -----

// CHECK-LABEL: test_relu0To1_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK: %[[VAR1:.*]] = "tosa.clamp"(%0) {max_fp = 1.000000e+00 : f32, max_int = 64 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu0To1_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>> {
  %0 = "tfl.relu_0_to_1"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>>
}

// -----

// CHECK-LABEL: test_relu6_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.rescale"(%arg0)
// CHECK: %[[VAR1:.*]] = "tosa.clamp"(%0) {max_fp = 6.000000e+00 : f32, max_int = 384 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}
func.func @test_relu6_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015639215707778931>>) -> tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015639215707778931>>
}

// -----

// CHECK-LABEL: test_relu6_qu8
// CHECK: %[[CAST:.+]] = "tosa.rescale"(%arg0) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[RESCALE:.+]] = "tosa.rescale"(%[[CAST]]) {double_round = false, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
// CHECK: %[[CLAMP:.+]] = "tosa.clamp"(%[[RESCALE]]) {max_fp = 6.000000e+00 : f32, max_int = 22 : i64, min_fp = 0.000000e+00 : f32, min_int = -128 : i64}
// CHECK: %[[OUT:.+]] = "tosa.rescale"(%[[CLAMP]]) {double_round = false, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 30>}
func.func @test_relu6_qu8(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.04>>) -> tensor<*x!quant.uniform<u8:f32, 0.04>> {
  %0 = "tfl.relu6"(%arg0) : (tensor<13x21x3x!quant.uniform<u8:f32, 0.04>>) -> tensor<*x!quant.uniform<u8:f32, 0.04>>
  func.return %0 : tensor<*x!quant.uniform<u8:f32, 0.04>>
}

// -----

// CHECK-LABEL: test_leaky_relu_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0> : tensor<1x1xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%[[VAR1]], %[[VAR0]])
// CHECK-DAG: %[[VAR4:.*]] = "tosa.rescale"(%arg0)
// CHECK-DAG: %[[VAR5:.*]] = "tosa.rescale"(%arg0)
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %[[VAR5]], %[[VAR4]])
func.func @test_leaky_relu_qi8(%arg0: tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>> {
  %0 = "tfl.leaky_relu"(%arg0) {alpha = 0.948724806 : f32} : (tensor<14x19x!quant.uniform<i8:f32, 0.015519863925874233:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.015519863925874233:-1>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 14, 14>, mode = "BILINEAR", offset = array<i64: 0, 0>, scale = array<i64: 16, 2, 16, 2>}
// CHECK: %[[VAR2:.*]] = "tosa.rescale"(%[[VAR1]]) {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i32: 38>}
func.func @test_resize_bilinear_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_half_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 7, 7>, mode = "BILINEAR", offset = array<i64: -7, -7>, scale = array<i64: 16, 2, 16, 2>}
func.func @test_resize_bilinear_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_align_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 0, 0>, mode = "BILINEAR", offset = array<i64: 0, 0>, scale = array<i64: 1278, 158, 1278, 158>}
func.func @test_resize_bilinear_align_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_bilinear_align_half_qi8
// CHECK-DAG: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: -560, -560>, mode = "BILINEAR", offset = array<i64: -560, -560>, scale = array<i64: 1278, 158, 1278, 158>}
func.func @test_resize_bilinear_align_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_bilinear"(%arg0, %0) {align_corners = true, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_qi8
// CHECK: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 14, 14>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 0, 0>, scale = array<i64: 16, 2, 16, 2>}
func.func @test_resize_nearest_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}


// -----

// CHECK-LABEL: test_resize_nearest_half_qi8
// CHECK: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 15, 15>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 1, 1>, scale = array<i64: 16, 2, 16, 2>}
func.func @test_resize_nearest_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_align_qi8
// CHECK: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 639, 639>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 639, 639>, scale = array<i64: 1278, 158, 1278, 158>}
func.func @test_resize_nearest_align_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = true, half_pixel_centers = false} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_resize_nearest_align_half_qi8
// CHECK: %[[VAR1:.*]] = "tosa.resize"(%arg0) {border = array<i64: 718, 718>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 718, 718>, scale = array<i64: 1278, 158, 1278, 158>}
func.func @test_resize_nearest_align_half_qi8(%arg0: tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>> {
  %0 = "tfl.pseudo_const"() {value = dense<640> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.resize_nearest_neighbor"(%arg0, %0) {align_corners = true, half_pixel_centers = true} : (tensor<1x80x80x2x!quant.uniform<i8:f32, 0.42546585202217102>>, tensor<2xi32>) -> tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
  func.return %1 : tensor<1x640x640x2x!quant.uniform<i8:f32, 0.42546585202217102>>
}

// -----

// CHECK-LABEL: test_fullyconnected_qi8
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<0> : tensor<28xi32>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.transpose"(%arg1, %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.fully_connected"(%arg0, %[[VAR2]], %[[VAR1]]) {quantization_info = #tosa.conv_quant<input_zp = -1, weight_zp = -1>}
// CHECK: %[[VAR4:.*]] = "tosa.rescale"(%[[VAR3]]) {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1353377973>, output_zp = 3 : i32, per_channel = false, scale32 = true, shift = array<i32: 40>}
func.func @test_fullyconnected_qi8(%arg0: tensor<14x19x!quant.uniform<i8:f32, 0.015685491263866425:-1>>, %arg1: tensor<19x28x!quant.uniform<i8:f32, 0.015685983002185822:-1>>) -> tensor<14x28x!quant.uniform<i8:f32, 0.19988977909088135:3>> {
  %0 = "tfl.pseudo_const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tfl.transpose"(%arg1, %0) : (tensor<19x28x!quant.uniform<i8:f32, 0.015685983002185822:-1>>, tensor<2xi32>) -> tensor<28x19x!quant.uniform<i8:f32, 0.015685983002185822:-1>>
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %2 = "tfl.fully_connected"(%arg0, %1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<14x19x!quant.uniform<i8:f32, 0.015685491263866425:-1>>, tensor<28x19x!quant.uniform<i8:f32, 0.015685983002185822:-1>>, none) -> tensor<14x28x!quant.uniform<i8:f32, 0.19988977909088135:3>>
  func.return %2 : tensor<14x28x!quant.uniform<i8:f32, 0.19988977909088135:3>>
}

// -----
// CHECK-LABEL: test_gather
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 63>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 49>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.gather"(%[[VAR4]], %[[VAR5]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: 7, 7, 21, 3>}
// CHECK: return %[[VAR7]]
func.func @test_gather(%arg0: tensor<13x21x3xf32>, %arg1: tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_dyn
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, -1, 63>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 49>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.gather"(%[[VAR4]], %[[VAR5]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: 7, 7, 21, 3>}
// CHECK: return %[[VAR7]]
func.func @test_gather_dyn(%arg0: tensor<?x21x3xf32>, %arg1 : tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<?x21x3xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}


// -----
// CHECK-LABEL: test_gather_channel_dyn
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, -1>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, 49>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.gather"(%[[VAR4]], %[[VAR5]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: 7, 7, 21, -1>}
// CHECK: return %[[VAR7]]
func.func @test_gather_channel_dyn(%arg0: tensor<13x21x?xf32>, %arg1: tensor<7x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x?xf32>, tensor<7x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_indices_dyn
// CHECK-DAG: %[[VAR4:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 63>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, -1>}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.gather"(%[[VAR4]], %[[VAR5]])
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: -1, 7, 21, 3>}
// CHECK: return %[[VAR7]]
func.func @test_gather_indices_dyn(%arg0: tensor<13x21x3xf32>, %arg1: tensor<?x7xi32>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<?x7xi32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----
// CHECK-LABEL: test_gather_batch
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 4, 16>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.gather"(%[[VAR1]], %[[VAR0]])
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: 1, 3, 4, 4>}
// CHECK: return %[[VAR3]]
func.func @test_gather_batch(%arg0: tensor<1x4x4x4xi32>) -> tensor<1x3x4x4xi32> {
  %0 = "tfl.pseudo_const"() {value = dense<[[0, 3, 1]]> : tensor<1x3xi32>} : () -> tensor<1x3xi32>
  %1 = "tfl.gather"(%arg0, %0) {axis = 1 : i32, batch_dims = 1 : i32} : (tensor<1x4x4x4xi32>, tensor<1x3xi32>) -> tensor<1x3x4x4xi32>
  func.return %1 : tensor<1x3x4x4xi32>
}

// -----
// CHECK-LABEL: test_gather_batch_dyn
// CHECK-DAG: %[[VAR1:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: -1, 4, 16>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.gather"(%[[VAR1]], %arg1)
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR2]]) {new_shape = array<i64: -1, 3, 4, 4>}
// CHECK: return %[[VAR3]]
func.func @test_gather_batch_dyn(%arg0: tensor<?x4x4x4xi32>, %arg1: tensor<?x3xi32>) -> tensor<?x3x4x4xi32> {
  %1 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32, batch_dims = 1 : i32} : (tensor<?x4x4x4xi32>, tensor<?x3xi32>) -> tensor<?x3x4x4xi32>
  func.return %1 : tensor<?x3x4x4xi32>
}

// -----
// CHECK-LABEL: test_gather_nd
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 273, 3>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 42, 2>}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.mul"(%[[VAR3]], %[[VAR1]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.reduce_sum"(%[[VAR5]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR6]]) {new_shape = array<i64: 1, 42>}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.gather"(%[[VAR2]], %[[VAR7]])
// CHECK: %[[VAR9:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = array<i64: 6, 7, 3>}
func.func @test_gather_nd(%arg0: tensor<13x21x3xf32>, %arg1: tensor<6x7x2xi32>) -> tensor<6x7x3xf32> {
  %1 = "tfl.gather_nd"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<6x7x2xi32>) -> tensor<6x7x3xf32>
  func.return %1 : tensor<6x7x3xf32>
}

// -----
// CHECK-LABEL: test_gather_cast
// CHECK-DAG: %[[VAR1:.*]] = "tosa.cast"(%arg1)
// CHECK-DAG: %[[VAR2:.*]] = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 13, 63>}
// CHECK-DAG: %[[VAR3:.*]] = "tosa.reshape"(%[[VAR1]]) {new_shape = array<i64: 1, 49>}
// CHECK-DAG: %[[VAR4:.*]] = "tosa.gather"(%[[VAR2]], %[[VAR3]])
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reshape"(%[[VAR4]]) {new_shape = array<i64: 7, 7, 21, 3>}
// CHECK: return %[[VAR5]]
func.func @test_gather_cast(%arg0: tensor<13x21x3xf32>, %arg1: tensor<7x7xi64>) -> tensor<*xf32> {
  %2 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<7x7xi64>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<{{\[\[}}48, 1]]> : tensor<1x2xi32>}
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() {value = dense<-1> : tensor<1x48x1xi64>}
// CHECK-DAG: %[[VAR2:.*]] = "tosa.cast"(%arg0)
// CHECK-DAG: %[[VAR4:.*]] = "tosa.mul"(%[[VAR2]], %[[VAR0]]) {shift = 0 : i32}
// CHECK-DAG: %[[VAR5:.*]] = "tosa.reduce_sum"(%[[VAR4]]) {axis = 1 : i64}
// CHECK-DAG: %[[VAR6:.*]] = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, -1, 1>}
// CHECK-DAG: %[[VAR7:.*]] = "tosa.reshape"(%[[VAR5]]) {new_shape = array<i64: 1, -1>}
// CHECK-DAG: %[[VAR8:.*]] = "tosa.scatter"(%[[VAR1]], %[[VAR7]], %[[VAR6]])
// CHECK-DAG: %[[VAR9:.*]] = "tosa.reshape"(%[[VAR8]]) {new_shape = array<i64: 1, 48>}
// CHECK: return %[[VAR9]]
func.func @sparse_to_dense(%arg0 : tensor<?x2xi64>, %arg1 : tensor<?xi64>) -> (tensor<1x48xi64>) {
  %0 = arith.constant dense<[1, 48]> : tensor<2xi64>
  %1 = arith.constant dense<-1> : tensor<i64>
  %2 = "tfl.sparse_to_dense"(%arg0, %0, %arg1, %1) : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xi64>, tensor<i64>) -> tensor<1x48xi64>
  func.return %2 : tensor<1x48xi64>
}

// -----

// CHECK-LABEL: @test_arg_max
func.func @test_arg_max(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  // CHECK: %[[ARGMAX:.+]] = "tosa.argmax"(%arg0) {axis = 1 : i64}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_max"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_arg_max_negative_dim
func.func @test_arg_max_negative_dim(%arg0: tensor<13x21x3xf32>) -> tensor<13x21xf32> {
  // CHECK: %[[ARGMAX:.+]] = "tosa.argmax"(%arg0) {axis = 2 : i64}
  %0 = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_max"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<13x21xf32>
  func.return %1 : tensor<13x21xf32>
}

// -----

// CHECK-LABEL: @test_arg_min_f32
func.func @test_arg_min_f32(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  // CHECK: %[[NEG:.+]] = "tosa.negate"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // CHECK: "tosa.argmax"(%[[NEG]]) {axis = 1 : i64}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_arg_min_i32
func.func @test_arg_min_i32(%arg0: tensor<13x21x3xi32>) -> tensor<*xi32> {
  // CHECK: %[[ONE:.+]] = "tosa.const"() {value = dense<-1> : tensor<1x1x1xi32>}
  // CHECK: %[[SUB:.+]] = "tosa.sub"(%[[ONE]], %arg0)
  // CHECK: %[[ARGMAX:.+]] = "tosa.argmax"(%[[SUB]]) {axis = 1 : i64}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_arg_min_ui8
func.func @test_arg_min_ui8(%arg0: tensor<13x21x3xui8>) -> tensor<*xui8> {
  // CHECK: %[[MAX:.+]] = "tosa.const"() {value = dense<255> : tensor<1x1x1xui8>}
  // CHECK: %[[SUB:.+]] = "tosa.sub"(%[[MAX]], %arg0)
  // CHECK: %[[ARGMAX:.+]] = "tosa.argmax"(%[[SUB]]) {axis = 1 : i64}
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tfl.arg_min"(%arg0, %0) : (tensor<13x21x3xui8>, tensor<i32>) -> tensor<*xui8>
  func.return %1 : tensor<*xui8>
}

// -----

// CHECK-LABEL: test_fakequant
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
func.func @test_fakequant(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %2 = "tfl.fake_quant"(%arg0)  {max = 2.000000e+00 : f32, min = -2.000000e+00 : f32, narrow_range = false, num_bits = 16 : i32}  : (tensor<13x21x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_fullyconnected_hybrid
func.func @test_fullyconnected_hybrid(%arg0: tensor<14x19xf32>) -> tensor<*xf32> {
  // This verifies that the constant is decomposed into a dequantization via a
  // cast, subtract, and multiplication.
  // CHECK: "tosa.sub"
  // CHECK: "tosa.fully_connected"
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<36x36x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<28x19xi8>} : () -> tensor<28x19x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_const"() {value = dense<0.0> : tensor<28xf32>} : () -> tensor<28xf32>
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<14x19xf32>, tensor<28x19x!quant.uniform<i8:f32, 1.0>>, tensor<28xf32>) -> tensor<*xf32>
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
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() {value = dense<3.000000e+00> : tensor<1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() {value = dense<4.471500e-02> : tensor<1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_3:.*]] = "tosa.const"() {value = dense<0.797884583> : tensor<1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_4:.*]] = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1x1xf32>}
// CHECK-DAG: %[[VAL_5:.*]] = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x1x1xf32>}
// CHECK: %[[VAL_6:.*]] = "tosa.pow"(%[[VAL_0]], %[[VAL_1]])
// CHECK: %[[VAL_7:.*]] = "tosa.mul"(%[[VAL_6]], %[[VAL_2]]) {shift = 0 : i32}
// CHECK: %[[VAL_8:.*]] = "tosa.add"(%[[VAL_0]], %[[VAL_7]])
// CHECK: %[[VAL_9:.*]] = "tosa.mul"(%[[VAL_8]], %[[VAL_3]]) {shift = 0 : i32}
// CHECK: %[[VAL_10:.*]] = "tosa.tanh"(%[[VAL_9]])
// CHECK: %[[VAL_11:.*]] = "tosa.add"(%[[VAL_10]], %[[VAL_4]])
// CHECK: %[[VAL_12:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_5]]) {shift = 0 : i32}
// CHECK: %[[VAL_13:.*]] = "tosa.mul"(%[[VAL_12]], %[[VAL_11]]) {shift = 0 : i32}
func.func @test_gelu(%arg0: tensor<1x4x8x19xf32>) -> tensor<1x4x8x19xf32> {
  %0 = "tfl.gelu"(%arg0) {approximate = true} : (tensor<1x4x8x19xf32>) -> tensor<1x4x8x19xf32>
  func.return %0 : tensor<1x4x8x19xf32>
}

// -----

// CHECK-LABEL: test_gelu_qi8
// CHECK-SAME: %[[VAR0:.*]]: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>
// CHECK: %[[VAR1:.*]] = "tosa.const"() {value = dense<{{.*}}> : tensor<256xi8>}
// CHECK: %[[VAR2:.*]] = "tosa.table"(%[[VAR0]], %[[VAR1]]) : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>, tensor<256xi8>)
func.func @test_gelu_qi8(%arg0: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>> {
  %0 = "tfl.gelu"(%arg0) {approximate = true} : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.015685562044382095:-1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
  func.return %0 : tensor<1x4x4x4x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
}

// -----

// CHECK-LABEL: mirrorpad_reflect
// CHECK-SAME: %[[VAL_0:.*]]: tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 2, 9>, start = array<i64: 1, 0>} : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_2:.*]] = "tosa.reverse"(%[[VAL_1]]) {axis = 0 : i64} : (tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_3:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 9>, start = array<i64: 2, 0>} : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_4:.*]] = "tosa.concat"(%[[VAL_2]], %[[VAL_0]], %[[VAL_3]]) {axis = 0 : i64} : (tensor<2x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<1x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_5:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 7, 2>, start = array<i64: 0, 1>} : (tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_6:.*]] = "tosa.reverse"(%[[VAL_5]]) {axis = 1 : i64} : (tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_7:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 7, 1>, start = array<i64: 0, 7>} : (tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
// CHECK: %[[VAL_8:.*]] = "tosa.concat"(%[[VAL_6]], %[[VAL_4]], %[[VAL_7]]) {axis = 1 : i64} : (tensor<7x2x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<7x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<7x1x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>)
func.func @mirrorpad_reflect(%arg0: tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>) -> tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>> {
  %0 = "tfl.pseudo_const"() {value = dense<[[2, 1], [2, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tfl.mirror_pad"(%arg0, %0) {mode = #tfl<mirror_pad_attr REFLECT>} : (tensor<4x9x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>, tensor<2x2xi32>) -> tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
  return %1 : tensor<7x12x!quant.uniform<i8:f32, 0.0083315325900912285:-108>>
}

// -----

// CHECK-LABEL: mirrorpad_symmetric
// CHECK-SAME: %[[VAL_0:.*]]: tensor<15x23x2xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 23, 2>, start = array<i64: 0, 0, 0>} : (tensor<15x23x2xf32>)
// CHECK: %[[VAL_2:.*]] = "tosa.concat"(%[[VAL_1]], %[[VAL_0]]) {axis = 0 : i64} : (tensor<1x23x2xf32>, tensor<15x23x2xf32>)
// CHECK: %[[VAL_3:.*]] = "tosa.slice"(%[[VAL_2]]) {size = array<i64: 16, 1, 2>, start = array<i64: 0, 0, 0>} : (tensor<16x23x2xf32>)
// CHECK: %[[VAL_4:.*]] = "tosa.concat"(%[[VAL_3]], %[[VAL_2]]) {axis = 1 : i64} : (tensor<16x1x2xf32>, tensor<16x23x2xf32>)
// CHECK: %[[VAL_5:.*]] = "tosa.slice"(%[[VAL_4]]) {size = array<i64: 16, 24, 1>, start = array<i64: 0, 0, 0>} : (tensor<16x24x2xf32>)
// CHECK: %[[VAL_6:.*]] = "tosa.concat"(%[[VAL_5]], %[[VAL_4]]) {axis = 2 : i64} : (tensor<16x24x1xf32>, tensor<16x24x2xf32>)
func.func @mirrorpad_symmetric(%arg0: tensor<15x23x2xf32>) -> tensor<16x24x3xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<[[1, 0], [1, 0], [1, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tfl.mirror_pad"(%arg0, %0) {mode = #tfl<mirror_pad_attr SYMMETRIC>} : (tensor<15x23x2xf32>, tensor<3x2xi32>) -> tensor<16x24x3xf32>
  return %1 : tensor<16x24x3xf32>
}

// -----

// CHECK-LABEL: test_tfl_custom
// CHECK-SAME: %[[ARG_0:.*]]: tensor<1x64x64x32xf32>
// CHECK: %[[VAL_0:.*]] = "tosa.custom"(%[[ARG_0]]) {config = "TFL", identifier = "MaxPoolingWithArgmax2D", implementation_attrs = "{{.*}}"} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
func.func @test_tfl_custom(%arg0: tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) {
  // custom op for "tfl.max_pooling_with_argmax_2d"(%arg0) {filter_h = 2 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  %0, %1 = "tfl.custom"(%arg0) {custom_option = #tfl<const_bytes : "0x01000000020000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "MaxPoolingWithArgmax2D"} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  func.return %0, %1 : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>
}

// -----
// CHECK-LABEL: test_tfl_while_loop
// CHECK: %[[VAL_0:.*]]: tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["placeholder_0"]}) -> (tensor<1x4x4x4xf32> {tf_saved_model.index_path = ["output_0"]}) {
// CHECK: %[[VAL_1:.*]] = "tosa.const"() {value = dense<2.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK: %[[VAL_2:.*]] = "tosa.while_loop"(%[[VAL_0]]) ({
// CHECK: ^bb0(%[[VAL_3:.*]]: tensor<1x4x4x4xf32>):
// CHECK: %[[VAL_4:.*]] = "tosa.reduce_sum"(%[[VAL_3]]) {axis = 1 : i64} : (tensor<1x4x4x4xf32>) -> tensor<1x1x4x4xf32>
// CHECK: %[[VAL_5:.*]] = "tosa.reduce_sum"(%[[VAL_4]]) {axis = 2 : i64} : (tensor<1x1x4x4xf32>) -> tensor<1x1x1x4xf32>
// CHECK: %[[VAL_6:.*]] = "tosa.reduce_sum"(%[[VAL_5]]) {axis = 3 : i64} : (tensor<1x1x1x4xf32>) -> tensor<1x1x1x1xf32>
// CHECK: %[[VAL_7:.*]] = "tosa.reshape"(%[[VAL_6]]) {new_shape = array<i64: 1>} : (tensor<1x1x1x1xf32>) -> tensor<1xf32>
// CHECK: %[[VAL_8:.*]] = "tosa.greater"(%[[VAL_1]], %[[VAL_7]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
// CHECK: %[[VAL_9:.*]] = "tosa.reshape"(%[[VAL_8]]) {new_shape = array<i64>} : (tensor<1xi1>) -> tensor<i1>
// CHECK: "tosa.yield"(%[[VAL_9]]) : (tensor<i1>) -> ()
// CHECK: }, {
// CHECK: ^bb0(%[[VAL_10:.*]]: tensor<1x4x4x4xf32>):
// CHECK: %[[VAL_11:.*]] = "tosa.sigmoid"(%[[VAL_10]]) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
// CHECK: %[[VAL_12:.*]] = "tosa.add"(%[[VAL_10]], %[[VAL_11]]) : (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
// CHECK: "tosa.yield"(%[[VAL_12]]) : (tensor<1x4x4x4xf32>) -> ()
// CHECK: }) : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
// CHECK: return %[[VAL_13:.*]] : tensor<1x4x4x4xf32>
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

// CHECK-LABEL: test_real
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9x2xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 8, 9, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<1x8x9x2xf32>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 8, 9>} : (tensor<1x8x9x1xf32>) -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_2]] : tensor<1x8x9xf32>
func.func @test_real(%arg0: tensor<1x8x9xcomplex<f32>>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.real"(%arg0) {} : (tensor<1x8x9xcomplex<f32>>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_real_non_complex
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.identity"(%[[VAL_0]]) : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_1]]
func.func @test_real_non_complex(%arg0: tensor<1x8x9xf32>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.real"(%arg0) {} : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_imag
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9x2xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 8, 9, 1>, start = array<i64: 0, 0, 0, 1>} : (tensor<1x8x9x2xf32>) -> tensor<1x8x9x1xf32>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 8, 9>} : (tensor<1x8x9x1xf32>) -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_2]] : tensor<1x8x9xf32>
func.func @test_imag(%arg0: tensor<1x8x9xcomplex<f32>>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.imag"(%arg0) {} : (tensor<1x8x9xcomplex<f32>>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}

// -----

// CHECK-LABEL: test_imag_non_complex
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x8x9xf32>
// CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x8x9xf32>} : () -> tensor<1x8x9xf32>
// CHECK: return %[[VAL_1]] : tensor<1x8x9xf32>
func.func @test_imag_non_complex(%arg0: tensor<1x8x9xf32>) -> (tensor<1x8x9xf32>) {
  %0 = "tfl.imag"(%arg0) {} : (tensor<1x8x9xf32>) -> tensor<1x8x9xf32>
  return %0 : tensor<1x8x9xf32>
}
