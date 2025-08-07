// RUN: litert-opt --tfl-decompose-hybrid-quantization --verify-each %s | FileCheck %s

// CHECK-LABEL: @test_conv2d_float
func.func @test_conv2d_float(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x16xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_const"() <{value = dense<42> : tensor<16x1x1x8xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_const"() <{value = dense<1> : tensor<16x1x1x8xi8>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.conv_2d"(%arg0, %[[VAL0]], %[[VAL1]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL2]]
  %0 = "tfl.pseudo_const"() {value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8xf32>
  %1 = "tfl.pseudo_const"() {value = dense<1> : tensor<16x1x1x8xi8>} : () -> tensor<16xf32>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  func.return %2 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: @test_conv2d_qi8
func.func @test_conv2d_qi8(%arg0: tensor<1x32x32x8x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<{{.+}}>, value = dense<42> : tensor<16x1x1x8xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<{{.+}}>, value = dense<0> : tensor<16xi32>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.conv_2d"(%arg0, %0, %1) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL2]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.0>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i8:f32, 1.0>>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_conv2d_qi16
func.func @test_conv2d_qi16(%arg0: tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>> {
  // CHECK-DAG: %[[BIAS:.+]] = arith.constant dense<0> : tensor<16xi64>
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<{{.+}}>, value = dense<42> : tensor<16x1x1x8xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.conv_2d"(%arg0, %[[VAL0]], %[[BIAS]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL1]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "arith.constant"() {value = dense<0> : tensor<16xi64>} : () -> tensor<16xi64>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8x!quant.uniform<i16:f32, 1.0>>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16xi64>) -> tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i16:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_conv2d_replace_qi8
func.func @test_conv2d_replace_qi8(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<{{.+}}>, value = dense<42> : tensor<16x1x1x8xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<{{.+}}>, value = dense<0> : tensor<16xi32>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL0]])
  // CHECK-DAG: %[[VAL3:.+]] = "tfl.dequantize"(%[[VAL1]])
  // CHECK-DAG: %[[VAL4:.+]] = "tfl.conv_2d"(%arg0, %[[VAL2]], %[[VAL3]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK-DAG: %[[VAL5:.+]] = "tfl.quantize"(%4) <{qtype = tensor<1x32x32x16x!quant.uniform<i8:f32, 1.000000e+00>>}>
  // CHECK: return %[[VAL5]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.0>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>>
  func.return %2 : tensor<1x32x32x16x!quant.uniform<i8:f32, 1.0>>
}

// -----

// CHECK-LABEL: @test_conv2d_replace_float
func.func @test_conv2d_replace_float(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x16xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<16x1x1x8x{{.+}}>, value = dense<42> : tensor<16x1x1x8xi8>}> : () -> tensor<16x1x1x8x!quant.uniform<{{.+}}>>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<16x{{.+}}>, value = dense<0> : tensor<16xi32>}> : () -> tensor<16x!quant.uniform<{{.+}}>>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL0]])
  // CHECK-DAG: %[[VAL3:.+]] = "tfl.dequantize"(%[[VAL1]])
  // CHECK-DAG: %[[VAL4:.+]] = "tfl.conv_2d"(%arg0, %[[VAL2]], %[[VAL3]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL4]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<16x1x1x8xi8>} : () -> tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.0>>, value = dense<0> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8x!quant.uniform<i8:f32, 1.0>>, tensor<16x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x32x32x16xf32>
  func.return %2 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: @test_conv3d_float
func.func @test_conv3d_float(%arg0: tensor<1x32x32x32x8xf32>) -> tensor<1x32x32x32x16xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<16xf32>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<1x1x8x16x!quant.uniform<{{.+}}>>, value = dense<42> : tensor<1x1x1x8x16xi8>}>
  // CHECK: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL1]]) : (tensor<1x1x1x8x16x!quant.uniform<{{.+}}>>) -> tensor<1x1x1x8x16xf32>
  // CHECK: %[[VAL3:.+]] = "tfl.conv_3d"(%arg0, %[[VAL2]], %[[VAL0]]) <{dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL3]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x1x8x16x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<1x1x1x8x16xi8>} : () -> tensor<1x1x1x8x16x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_const"() { value = dense<1.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %2 = "tfl.conv_3d"(%arg0, %0, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, dilation_d_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, stride_d = 1 : i32} : (tensor<1x32x32x32x8xf32>, tensor<1x1x1x8x16x!quant.uniform<i8:f32, 1.0>>, tensor<16xf32>) -> tensor<1x32x32x32x16xf32>
  func.return %2 : tensor<1x32x32x32x16xf32>
}

// -----

// CHECK-LABEL: @test_transpose_conv2d
func.func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x16xf32> {
  // CHECK-DAG: %[[SHAPE:.+]] = "tfl.pseudo_const"() <{value = dense<[1, 32, 32, 16]>
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<16x{{.+}}>, value = dense<1> : tensor<16xi32>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<16x{{.+}}>, value = dense<2> : tensor<16xi32>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL0]])
  // CHECK-DAG: %[[VAL3:.+]] = "tfl.dequantize"(%[[VAL1]])
  // CHECK-DAG: %[[VAL4:.+]] = "tfl.transpose_conv"(%[[SHAPE]], %[[VAL2]], %arg0, %[[VAL3]]) <{fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}>
  // CHECK: return %[[VAL4]]
  %0 = "tfl.pseudo_const"() { value = dense<[1, 32, 32, 16]> : tensor<4xi32> } : () -> tensor<4xi32>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.0>>, value = dense<1> : tensor<16xi32>} : () -> tensor<16x1x1x8x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<16x!quant.uniform<i32:f32, 1.0>>, value = dense<2> : tensor<16xi32>} : () -> tensor<16x!quant.uniform<i32:f32, 1.0>>
  %3 = "tfl.transpose_conv"(%0, %1, %arg0, %2)  {fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}  : (tensor<4xi32>, tensor<16x1x1x8x!quant.uniform<i32:f32, 1.0>>, tensor<1x32x32x8xf32>, tensor<16x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x32x32x16xf32>
  func.return %3 : tensor<1x32x32x16xf32>
}

// -----

// CHECK-LABEL: @test_depthwise_conv2d_replace_float
func.func @test_depthwise_conv2d_replace_float(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<{{.+}}>>, value = dense<42> : tensor<32x3x3x3xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<{{.+}}>>, value = dense<0> : tensor<32xi32>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL0]]) : (tensor<32x3x3x3x!quant.uniform<{{.+}}>>)
  // CHECK-DAG: %[[VAL3:.+]] = "tfl.dequantize"(%[[VAL1]]) : (tensor<32x!quant.uniform<{{.+}})
  // CHECK-DAG: %[[VAL4:.+]] = "tfl.depthwise_conv_2d"(%arg0, %[[VAL2]], %[[VAL3]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}>
  // CHECK: return %[[VAL4]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.0>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.depthwise_conv_2d"(%arg0, %0, %1) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3x!quant.uniform<i8:f32, 1.0>>, tensor<32x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  func.return %2 : tensor<1x112x112x32xf32>
}

// -----

// CHECK-LABEL: @test_fullyconnected_replace_float
func.func @test_fullyconnected_replace_float(%arg0: tensor<4x256x6x6xf32>) -> tensor<4x256x36xf32> {
  // CHECK-DAG: %[[VAL0:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<36x36x!quant.uniform<{{.+}}>>, value = dense<42> : tensor<36x36xi8>}>
  // CHECK-DAG: %[[VAL1:.+]] = "tfl.pseudo_qconst"() <{qtype = tensor<36x!quant.uniform<{{.+}}>>, value = dense<0> : tensor<36xi32>}>
  // CHECK-DAG: %[[VAL2:.+]] = "tfl.dequantize"(%[[VAL0]]) : (tensor<36x36x!quant.uniform<i8:f32, 1.000000e+00>>)
  // CHECK-DAG: %[[VAL3:.+]] = "tfl.dequantize"(%[[VAL1]]) : (tensor<36x!quant.uniform<i32:f32, 1.000000e+00>>)
  // CHECK: %[[VAL4:.+]] = "tfl.fully_connected"(%arg0, %[[VAL2]], %[[VAL3]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
  // CHECK: return %[[VAL4]]
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<36x36x!quant.uniform<i8:f32, 1.0>>, value = dense<42> : tensor<36x36xi8>} : () -> tensor<36x36x!quant.uniform<i8:f32, 1.0>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<36x!quant.uniform<i32:f32, 1.0>>, value = dense<0> : tensor<36xi32>} : () -> tensor<36x!quant.uniform<i32:f32, 1.0>>
  %2 = "tfl.fully_connected"(%arg0, %0, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x256x6x6xf32>, tensor<36x36x!quant.uniform<i8:f32, 1.0>>, tensor<36x!quant.uniform<i32:f32, 1.0>>) -> tensor<4x256x36xf32>
  func.return %2 : tensor<4x256x36xf32>
}
