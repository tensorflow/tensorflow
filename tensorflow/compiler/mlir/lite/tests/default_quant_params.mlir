// RUN: tf-opt %s --tfl-default-quant --tfl-quantize | FileCheck %s

// CHECK-LABEL: hardcode_all
func @hardcode_all(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x2xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function="NONE"}: (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<2x1x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}
// Quantized tfl.add
// CHECK: %[[add:.*]] = "tfl.add"(%[[q1]], %[[q0]]) {fused_activation_function = "NONE"} : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[add]]) : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: hardcode_input
func @hardcode_input(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x2xf32> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>}: (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x2xf32>
  %4 = "tfl.add"(%1, %arg1) {fused_activation_function="NONE"}: (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %4 : tensor<2x2xf32>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<2x1x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e+00:128>>}
// CHECK: %[[add:.*]] = "tfl.add"(%[[q1]], %[[q0]]) {fused_activation_function = "NONE"} : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e+00:128>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[add]]) : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: hardcode_input_deq
func @hardcode_input_deq(%arg0: tensor<2x2x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<2x1xf32>) -> tensor<2x2xf32> {
  %1 = "tfl.dequantize"(%arg0) : (tensor<2x2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2x2xf32>
  %4 = "tfl.add"(%1, %arg1) {fused_activation_function="NONE"}: (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %4 : tensor<2x2xf32>

// CHECK: %[[q:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<2x1x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}
// CHECK: %[[add:.*]] = "tfl.add"(%arg0, %[[q]]) {fused_activation_function = "NONE"} : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[add]]) : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: hardcode_output
func @hardcode_output(%arg0: tensor<2x2xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x2xf32> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>}: (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>
  %1 = "tfl.quantize"(%arg1) {qtype = tensor<2x1x!quant.uniform<u8:f32, 1.0:128>>}: (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<u8:f32, 1.0:128>>
  %2 = "tfl.dequantize"(%0) : (tensor<2x2x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x2xf32>
  %3 = "tfl.dequantize"(%1) : (tensor<2x1x!quant.uniform<u8:f32, 1.0:128>>) -> tensor<2x1xf32>
  %4 = "tfl.add"(%2, %3) {fused_activation_function="NONE"}: (tensor<2x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>
  return %4 : tensor<2x2xf32>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e+00:128>>}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<2x1x!quant.uniform<u8:f32, 1.000000e+00:128>>}
// CHECK: %[[add:.*]] = "tfl.add"(%[[q0]], %[[q1]]) {fused_activation_function = "NONE"} : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e+00:128>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[add]]) : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: test_conv_2d_add
func @test_conv_2d_add(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>, %arg2: tensor<32x!quant.uniform<i32:f32, 1.0>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>> {
    %0 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x224x224x3xf32>
    %1 = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<32x3x3x3xf32>
    %2 = "tfl.dequantize"(%arg2) : (tensor<32x!quant.uniform<i32:f32, 1.0>>) -> tensor<32xf32>
    %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    %4 = "tfl.pseudo_qconst"() {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>, value = dense<1> : tensor<1x112x112x32xi8>} : () -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>
    %5 = "tfl.dequantize"(%4) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
    %6 = "tfl.add"(%3, %5) {fused_activation_function="NONE"}: (tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    %7 = "tfl.quantize"(%6) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>
    return %7 : tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %arg1, %arg2)
// CHECK-SAME: -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
// CHECK: %[[cst:.*]] = "tfl.pseudo_qconst"()
// CHECK: %[[add:.*]] = "tfl.add"(%[[conv]], %[[cst]])
// CHECK-SAME: -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: return %[[add]]
}

// CHECK-LABEL: test_conv_2d_activation_and_bias
func @test_conv_2d_activation_and_bias(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>, %arg2: tensor<32xf32>) -> tensor<1x112x112x32xf32> {
    %0 = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<32x3x3x3xf32>
    %1 = "tfl.conv_2d"(%arg0, %0, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
    return %1 : tensor<1x112x112x32xf32>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg2) {qtype = tensor<32x!quant.uniform<i32:f32, 0.0078431372549019607>>}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%[[q1]], %arg1, %[[q0]])
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[conv]]) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.0078431372549019607:128>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: test_region
func @test_region(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
    %arg1: tensor<1x!quant.uniform<u8:f32, 0.2:127>>, %arg2: tensor<1x!quant.uniform<u8:f32, 0.4:127>>,
    %arg3: tensor<1xi32>) -> (tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>) -> tensor<128x128xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x!quant.uniform<u8:f32, 0.2:127>>) -> tensor<1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<1x!quant.uniform<u8:f32, 0.4:127>>) -> tensor<1xf32>
  %3 = "tfl.custom_tf"(%0, %1, %2, %arg3) ( {
  ^bb0(%a1: tensor<128x128xf32>, %a2: tensor<1xf32>, %a3: tensor<1xf32>, %a4: tensor<1xi32>):  // no predecessors
    %4 = "tf.LayerNorm"(%a1, %a2, %a3, %a4) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
   "tfl.yield"(%4) : (tensor<128x128xf32>) -> ()
  }) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>} : (tensor<128x128xf32>) -> tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
  return %4 : tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>

// CHECK: "tfl.custom_tf"
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):  // no predecessors
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {_tfl_quant_trait = "fully_quantizable", device = ""} :
// CHECK-SAME: (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
}

