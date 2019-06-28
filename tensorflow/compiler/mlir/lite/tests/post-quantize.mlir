// RUN: tf-opt %s -tfl-post-quantize | FileCheck %s

func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1001xf32> {
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %4 = "tfl.conv_2d"(%1, %2, %3) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.reshape"(%4) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>
  %6 = "tfl.softmax"(%5) {beta = 1.000000e+00 : f32} : (tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>
  %7 = "tfl.dequantize"(%6) : (tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x1001xf32>
  return %7 : tensor<1x1001xf32>
}

func @main2(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %2 = "tfl.pseudo_input"(%arg1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %4 = tfl.add %1, %3 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %5 = "tfl.dequantize"(%4) : (tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4xf32>
  return %5 : tensor<2x4xf32>
}

// CHECK: func @main(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK-NEXT:  %0 = "tfl.pseudo_input"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK-NEXT:  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}
// CHECK-NEXT:  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>}
// CHECK-NEXT:  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32}
// CHECK-NEXT:  %4 = "tfl.reshape"(%3) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK-NEXT:  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK-NEXT:  return %5 : tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>
// CHECK-NEXT:}

// CHECK: func @main2(%arg0: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>, %arg1: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>> {
// CHECK-NEXT:  %0 = "tfl.pseudo_input"(%arg1) : (tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:  %1 = "tfl.pseudo_input"(%arg0) : (tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:  %2 = tfl.add %1, %0 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:  return %2 : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:}
