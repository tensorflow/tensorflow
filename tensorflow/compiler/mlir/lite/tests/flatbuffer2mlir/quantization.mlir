// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s

func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x1001xf32> {
// CHECK:   %{{.*}} = "tfl.quantize"(%{{.*}}) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
// The float values here doesn't match exactly because double -> float -> double is lossy
// CHECK-NEXT:   %{{.*}} = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678{{[0-9]*}}:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678{{[0-9]*}}:151>>
// CHECK-NEXT:   %{{.*}} = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092{{[0-9]*}}E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092{{[0-9]*}}E-4>>
// CHECK:   %{{.*}} = "tfl.dequantize"(%{{.*}}) : (tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x1001xf32>

  %cst = constant dense<[1, 1001]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x1001x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x1001x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x1001xf32>
  return %6 : tensor<1x1001xf32>
}
