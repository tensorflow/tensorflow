// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize | FileCheck %s

// CHECK-LABEL: QuantizeFloatConst
func @QuantizeFloatConst() -> tensor<f32> {
  %0 = constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<f32>
  return %2 : tensor<f32>

// CHECK:  %0 = "tfl.pseudo_qconst"() {qtype = tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<0> : tensor<2x2xi8>}
// CHECK:  %1 = "tfl.dequantize"(%0)
// CHECK:  return %1 : tensor<f32>
}

// CHECK-LABEL: QuantizeDenseFloatConst
func @QuantizeDenseFloatConst() -> tensor<2x2xf32> {
  %0 = constant dense<[[-0.1, 1.0], [1.0, 3.0]]> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK:  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<{{\[\[}}0, -1], {{\[}}-1, -1]]> : tensor<2x2xi8>}
// CHECK:  %1 = "tfl.dequantize"(%0)
// CHECK:  return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: QuantizeSplatFloatConst
func @QuantizeSplatFloatConst() -> tensor<2x2xf32> {
  %0 = constant dense<3.0> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK:  "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}
// CHECK:  %1 = "tfl.dequantize"(%0)
// CHECK:  return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: DequantizeAndQuantize
func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %0 = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}
// CHECK:  return %0 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// CHECK-LABEL: QuantizeConv2D
func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}
// CHECK: %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}
// CHECK: %2 = "tfl.conv_2d"(%arg0, %1, %0) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %2 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}


// CHECK-LABEL: QuantizeDepthwiseConv2D
func @QuantizeDepthwiseConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.depthwise_conv_2d"(%2, %4, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %0 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}
// CHECK: %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}
// CHECK: %2 = "tfl.depthwise_conv_2d"(%arg0, %1, %0) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}
// CHECK: return %2
}

// CHECK-LABEL: QuantizeAveragePool2D
func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  return %1 : tensor<1x1x1x16xf32>

// CHECK: %0 = "tfl.average_pool_2d"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %1 : tensor<1x1x1x16xf32>
}

// CHECK-LABEL: QuantizeReshape2D
func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x36x16xf32>
  return %1 : tensor<1x36x16xf32>

// CHECK: %0 = "tfl.reshape"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32>
// CHECK: return %1 : tensor<1x36x16xf32>
}

// CHECK-LABEL: QuantizeSoftmax
func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %1 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.softmax"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03:-128>>) -> tensor<1x6x6x16xf32>
// CHECK: return %1 : tensor<1x6x6x16xf32>
}

// CHECK-LABEL: QuantizeAdd
func @QuantizeAdd(tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>> {
^bb0(%arg0: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, %arg1: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>) -> tensor<1x56x56x24xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24xf32>
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<1x56x56x24xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>} : (tensor<1x56x56x24xf32>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
  return %3 : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>

// CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>)
// CHECK: return %0 : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
}

// CHECK-LABEL: QuantizeConcat
func @QuantizeConcat(tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>> {
^bb0(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %2 = "tfl.concatenation"(%1, %0) {axis = 0 : i32, fused_activation_function = "NONE"}
// CHECK: return %2 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeConcatRequantize
func @QuantizeConcatRequantize(tensor<2x!quant.uniform<u8:f32, 2.0:128>>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<2x!quant.uniform<u8:f32, 2.0:128>>, %arg1: tensor<2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 2.0:128>>) -> tensor<2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"}
// CHECK: return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeMaxPool2D
func @QuantizeMaxPool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.max_pool_2d"(%0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  return %1 : tensor<1x1x1x16xf32>

// CHECK: %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %1 : tensor<1x1x1x16xf32>
}