// RUN: tf-opt %s -tfl-prepare-quantize | FileCheck %s

// CHECK-LABEL: DequantizeAndQuantize
func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %0 = "tfl.pseudo_qconst"()
// CHECK:  %1 = "tfl.dequantize"(%0)
// CHECK:  %2 = "tfl.quantize"(%1)
// CHECK:  return %2
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

// CHECK: %cst = constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
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

// CHECK: %cst = constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.depthwise_conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// CHECK-LABEL: QuantizeAveragePool2D
func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  return %1 : tensor<1x1x1x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.average_pool_2d"(%0)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x1x1x16xf32>
}

// CHECK-LABEL: QuantizeReshape2D
func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x36x16xf32>
  return %1 : tensor<1x36x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %1 = "tfl.reshape"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x36x16xf32>
// CHECK: %2 = "tfl.quantize"(%1) {qtype = tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>}
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: return %3 : tensor<1x36x16xf32>
}

// CHECK-LABEL: QuantizeSoftmax
func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %1 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
// CHECK: %2 = "tfl.quantize"(%1) {qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>}
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x6x6x16xf32>
}

// CHECK-LABEL: QuantizeConcatOperand0ToAll
func @QuantizeConcatOperand0ToAll(tensor<2x!quant.uniform<u8:f32, 0.1:128>>, tensor<2xf32>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHeCK: return %5 : tensor<2x2xf32>
}

// CHECK-LABEL: QuantizeConcatOperand1ToAll
func @QuantizeConcatOperand1ToAll(tensor<2xf32>, tensor<2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<2xf32>, %arg1: tensor<2x!quant.uniform<u8:f32, 0.1:128>>):
  %0 = "tfl.dequantize"(%arg1) : (tensor<2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2xf32>
  %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg1) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %3 = "tfl.concatenation"(%1, %2) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHECK: return %5 : tensor<2x2xf32>
}

// CHECK-LABEL: QuantizeConcatResToAll
func @QuantizeConcatResToAll(tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %2 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %4 = "tfl.concatenation"(%3, %1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK: %5 = "tfl.quantize"(%4) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %5 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeConcatResToAllNoRequantize
func @QuantizeConcatResToAllNoRequantize(tensor<2x!quant.uniform<u8:f32, 0.1:128>>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %2 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHeCK: return %4 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeConcatResToAllRequantize
func @QuantizeConcatResToAllRequantize(tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>):
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 2.0:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 2.0:128>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x!quant.uniform<u8:f32, 2.0:128>>) -> tensor<2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 2.000000e+00:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 2.000000e+00:128>>
// CHECK %1 = "tfl.quantize"(%0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x!quant.uniform<u8:f32, 2.000000e+00:128>>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK %2 = "tfl.dequantize"(%1) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK %3 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK %4 = "tfl.dequantize"(%3) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK %5 = "tfl.concatenation"(%2, %4) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK %6 = "tfl.quantize"(%5) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK return %6 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeConcatResToAllRequantizeArg
func @QuantizeConcatResToAllRequantizeArg(tensor<2x!quant.uniform<u8:f32, 2.0:128>>, tensor<2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<2x!quant.uniform<u8:f32, 2.0:128>>, %arg1: tensor<2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<2x!quant.uniform<u8:f32, 2.0:128>>) -> tensor<2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK %1 = "tfl.quantize"(%arg0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x!quant.uniform<u8:f32, 2.000000e+00:128>>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK %2 = "tfl.dequantize"(%1) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK %3 = "tfl.quantize"(%arg1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK %4 = "tfl.dequantize"(%3) : (tensor<2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2xf32>
// CHECK %5 = "tfl.concatenation"(%2, %4) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK %6 = "tfl.quantize"(%5) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK return %6 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeChain
func @QuantizeChain(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.average_pool_2d"(%2) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %6 = "tfl.conv_2d"(%5, %4, %cst) {
      dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32
    } : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %7 = "tfl.quantize"(%6) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x6x6x16xf32>
  %9 = "tfl.reshape"(%8) : (tensor<1x6x6x16xf32>) -> tensor<1x36x16xf32>
  %10 = "tfl.softmax"(%9) {beta = 1.000000e+00 : f32} : (tensor<1x36x16xf32>) -> tensor<1x36x16xf32>
  return %10 : tensor<1x36x16xf32>

// CHECK: %cst = constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>)
// CHECK: %5 = "tfl.average_pool_2d"(%2)
// CHECK: %6 = "tfl.quantize"(%5) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>}
// CHECK: %7 = "tfl.dequantize"(%6) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %8 = "tfl.conv_2d"(%7, %4, %1)
// CHECK: %9 = "tfl.quantize"(%8) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>}
// CHECK: %10 = "tfl.dequantize"(%9) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %11 = "tfl.reshape"(%10)
// CHECK: %12 = "tfl.quantize"(%11) {qtype = tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>}
// CHECK: %13 = "tfl.dequantize"(%12) : (tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %14 = "tfl.softmax"(%13)
// CHECK: %15 = "tfl.quantize"(%14) {qtype = tensor<1x36x16x!quant.uniform<u8:f32, 3.906250e-03>>}
// CHECK: %16 = "tfl.dequantize"(%15) : (tensor<1x36x16x!quant.uniform<u8:f32, 3.906250e-03>>)
// CHECK: return %16 : tensor<1x36x16xf32>
}

// CHECK-LABEL: QuantizeConstant
func @QuantizeConstant() -> tensor<2x3xf32> {
  %cst = constant dense<[[-3.0, -1.0, 0.0], [0.0, 1.0, 3.0]]> : tensor<2x3xf32>
  return %cst : tensor<2x3xf32>

// CHECK: %cst = constant dense{{.*}}tensor<2x3xf32>
// CHECK: %0 = "tfl.quantize"(%cst) {qtype = tensor<2x3x!quant.uniform<u8:f32, 0.023529411764705882:128>>}
// CHECK: %1 = "tfl.dequantize"(%0)
// CHECK: return %1 : tensor<2x3xf32>
}