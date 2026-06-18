// RUN: litert-opt %s -split-input-file -tfl-prepare-quantize="quantize-allowlist=quantize_float_placeholder_only,not_reset_input" | FileCheck %s
// RUN: litert-opt %s -split-input-file -tfl-prepare-quantize="disable-set-input-nodes-quantization-params=true" | FileCheck --check-prefix=MixedPrecision %s
// RUN: litert-opt %s -split-input-file -tfl-prepare-quantize="qdq-conversion-mode=Static" | FileCheck --check-prefix=QDQ %s

// -----

// CHECK-LABEL: main
// Uses `main` function to match the default target function of QuantSpecs and
// execute the production code path.
func.func @main(%arg0: tensor<2x1xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x4xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x1x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<i16:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x1xf32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<2x3x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i16:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x3x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x3xf32>)
  %4 = "tfl.concatenation"(%1, %3) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  func.return %4: tensor<2x4xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[q_0:.*]] = "tfl.quantize"(%arg1)
// CHECK-NEXT: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK-NEXT: %[[c:.*]] = "tfl.concatenation"(%[[dq]], %[[dq_0]])
// CHECK-NEXT: %[[q_1:.*]] = "tfl.quantize"(%[[c]])
// CHECK-NEXT: %[[dq_1:.*]] = "tfl.dequantize"(%[[q_1]])
// CHECK-NEXT: return %[[dq_1:.*]]
}

// -----

// MixedPrecision-LABEL: partial_quantized
func.func @partial_quantized(%arg0: tensor<2x1xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x1x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<i16:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x1xf32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<2x3x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i16:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x3x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x3xf32>)
  %4 = "tfl.concatenation"(%1, %3) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  %5 = "tfl.add"(%4, %arg2) {fused_activation_function = "NONE"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %5: tensor<2x4xf32>

// MixedPrecision-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0)
// MixedPrecision-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// MixedPrecision-NEXT: %[[q_0:.*]] = "tfl.quantize"(%arg1)
// MixedPrecision-NEXT: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// MixedPrecision-NEXT: %[[c:.*]] = "tfl.concatenation"(%[[dq]], %[[dq_0]])
// MixedPrecision-NEXT: %[[q_1:.*]] = "tfl.quantize"(%[[c]])
// MixedPrecision-NEXT: %[[dq_1:.*]] = "tfl.dequantize"(%[[q_1]])
// MixedPrecision-NEXT: %[[v:.*]] = tfl.add %[[dq_1]], %arg2
// MixedPrecision-NEXT: return %[[v:.*]]
}

// -----

// CHECK-LABEL: quantize_float_placeholder_only
func.func @quantize_float_placeholder_only(%arg0: tensor<f32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xf32>) -> (tensor<f32>, tensor<2x3xi32>, tensor<2x3xf32>) {
  func.return %arg0, %arg1, %arg2: tensor<f32>, tensor<2x3xi32>, tensor<2x3xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[q_0:.*]] = "tfl.quantize"(%arg2)
// CHECK-NEXT: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK-NEXT: %[[dq]], %arg1, %[[dq_0]]
}

// -----

// CHECK-LABEL: not_reset_input
func.func @not_reset_input(%arg0: tensor<f32>) -> (tensor<!quant.uniform<i16:f32, 1.0>>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<!quant.uniform<i16:f32, 1.0>>} : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.0>>
  func.return %0: tensor<!quant.uniform<i16:f32, 1.0>>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<!quant.uniform<i16:f32, 1.000000e+00>>}>
// CHECK-NEXT: return %[[q]]
}

// -----

// CHECK-LABEL: DequantizeAndQuantize
func.func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %0 = "tfl.pseudo_qconst"()
// CHECK:  %1 = "tfl.dequantize"(%0)
// CHECK:  %2 = "tfl.quantize"(%1)
// CHECK:  return %2
}

// -----

// CHECK-LABEL: prepareStatistics
func.func @prepareStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  %1 = "quantfork.stats"(%0) {
    layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  func.return %1 : tensor<8x4x3xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<8x4x3x!quant.uniform<u8:f32, 0.0078431372549019607:128>>}> {volatile}
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]])
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[dq1]]) <{qtype = tensor<8x4x3x!quant.uniform<u8:f32:2, {0.0078431372549019607:128,0.062745098039215685:128,0.0039215686274509803:128}>>}> {volatile}
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
// CHECK: return %[[dq2]]
}

// -----

// CHECK-LABEL: prepareNarrowStatistics
func.func @prepareNarrowStatistics(%arg0: tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
  %0 = "quantfork.stats"(%arg0) {
    layerStats = dense<[-1.0e-9, 1.0e-9]> : tensor<2xf32>
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  func.return %0 : tensor<8x4x3xf32>

// CHECK: %[[q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<8x4x3x!quant.uniform<u8:f32, 7.8509803919350426E-9:128>>}> {volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: return %[[dq]]
}

// -----

// CHECK-LABEL: QuantizeConv2DPerChannel
func.func @QuantizeConv2DPerChannel(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = arith.constant dense<1.0> : tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}> {volatile}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2DPerChannelConst
func.func @QuantizeConv2DPerChannelConst(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<32xf32>} : () -> tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.5>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.500000e+00,3.000000e+00,4.500000e+00}>>}> {volatile}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2DPerChannels
func.func @QuantizeConv2DPerChannels(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32:3, {1.0,2.0,3.0}>>,
                               %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x112x112x32xf32> {
  %bias = arith.constant dense<1.0> : tensor<32xf32>
  %input = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32:3, {1.0,2.0,3.0}>>) -> tensor<1x224x224x3xf32>
  %weight = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32:3, {1.0,2.0,3.0}>>) -> tensor<32x3x3x3xf32>
  %conv = "tfl.conv_2d"(%input, %weight, %bias) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %conv : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK-NEXT: %[[qbias:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32:0, {1.000000e+00,4.000000e+00,9.000000e+00}>>}> {volatile}
// CHECK-NEXT: %[[bias:.*]] = "tfl.dequantize"(%[[qbias]])
// CHECK-NEXT: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[w:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[in]], %[[w]], %[[bias]])
// CHECK-NEXT: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>} : () -> tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x12xf32>
  %5 = "tfl.fully_connected"(%2, %4, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.fully_connected"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeDepthwiseConv2D
func.func @QuantizeDepthwiseConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.depthwise_conv_2d"(%2, %4, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: %5 = "tfl.depthwise_conv_2d"(%2, %4, %1)
// CHECK: %6 = "tfl.quantize"(%5)
// CHECK: return %6
}

// -----

// CHECK-LABEL: QuantizeAveragePool2D
func.func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %1 : tensor<1x1x1x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.average_pool_2d"(%0)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: QuantizeMaximum
func.func @QuantizeMaximum(tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.maximum"(%0, %1) : (tensor<1x6x6x16xf32>, tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %2 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%arg1)
// CHECK: %2 = "tfl.maximum"(%0, %1)
// CHECK: %3 = "tfl.quantize"(%2)
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: return %4 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeMinimum
func.func @QuantizeMinimum(tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.minimum"(%0, %1) : (tensor<1x6x6x16xf32>, tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %2 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%arg1)
// CHECK: %2 = "tfl.minimum"(%0, %1)
// CHECK: %3 = "tfl.quantize"(%2)
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: return %4 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeSlice
func.func @QuantizeSlice(tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32> {
^bb0(%arg0: tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x3x5xf32>
  %1 = "tfl.slice"(%0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %1 : tensor<?x3x5xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.slice"(%0, %arg1, %arg2)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<?x3x5xf32>
}

// -----

// CHECK-LABEL: QuantizeStridedSlice
func.func @QuantizeStridedSlice(tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32> {
^bb0(%arg0: tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>) -> tensor<12x2x2x5xf32>
  %1 = "tfl.strided_slice"(%0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  func.return %1 : tensor<1x2x2x5xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.strided_slice"(%0, %arg1, %arg2, %arg3)
// CHECK: %2 = "tfl.quantize"(%1) <{qtype = tensor<1x2x2x5x!quant.uniform<u8:f32, 1.000000e-01>>}> {volatile}
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x2x2x5xf32>
}

// -----

// CHECK-LABEL: QuantizePad
func.func @QuantizePad(tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, tensor<3x2xi32>) -> tensor<?xf32> {
^bb0(%arg0: tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<3x2xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x1x3xf32>
  %1 = "tfl.pad"(%0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.pad"(%0, %arg1)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: QuantizePad2
// only the second tfl.pad has sufficient quantization information.
func.func @QuantizePad2(tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, tensor<2x1x3xf32>, tensor<3x2xi32>) -> (tensor<?xf32>, tensor<?xf32>) {
^bb0(%arg0: tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<2x1x3xf32>, %arg2: tensor<3x2xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x1x3xf32>
  %1 = "tfl.pad"(%arg1, %arg2) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  %2 = "tfl.pad"(%0, %arg2) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  func.return %1, %2 : tensor<?xf32>, tensor<?xf32>

// CHECK: %[[dq:.*]] = "tfl.dequantize"(%arg0)
// CHECK: %[[pad1:.*]] = "tfl.pad"(%arg1, %arg2)
// CHECK: %[[pad2:.*]] = "tfl.pad"(%[[dq]], %arg2)
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[pad2]])
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
}

// -----

// CHECK-LABEL: QuantizeReshape2D
func.func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  func.return %1 : tensor<1x36x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %1 = "tfl.reshape"(%0, %{{.*}}) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
// CHECK: %2 = "tfl.quantize"(%1) <{qtype = tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>}> {volatile}
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: return %3 : tensor<1x36x16xf32>
}

// -----

// CHECK-LABEL: QuantizeSoftmax
func.func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.softmax"(%0) <{beta = 1.000000e+00 : f32}> : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
// CHECK: %2 = "tfl.quantize"(%1) <{qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>}> {volatile}
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeLogistic
func.func @QuantizeLogistic(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
// CHECK: %2 = "tfl.quantize"(%1) <{qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>}> {volatile}
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x6x6x16xf32>
// CHECK: return %3 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: NotRescaleLogistic
func.func @NotRescaleLogistic(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>> {
  %0 = "tfl.logistic"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
  func.return %0 : tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>

// CHECK:  %[[log:.*]] = "tfl.logistic"(%arg0)
// CHECK: return %[[log]]
}

// -----

// QDQ-LABEL: QDQNoQuantizeLogistic
func.func @QDQNoQuantizeLogistic(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// QDQ: %0 = "tfl.dequantize"(%arg0)
// QDQ: %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
// QDQ-NOT:"tfl.quantize"
// QDQ: return %1 : tensor<1x6x6x16xf32>
}

// -----

// QDQ-LABEL: QDQNoQuantizeSoftmax
func.func @QDQNoQuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// QDQ: %0 = "tfl.dequantize"(%arg0)
// QDQ: %1 = "tfl.softmax"(%0) <{beta = 1.000000e+00 : f32}> : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
// QDQ-NOT: "tfl.quantize"
// QDQ: return %1 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeL2Norm
func.func @QuantizeL2Norm(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x6x6x16xf32> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.l2_normalization"(%0) {fused_activation_function = "NONE"} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// CHECK: %[[in:.*]] = "tfl.dequantize"(%arg0)
// CHECK: %[[l2:.*]] = "tfl.l2_normalization"(%[[in]])
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[l2]]) <{qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>}> {volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: return %[[dq]] : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: NotQuantizeConcatConstantOperand
func.func @NotQuantizeConcatConstantOperand(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  %0 = arith.constant dense<1.0> : tensor<1x2xf32>
  %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<1x2xf32>
// CHECK-NEXT: %[[cc:.*]] = "tfl.concatenation"(%arg0, %[[cst]])
// CHECK-NEXT: return %[[cc]]
}

// -----

// CHECK-LABEL: QuantizeConcatOperand0ToAll
func.func @QuantizeConcatOperand0ToAll(tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x2xf32>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHECK: return %5 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: QuantizeConcatOperand1ToAll
func.func @QuantizeConcatOperand1ToAll(tensor<1x2xf32>, tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>):
  %0 = "tfl.dequantize"(%arg1) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg1) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%1, %2) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHECK: return %5 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAll
func.func @QuantizeConcatResToAll(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %4 = "tfl.concatenation"(%3, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %5 = "tfl.quantize"(%4) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %5 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllNoRequantize
func.func @QuantizeConcatResToAllNoRequantize(tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %2 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %4 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllRequantize
func.func @QuantizeConcatResToAllRequantize(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>

// CHECK: %[[Q1:.*]] =  "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %[[DQ1:.*]] = "tfl.dequantize"(%[[Q1]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[Q0:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:128>>}> : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:128>>
// CHECK: %[[RQ0:.*]] = "tfl.quantize"(%[[Q0]]) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:128>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: %[[DQ0:.*]] = "tfl.dequantize"(%[[RQ0]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[CONC:.*]] = "tfl.concatenation"(%[[DQ0]], %[[DQ1]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[Q:.*]] = "tfl.quantize"(%[[CONC]]) <{qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: return %[[Q]] : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllRequantizeArg
func.func @QuantizeConcatResToAllRequantizeArg(tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>, %arg1: tensor<1x2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>

// CHECK: %[[Q1:.*]] =  "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %[[DQ1:.*]] = "tfl.dequantize"(%[[Q1]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[RQ0:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:128>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: %[[DQ0:.*]] = "tfl.dequantize"(%[[RQ0]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[CONC:.*]] = "tfl.concatenation"(%[[DQ0]], %[[DQ1]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[Q:.*]] = "tfl.quantize"(%[[CONC]]) <{qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: return %[[Q]] : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: NotRequantizeAlreadyQuantizedModel
func.func @NotRequantizeAlreadyQuantizedModel(%arg0: tensor<1x73x73x64x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<1x147x147x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>> {
  %9 = "tfl.max_pool_2d"(%arg1) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x147x147x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x96x!quant.uniform<u8:f32, 2.0>>
  %10 = "tfl.concatenation"(%arg0, %9) {axis = 3 : i32, fused_activation_function = "NONE"} : (tensor<1x73x73x64x!quant.uniform<u8:f32, 1.0>>, tensor<1x73x73x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>>
  func.return %10 : tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[max:.*]] = "tfl.max_pool_2d"(%arg1) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x147x147x96x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<1x73x73x96x!quant.uniform<u8:f32, 2.000000e+00>>
// CHECK: %[[cat:.*]] = "tfl.concatenation"(%arg0, %[[max]]) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x73x73x64x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<1x73x73x96x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: return %[[cat]] : tensor<1x73x73x160x!quant.uniform<u8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: QuantizeChain
func.func @QuantizeChain(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.average_pool_2d"(%2) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %6 = "tfl.conv_2d"(%5, %4, %cst) {
      dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32
    } : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %7 = "tfl.quantize"(%6) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x6x6x16xf32>
  %9 = "tfl.reshape"(%8, %cst_0) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  %10 = "tfl.softmax"(%9) {beta = 1.000000e+00 : f32} : (tensor<1x36x16xf32>) -> tensor<1x36x16xf32>
  func.return %10 : tensor<1x36x16xf32>

// CHECK: %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>)
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %3 = "tfl.pseudo_qconst"()
// CHECK: %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>)
// CHECK: %5 = "tfl.average_pool_2d"(%2)
// CHECK: %6 = "tfl.quantize"(%5) <{qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>}> {volatile}
// CHECK: %7 = "tfl.dequantize"(%6) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %8 = "tfl.conv_2d"(%7, %4, %1)
// CHECK: %9 = "tfl.quantize"(%8) <{qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>}>
// CHECK: %10 = "tfl.dequantize"(%9) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %11 = "tfl.reshape"(%10, %{{.*}})
// CHECK: %12 = "tfl.quantize"(%11) <{qtype = tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>}> {volatile}
// CHECK: %13 = "tfl.dequantize"(%12) : (tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %14 = "tfl.softmax"(%13)
// CHECK: %15 = "tfl.quantize"(%14) <{qtype = tensor<1x36x16x!quant.uniform<u8:f32, 3.906250e-03>>}> {volatile}
// CHECK: %16 = "tfl.dequantize"(%15) : (tensor<1x36x16x!quant.uniform<u8:f32, 3.906250e-03>>)
// CHECK: return %16 : tensor<1x36x16xf32>
}

// -----

// CHECK-LABEL: QuantizeConstant
func.func @QuantizeConstant() -> tensor<2x3xf32> {
  %cst = arith.constant dense<[[-3.0, -1.0, 0.0], [0.0, 1.0, 3.0]]> : tensor<2x3xf32>
  func.return %cst : tensor<2x3xf32>

// CHECK: %cst = arith.constant dense{{.*}}tensor<2x3xf32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<2x3x!quant.uniform<u8:f32, 0.023529411764705882:128>>}> {volatile}
// CHECK: %1 = "tfl.dequantize"(%0)
// CHECK: return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: NotQuantizeNoneType
func.func @NotQuantizeNoneType() -> none {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  func.return %cst : none

// CHECK-NEXT:  %[[cst:.*]] = "tfl.no_value"() <{value}> : () -> none
// CHECK-NEXT:  return %[[cst]]
}

// -----

// CHECK-LABEL: QuantizeZeroSplat
func.func @QuantizeZeroSplat() -> tensor<2x3xf32> {
  %cst = arith.constant dense<0.0> : tensor<2x3xf32>
  func.return %cst : tensor<2x3xf32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<2x3x!quant.uniform<u8:f32, 3.9215686274509805E-9:127>>}> {volatile}
}

// -----

// CHECK-LABEL: QuantizeZeroScalar
func.func @QuantizeZeroScalar() -> tensor<f32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  func.return %cst : tensor<f32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<!quant.uniform<u8:f32, 3.9215686274509805E-9:127>>}> {volatile}
}

// -----

// CHECK-LABEL: QuantizePositiveSplat
func.func @QuantizePositiveSplat() -> tensor<2x3xf32> {
  %cst = arith.constant dense<25.4> : tensor<2x3xf32>
  func.return %cst : tensor<2x3xf32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<2.540000e+01> : tensor<2x3xf32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<2x3x!quant.uniform<u8:f32, 0.099607841641295186>>}> {volatile}
}

// -----

// CHECK-LABEL: QuantizePositiveScalar
func.func @QuantizePositiveScalar() -> tensor<f32> {
  %cst = arith.constant dense<2.54> : tensor<f32>
  func.return %cst : tensor<f32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<2.540000e+00> : tensor<f32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<!quant.uniform<u8:f32, 0.0099607841641295193>>}> {volatile}
}

// -----

// CHECK-LABEL: QuantizeNegativeSplat
func.func @QuantizeNegativeSplat() -> tensor<2x3xf32> {
  %cst = arith.constant dense<-2.54> : tensor<2x3xf32>
  func.return %cst : tensor<2x3xf32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<-2.540000e+00> : tensor<2x3xf32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<2x3x!quant.uniform<u8:f32, 0.0099607841641295193:255>>}> {volatile}
}

// -----

// CHECK-LABEL: QuantizeNegativeScalar
func.func @QuantizeNegativeScalar() -> tensor<f32> {
  %cst = arith.constant dense<-25.4> : tensor<f32>
  func.return %cst : tensor<f32>

// CHECK-NEXT:  %[[cst:.*]] = arith.constant dense<-2.540000e+01> : tensor<f32>
// CHECK-NEXT:  "tfl.quantize"(%[[cst]]) <{qtype = tensor<!quant.uniform<u8:f32, 0.099607841641295186:255>>}> {volatile}
}

// -----

// Make sure biases are not shared.
// CHECK-LABEL: QuantizeSharedBiases
func.func @QuantizeSharedBiases(
    %arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x224x224x3xf32>
  %2 = "tfl.dequantize"(%arg1) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<32x3x3x3xf32>
  %conv1 = "tfl.conv_2d"(%1, %2, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %3 = "tfl.quantize"(%conv1) {qtype = tensor<1x112x112x32xf32>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>

  %4 = "tfl.dequantize"(%3) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %5 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%4, %5, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %6 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

  func.return %6 : tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[cst_0:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q_0:.*]] = "tfl.quantize"(%[[cst_0]])
// CHECK: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]]) : (tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>)
// CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]])
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]]) : (tensor<32x!quant.uniform<i32:f32, 1.000000e+00>>)
// CHECK: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[dq]])
// CHECK: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[dq_0]])
}

// -----

// Make sure biases are not shared.
// CHECK-LABEL: QuantizeSharedBiases2
func.func @QuantizeSharedBiases2(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %1 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %add = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %3 = "tfl.quantize"(%add) {qtype = tensor<32xf32>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  %5 = "tfl.dequantize"(%arg1) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%5, %6, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %7 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>
  func.return %3, %7 : tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]])
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[cst_0:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q_0:.*]] = "tfl.quantize"(%[[cst_0]])
// CHECK: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK: %{{.*}} = tfl.add %{{.*}}, %[[dq_0]]
// CHECK: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[dq]])
}

// -----

// Make sure biases are not shared.
// CHECK-LABEL: QuantizeSharedBiases3
func.func @QuantizeSharedBiases3(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>,
    %arg2: tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>) {
  %cst = arith.constant dense<1.0> : tensor<32xf32>
  %5 = "tfl.dequantize"(%arg1) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 1.0>>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.dequantize"(%arg2) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 2.0>>) -> tensor<32x3x3x3xf32>
  %conv2 = "tfl.conv_2d"(%5, %6, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x112x112x32xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x56x56x32xf32>
  %7 = "tfl.quantize"(%conv2) {qtype = tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>} : (tensor<1x56x56x32xf32>) -> tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

  %1 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %add = "tfl.add"(%1, %cst) {fused_activation_function = "NONE"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %3 = "tfl.quantize"(%add) {qtype = tensor<32xf32>} : (tensor<32xf32>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %3, %7 : tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<1x56x56x32x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[cst]]) <{qtype = tensor<32x!quant.uniform<i32:f32, 2.000000e+00>>}> {volatile}
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: %[[cst_0:.*]] = arith.constant dense<1.000000e+00> : tensor<32xf32>
// CHECK: %[[q_0:.*]] = "tfl.quantize"(%[[cst_0]])
// CHECK: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK: %{{.*}} = "tfl.conv_2d"(%{{.*}}, %{{.*}}, %[[dq]])
// CHECK: %{{.*}} = tfl.add %{{.*}}, %[[dq_0]]
}

// -----

// Make sure constants are duplicataed for all users.
// CHECK-LABEL: QuantizeSharedConstantsMultipleUsers
func.func @QuantizeSharedConstantsMultipleUsers(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<32x!quant.uniform<u8:f32, 2.0>>,
    %arg2: tensor<32x!quant.uniform<u8:f32, 3.0>>,
    %arg3: tensor<32x!quant.uniform<u8:f32, 4.0>>) -> (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) {
  %cst = arith.constant dense<0.0> : tensor<32xf32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<32x!quant.uniform<u8:f32, 2.0>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<32x!quant.uniform<u8:f32, 3.0>>) -> tensor<32xf32>
  %3 = "tfl.dequantize"(%arg3) : (tensor<32x!quant.uniform<u8:f32, 4.0>>) -> tensor<32xf32>

  %4 = "tfl.minimum"(%0, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %5 = "tfl.minimum"(%1, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %6 = "tfl.minimum"(%2, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %7 = "tfl.minimum"(%3, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  func.return %4, %5, %6, %7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>

// CHECK-DAG: %[[cst1:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst2:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst3:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 3.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst4:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 4.000000e+00>>) -> tensor<32xf32>
// CHECK-NOT: BLOCK_DAG
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst1]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst2]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst3]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst4]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
}

// -----

// Make sure quantization parameters are scanned from weight, but not from bias.
// CHECK-LABEL: QuantizeWeight
func.func @QuantizeWeight(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  %w = arith.constant dense<1.0> : tensor<32x3x3x3xf32>
  %b = arith.constant dense<-1.0> : tensor<32xf32>
  %c = "tfl.conv_2d"(%arg0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
  : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  func.return %c : tensor<1x112x112x32xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<32x3x3x3xf32>
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.003937007874015748:1>>}> {volatile} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.003937007874015748:1>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]]) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.003937007874015748:1>>) -> tensor<32x3x3x3xf32>
// CHECK: %[[b:.*]] = arith.constant dense<-1.000000e+00> : tensor<32xf32>
// CHECK: %[[c:.*]] = "tfl.conv_2d"(%arg0, %[[dq]], %[[b]]) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
// CHECK: return %[[c]] : tensor<1x112x112x32xf32>
}

// -----

// Make sure quantization parameters are not scanned if quantize op is presented.
// CHECK-LABEL: NoRedundantQuantizeWeight
func.func @NoRedundantQuantizeWeight() -> tensor<1x112x112x32xf32> {
  %w = arith.constant dense<1.0> : tensor<1x112x112x32xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %dq = "tfl.dequantize"(%q) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x112x112x32xf32>
  func.return %dq : tensor<1x112x112x32xf32>

// CHECK-NEXT: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<1x112x112x32xf32>
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>}>
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: return %[[dq]] : tensor<1x112x112x32xf32>
}

// -----

// CHECK-LABEL: ReturnQuantizedResult
func.func @ReturnQuantizedResult(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<32x3x3x3xf32>, %arg2: tensor<32xf32>) -> (tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>) {
  %0 = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> (tensor<1x112x112x32xf32>)
  func.return %0, %2 : tensor<1x112x112x32xf32>, tensor<1x112x112x32xf32>

// CHECK: %[[dw:.*]] = "tfl.depthwise_conv_2d"(%arg0, %arg1, %arg2)
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[dw]])
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: return %[[dq]], %[[dq]]
}

// -----

// QDQ-LABEL: BroadcastToPerTensorQuantizationPropagation
func.func @BroadcastToPerTensorQuantizationPropagation() -> tensor<2x5xf32> {
  %shape = arith.constant dense<[2, 5]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>} : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>) -> tensor<5xf32>
  %t = "tfl.broadcast_to"(%dq, %shape) : (tensor<5xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

  // QDQ: %[[shape:.*]] = arith.constant dense<[2, 5]> : tensor<2xi32>
  // QDQ-NEXT: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5xf32>
  // QDQ-NEXT: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>}> : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>
  // QDQ-NEXT: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>) -> tensor<5xf32>
  // QDQ-NEXT: %[[bt:.*]] = "tfl.broadcast_to"(%[[dqw]], %[[shape]]) : (tensor<5xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  // QDQ-NEXT: %[[qtw:.*]] = "tfl.quantize"(%[[bt]]) <{qtype = tensor<2x5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>}> {volatile} : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>
  // QDQ-NEXT: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>) -> tensor<2x5xf32>
  // QDQ-NEXT: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// QDQ-LABEL: TransposePerTensorQuantizationPropagation
func.func @TransposePerTensorQuantizationPropagation() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

  // QDQ: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
  // QDQ-NEXT: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
  // QDQ-NEXT: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32
  // QDQ-NEXT: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32
  // QDQ-NEXT: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  // QDQ-NEXT: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i8<-127:127>:f32
  // QDQ-NEXT: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32
  // QDQ-NEXT: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// QDQ-LABEL: TransposePerChannelNewQuantDim
func.func @TransposePerChannelNewQuantDim() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

// QDQ: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// QDQ-NEXT: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
// QDQ-NEXT: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32:0
// QDQ-NEXT: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32:0
// QDQ-NEXT: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
// QDQ-NEXT: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i8<-127:127>:f32:1
// QDQ-NEXT: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32:1
// QDQ-NEXT: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// QDQ-LABEL: ReshapePerChannelNewQuantDim
func.func @ReshapePerChannelNewQuantDim() -> tensor<24x5xf32> {
  %cst = arith.constant dense<1.0> : tensor<1x2x3x4x5xf32>
  %cst_1 = arith.constant dense<[24, 5]> : tensor<2xi32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>} : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>
  %dq = "tfl.dequantize"(%q) : (tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>) -> tensor<1x2x3x4x5xf32>
  %0 = "tfl.reshape"(%dq, %cst_1) : (tensor<1x2x3x4x5xf32>, tensor<2xi32>) -> tensor<24x5xf32>
  func.return %0 : tensor<24x5xf32>

// QDQ: %cst = arith.constant dense<1.000000e+00> : tensor<1x2x3x4x5xf32>
// QDQ-NEXT: %cst_0 = arith.constant dense<[24, 5]> : tensor<2xi32>
// QDQ-NEXT: %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>}> : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>
// QDQ-NEXT: %1 = "tfl.dequantize"(%0) : (tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>) -> tensor<1x2x3x4x5xf32>
// QDQ-NEXT: %2 = "tfl.reshape"(%1, %cst_0) : (tensor<1x2x3x4x5xf32>, tensor<2xi32>) -> tensor<24x5xf32>
// QDQ-NEXT: %3 = "tfl.quantize"(%2) <{qtype = tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>}> : (tensor<24x5xf32>) -> tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>
// QDQ-NEXT: %4 = "tfl.dequantize"(%3) : (tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>) -> tensor<24x5xf32>
// QDQ-NEXT: return %4 : tensor<24x5xf32>
}

// -----

// QDQ-LABEL: TransposePerChannelNewQuantDim_int4
func.func @TransposePerChannelNewQuantDim_int4() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

// QDQ: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// QDQ-NEXT: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
// QDQ-NEXT: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i4<-7:7>:f32:0
// QDQ-NEXT: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i4<-7:7>:f32:0
// QDQ-NEXT: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
// QDQ-NEXT: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i4<-7:7>:f32:1
// QDQ-NEXT: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i4<-7:7>:f32:1
// QDQ-NEXT: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: concat_requantize_inputs_and_outputs_if_different_scales
func.func @concat_requantize_inputs_and_outputs_if_different_scales(%arg0: tensor<2x1xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x4xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x1x!quant.uniform<u8:f32, 2.0>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<u8:f32, 2.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x!quant.uniform<u8:f32, 2.0>>) -> (tensor<2x1xf32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<2x3x!quant.uniform<u8:f32, 1.0>>} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x3x!quant.uniform<u8:f32, 1.0>>) -> (tensor<2x3xf32>)
  %4 = "tfl.concatenation"(%1, %3) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  func.return %4: tensor<2x4xf32>

// CHECK:      %0 = "tfl.quantize"(%arg0) <{qtype = tensor<2x1x!quant.uniform<u8:f32, 2.000000e+00>>}> : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<u8:f32, 2.000000e+00>>
// CHECK-NEXT: %1 = "tfl.dequantize"(%0)
// CHECK-NEXT: %2 = "tfl.quantize"(%arg1) <{qtype = tensor<2x3x!quant.uniform<u8:f32, 1.000000e+00>>}> : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK-NEXT: %3 = "tfl.dequantize"(%2)
// CHECK-NEXT: %4 = "tfl.concatenation"(%1, %3) <{axis = -1 : i32, fused_activation_function = "NONE"}> : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
// CHECK-NEXT: %5 = "tfl.quantize"(%4) <{qtype = tensor<2x4x!quant.uniform<u8:f32, 2.000000e+00>>}> {volatile} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 2.000000e+00>>
// CHECK-NEXT: %6 = "tfl.dequantize"(%5)
// CHECK-NEXT: return %6
}

// -----

