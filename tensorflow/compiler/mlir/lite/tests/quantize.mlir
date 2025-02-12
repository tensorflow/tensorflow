// RUN: tf-opt %s -split-input-file -tfl-prepare-quantize -tfl-quantize  | FileCheck %s
// RUN: tf-opt %s -split-input-file -tfl-quantize="legacy-quantize=true" | FileCheck --check-prefix=LEGACY %s
// RUN: tf-opt %s -split-input-file -tfl-prepare-quantize -tfl-quantize="ops-blocklist=tfl.fully_connected,tfl.softmax locs-blocklist=Block,NullBlock" | FileCheck --check-prefix=BLOCK %s

// CHECK-LABEL: QuantizeFloatConst
func.func @QuantizeFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<0> : tensor<2x2xi8>}>
// CHECK:  return %[[cst]]
}

// -----

// CHECK-LABEL: QuantizeFloatConst4Bits
func.func @QuantizeFloatConst4Bits() -> tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>> {
  %0 = arith.constant dense<[[-0.75, -0.5, -0.25, 0.0], [0.25, 0.5, 0.75, 1.0]]> : tensor<2x4xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>
  func.return %1 : tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>, value = dense<{{\[\[}}-4, -3, -2, -1{{\]}}, [0, 1, 2, 3{{\]\]}}> : tensor<2x4xi4>}>
// CHECK:  return %[[cst]]
}

// -----

// CHECK-LABEL: QuantizeDenseFloatConst
func.func @QuantizeDenseFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<[[-0.1, 1.0], [1.0, 3.0]]> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<{{\[\[}}0, -1], {{\[}}-1, -1]]> : tensor<2x2xi8>}>
// CHECK:  return %[[cst]]
}

// -----

// CHECK-LABEL: QuantizeSplatFloatConst
func.func @QuantizeSplatFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<3.0> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}>
// CHECK:  return %[[cst]]
}

// -----

// CHECK-LABEL: NotQuantizeFloatConst
func.func @NotQuantizeFloatConst() -> tensor<2x2xf32> {
  %0 = arith.constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  func.return %2 : tensor<2x2xf32>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_const"(){{.*}}dense<-1.000000e-01> : tensor<2x2xf32>
// CHECK:  return %[[cst]] : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: DequantizeAndQuantize
func.func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}>
// CHECK:  return %[[cst]] : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// -----

// CHECK-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<-1.0> : tensor<32x3x3x3xf32>
  %3 = "tfl.quantize"(%w) {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi8>}>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeConv2D4Bit
func.func @QuantizeConv2D4Bit(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<-1.0> : tensor<32x3x3x3xf32>
  %3 = "tfl.quantize"(%w) {qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi4>}>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
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

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}>
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[cst1]], %[[cst0]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}>
// CHECK: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeDepthwiseConv2D4Bit
func.func @QuantizeDepthwiseConv2D4Bit(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>, value = dense<-7> : tensor<32x3x3x3xi4>} : () -> tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.depthwise_conv_2d"(%2, %4, %cst) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>, value = dense<-400> : tensor<32xi32>}>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.39599830800000002:8>>, value = dense<-7> : tensor<32x3x3x3xi4>}>
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[cst1]], %[[cst0]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}>
// CHECK: return %[[conv]]
}

// -----

// CHECK-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>} : () -> tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x12xf32>
  %5 = "tfl.fully_connected"(%2, %4, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst_0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}>
// CHECK: %[[cst_1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>}>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[cst_1]], %[[cst_0]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK: return %[[fc]]

// BLOCK: %[[cst:.*]] = "tfl.pseudo_const"(){{.*}}dense<-1.23697901>
// BLOCK: %[[dq1:.*]] = "tfl.dequantize"(%arg0)
// BLOCK: %[[cst2:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>}>
// BLOCK: %[[dq2:.*]] = "tfl.dequantize"(%[[cst2]])
// BLOCK: %[[fc:.*]] = "tfl.fully_connected"(%[[dq1]], %[[dq2]], %[[cst]])
// BLOCK: %[[q:.*]] = "tfl.quantize"(%[[fc]])
// BLOCK: return %[[q]]
}

// -----

// CHECK-LABEL: QuantizeFullyConnected4Bit
func.func @QuantizeFullyConnected4Bit(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>, value = dense<-7> : tensor<32x12xi4>} : () -> tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>) -> tensor<32x12xf32>
  %5 = "tfl.fully_connected"(%2, %4, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst_0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>, value = dense<-400> : tensor<32xi32>}>
// CHECK: %[[cst_1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.39599830800000002:8>>, value = dense<-7> : tensor<32x12xi4>}>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[cst_1]], %[[cst_0]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
// CHECK: return %[[fc]]

// BLOCK: %[[cst:.*]] = "tfl.pseudo_const"(){{.*}}dense<-1.23697901>
// BLOCK: %[[dq1:.*]] = "tfl.dequantize"(%arg0)
// BLOCK: %[[cst2:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.39599830800000002:8>>, value = dense<-7> : tensor<32x12xi4>}>
// BLOCK: %[[dq2:.*]] = "tfl.dequantize"(%[[cst2]])
// BLOCK: %[[fc:.*]] = "tfl.fully_connected"(%[[dq1]], %[[dq2]], %[[cst]])
// BLOCK: %[[q:.*]] = "tfl.quantize"(%[[fc]])
// BLOCK: return %[[q]]
}

// -----

// CHECK-LABEL: QuantizeNoBiasFullyConnected
func.func @QuantizeNoBiasFullyConnected(%arg0: tensor<3x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>, %arg2: none) -> tensor<3x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<3x!quant.uniform<u8:f32, 1.0>>) -> tensor<3xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<3x3xf32>
  %2 = "tfl.fully_connected"(%0, %1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<3xf32>, tensor<3x3xf32>, none) -> tensor<3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<3x!quant.uniform<u8:f32, 1.0>>} : (tensor<3xf32>) -> tensor<3x!quant.uniform<u8:f32, 1.0>>
  func.return %3 : tensor<3x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2)
// CHECK: return %[[fc]]
}

// -----

// CHECK-LABEL: QuantizeAveragePool2D
func.func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %1 : tensor<1x1x1x16xf32>

// CHECK: %[[avgp:.*]] = "tfl.average_pool_2d"(%arg0)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[avgp]]) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %[[dq]] : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: QuantizeReshape2D
func.func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  func.return %1 : tensor<1x36x16xf32>

// CHECK: %[[rs:.*]] = "tfl.reshape"(%arg0, %{{.*}})
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[rs]]) : (tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32>
// CHECK: return %[[dq]] : tensor<1x36x16xf32>
}

// -----

// CHECK-LABEL: QuantizeSoftmax
func.func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// CHECK: %[[sm:.*]] = "tfl.softmax"(%arg0)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[sm]]) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x6x6x16xf32>
// CHECK: return %[[dq]] : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeLogistic
func.func @QuantizeLogistic(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %1 : tensor<1x6x6x16xf32>

// CHECK: %[[lg:.*]] = "tfl.logistic"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[lg]]) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>)
// CHECK: return %[[dq]]
}

// -----

// CHECK-LABEL: QuantizeAdd
func.func @QuantizeAdd(tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>> {
^bb0(%arg0: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, %arg1: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>) -> tensor<1x56x56x24xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24xf32>
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<1x56x56x24xf32> loc("Block")
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>} : (tensor<1x56x56x24xf32>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
  func.return %3 : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>

// CHECK: %[[add:.*]] = tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>)
// CHECK: return %[[add]] : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>

// BLOCK: %[[dq0:.*]] = "tfl.dequantize"(%arg0) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>)
// BLOCK: %[[dq1:.*]] = "tfl.dequantize"(%arg1) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>)
// BLOCK: %[[add:.*]] = tfl.add %[[dq0]], %[[dq1]] {fused_activation_function = "NONE"} : tensor<1x56x56x24xf32>
// BLOCK: %[[q:.*]] = "tfl.quantize"(%[[add]]) <{qtype = tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>}>
// BLOCK: return %[[q]] : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
}

// -----

// CHECK-LABEL: QuantizeConcat
func.func @QuantizeConcat(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %[[cc:.*]] = "tfl.concatenation"(%[[q1]], %[[q0]]) <{axis = 0 : i32, fused_activation_function = "NONE"}>
// CHECK: return %[[cc]] : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatRequantize
func.func @QuantizeConcatRequantize(tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>, %arg1: tensor<1x2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {volatile}
// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}>
// CHECK: %[[cc:.*]] = "tfl.concatenation"(%[[q0]], %[[q1]]) <{axis = 0 : i32, fused_activation_function = "NONE"}>
// CHECK: return %[[cc]] : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeMaxPool2D
func.func @QuantizeMaxPool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.max_pool_2d"(%0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %1 : tensor<1x1x1x16xf32>

// CHECK: %[[mp:.*]] = "tfl.max_pool_2d"(%arg0) <{filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[mp]]) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %[[dq]] : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: QuantizeSplit
func.func @QuantizeSplit(%arg: tensor<4x!quant.uniform<u8:f32, 1.0>>, %cst: tensor<i32>) -> (tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %1:2 = "tfl.split"(%cst, %0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %2 = "tfl.quantize"(%1#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.quantize"(%1#1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  func.return %2, %3 : tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[sp:.*]]:2 = "tfl.split"(%arg1, %arg0) <{num_splits = 2 : i32}> : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>)
// CHECK: return %[[sp]]#0, %[[sp]]#1
}

// -----

// CHECK-LABEL: QuantizeSplitUnusedResults
func.func @QuantizeSplitUnusedResults(%arg: tensor<4x!quant.uniform<u8:f32, 1.0>>, %cst: tensor<i32>)
  -> (tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %1:4 = "tfl.split"(%cst, %0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>,tensor<2xf32>, tensor<2xf32>)
  %2 = "tfl.quantize"(%1#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.quantize"(%1#1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  func.return %2, %3 : tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[sp:.*]]:4 = "tfl.split"(%arg1, %arg0) <{num_splits = 4 : i32}> : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>)
// CHECK: return %[[sp]]#0, %[[sp]]#1
}

// -----

// CHECK-LABEL: QuantizeShape
func.func @QuantizeShape(%arg0: tensor<*x!quant.uniform<u8:f32, 1.0>>,
                    %arg1: tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<?xi32>,tensor<3xi32>) {
  %2 = "tfl.dequantize"(%arg0) : (tensor<*x!quant.uniform<u8:f32, 1.0>>) -> tensor<*xf32>
  %3 = "tfl.dequantize"(%arg1) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> tensor<?x?x4xf32>
  %5 = "tfl.shape"(%2) : (tensor<*xf32>) -> tensor<?xi32>
  %6 = "tfl.shape"(%3) : (tensor<?x?x4xf32>) -> tensor<3xi32>
  func.return %5, %6 : tensor<?xi32>, tensor<3xi32>

// CHECK: %[[s2:.*]] = "tfl.shape"(%arg0) : (tensor<*x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<?xi32>
// CHECK-NEXT: %[[s3:.*]] = "tfl.shape"(%arg1) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<3xi32>
// CHECK-NEXT: %[[s2]], %[[s3]] : tensor<?xi32>, tensor<3xi32>
}

// -----

// CHECK-LABEL: QuantizeMultipleUsers
func.func @QuantizeMultipleUsers(%arg1: tensor<?x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1xi32>,tensor<1xi32>) {
  %1 = "tfl.dequantize"(%arg1) : (tensor<?x!quant.uniform<u8:f32, 1.0>>) -> tensor<?xf32>
  %2 = "tfl.shape"(%1) : (tensor<?xf32>) -> tensor<1xi32>
  func.return %2, %2 : tensor<1xi32>, tensor<1xi32>

// CHECK: %[[s1:.*]] = "tfl.shape"(%arg0) : (tensor<?x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<1xi32>
// CHECK-NEXT: %[[s1]], %[[s1]] : tensor<1xi32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: NotQuantizePow
func.func @NotQuantizePow(%arg0: tensor<4x!quant.uniform<u8:f32, 1.0>>,
                     %arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<4x!quant.uniform<u8:f32, 1.0>>) {
  %1 = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %3 = "tfl.pow"(%1, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>

  func.return %4 : tensor<4x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[pow:.*]] = tfl.pow %[[dq1]], %[[dq2]]
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[pow]])
// CHECK-NEXT: return %[[q]]

}

// -----

// CHECK-LABEL: QuantizeCustomTfOp
func.func @QuantizeCustomTfOp(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
    %arg1: tensor<1x!quant.uniform<u8:f32, 0.2:127>>, %arg2: tensor<1x!quant.uniform<u8:f32, 0.4:127>>,
    %arg3: tensor<1xi32>) -> (tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>) -> tensor<128x128xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x!quant.uniform<u8:f32, 0.2:127>>) -> tensor<1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<1x!quant.uniform<u8:f32, 0.4:127>>) -> tensor<1xf32>
  %3 = "tfl.custom_tf"(%0, %1, %2, %arg3) ({
  ^bb0(%a1: tensor<128x128xf32>, %a2: tensor<1xf32>, %a3: tensor<1xf32>, %a4: tensor<1xi32>):
    %4 = "tf.LayerNorm"(%a1, %a2, %a3, %a4) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
   "tfl.yield"(%4) : (tensor<128x128xf32>) -> ()
  }) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>} : (tensor<128x128xf32>) -> tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
  func.return %4 : tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>

// CHECK: %4 = "tfl.custom_tf"(%arg0, %arg1, %arg2, %arg3) ({
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {_tfl_quant_trait = "fully_quantizable", device = ""} :
// CHECK-SAME: (tensor<128x128x!quant.uniform<u8:f32, 1.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 2.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 4.000000e-01:127>>, tensor<1xi32>)
// CHECK-SAME: -> tensor<128x128x!quant.uniform<u8:f32, 2.000000e-01:125>>
}

// -----

// CHECK-LABEL: NotQuantizeCustomTfOp
func.func @NotQuantizeCustomTfOp(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
    %arg1: tensor<1x!quant.uniform<u8:f32, 0.2:127>>, %arg2: tensor<1x!quant.uniform<u8:f32, 0.4:127>>,
    %arg3: tensor<1xi32>) -> (tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>) -> tensor<128x128xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x!quant.uniform<u8:f32, 0.2:127>>) -> tensor<1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<1x!quant.uniform<u8:f32, 0.4:127>>) -> tensor<1xf32>
  %3 = "tfl.custom_tf"(%0, %1, %2, %arg3) ({
  ^bb0(%a1: tensor<128x128xf32>, %a2: tensor<1xf32>, %a3: tensor<1xf32>, %a4: tensor<1xi32>):
    %4 = "tf.LayerNorm"(%a1, %a2, %a3, %a4) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
   "tfl.yield"(%4) : (tensor<128x128xf32>) -> ()
  }) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>} : (tensor<128x128xf32>) -> tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
  func.return %4 : tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>

// CHECK: "tfl.custom_tf"
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
}

// -----

// CHECK-LABEL: NotQuantizableValues
func.func @NotQuantizableValues(%arg0: tensor<1x!tf_type.string>) -> (tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x!tf_type.string>, tensor<1xi32>) {
  %0:3 = "tfl.custom_tf"(%arg0) ({
  ^bb0(%arg1: tensor<1x!tf_type.string>):
    %1:3 = "tf.SequenceStringProjection"(%arg1) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf_type.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf_type.string>, tensor<1xi32>)
    "tfl.yield"(%1#0, %1#1, %1#2) : (tensor<1x?x16xf32>, tensor<1x!tf_type.string>, tensor<1xi32>) -> ()
  }) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf_type.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf_type.string>, tensor<1xi32>)
  %2 = "tfl.quantize"(%0#0) {qtype = tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>} : (tensor<1x?x16xf32>) -> tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>
  func.return %2, %0#1, %0#2 : tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x!tf_type.string>, tensor<1xi32>

// CHECK: "tfl.custom_tf"(%arg0) ({
// CHECK-NEXT: ^bb0(%arg1: tensor<1x!tf_type.string>):
// CHECK-NEXT:   "tf.SequenceStringProjection"(%arg1) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf_type.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf_type.string>, tensor<1xi32>)
// CHECK-NEXT:   "tfl.yield"
// CHECK: }) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf_type.string>) -> (tensor<1x?x16x!quant.uniform<u8:f32, 1.000000e-01:128>>, tensor<1x!tf_type.string>, tensor<1xi32>)
}

// -----

// Checks that legacy path correctly handles asymmetric quantized values.
// LEGACY-LABEL: CheckLegacyQuantizeAdd
func.func @CheckLegacyQuantizeAdd() -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>> {
  %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
  %0 = "tfl.quantize"(%cst) {qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>

// LEGACY:  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, value = dense<{{\[\[}}-1, 127]]> : tensor<1x2xi8>}>
}

// -----

func.func private @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: NotQuantizeIf
func.func @NotQuantizeIf(%arg0: tensor<i1>,
                    %arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<4x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %1 = "tf.If"(%arg0, %0) {then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false} : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>

  func.return %2 : tensor<4x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[dq:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[if:.*]] = "tf.If"(%arg0, %[[dq]]
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[if]])
// CHECK-NEXT: return %[[q]]
}

// -----

// CHECK-LABEL: NotQuantizeReadVariable
func.func @NotQuantizeReadVariable() -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>> {
  %0 = "tfl.var_handle"() {container = "", shared_name = "states"} : () -> tensor<!tf_type.resource<tensor<1x2x3xf32>>>
  %1 = "tfl.read_variable"(%0) : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>} : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
  func.return %2 : tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
  // CHECK: %[[handle:.*]] = "tfl.var_handle"() <{container = "", shared_name = "states"}> : () -> tensor<!tf_type.resource<tensor<1x2x3xf32>>>
  // CHECK-NEXT: %[[read:.*]] = "tfl.read_variable"(%[[handle]]) : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32>
  // CHECK-NEXT: %[[quantize:.*]] = "tfl.quantize"(%[[read]]) <{qtype = tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>}> : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
  // CHECK-NEXT: return %[[quantize]]
}

// -----

// CHECK-LABEL: foldQuantWeightsIntoTposeConv
func.func @foldQuantWeightsIntoTposeConv(%arg0: tensor<2x2x3x2048xf32>) -> tensor<2x3x2x2048xf32> {
  %output_shape = arith.constant dense<[2, 3, 2, 2048]> : tensor<4xi32>
  %q_weighs = "tfl.pseudo_qconst"() {qtype = tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>, value = dense<-76> : tensor<4x2x2x2048xi8>} : () -> tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>
  %dq_weights = "tfl.dequantize"(%q_weighs) : (tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>) -> tensor<4x2x2x2048xf32>
  %bias = "tfl.no_value"() {value} : () -> none
  %out = "tfl.transpose_conv"(%output_shape, %dq_weights, %arg0, %bias) {fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<4xi32>, tensor<4x2x2x2048xf32>, tensor<2x2x3x2048xf32>, none) -> tensor<2x3x2x2048xf32>
  func.return %out : tensor<2x3x2x2048xf32>

  // CHECK-NOT: "tfl.dequantize"
  // CHECK: "tfl.transpose_conv"(%cst, %1, %arg0, %0) <{fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<4xi32>, tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32
}

// -----

// CHECK-LABEL: foldQuantWeightsIntoTposeConvf16NotFolded
func.func @foldQuantWeightsIntoTposeConvf16NotFolded(%arg0: tensor<2x2x3x2048xf32>) -> tensor<2x3x2x2048xf32> {
  %output_shape = arith.constant dense<[2, 3, 2, 2048]> : tensor<4xi32>
  %f16_weights = "tfl.pseudo_const"() {value = dense<1.0> : tensor<4x2x2x2048xf16>} : () -> tensor<4x2x2x2048xf16>
  %dq_weights = "tfl.dequantize"(%f16_weights) : (tensor<4x2x2x2048xf16>) -> tensor<4x2x2x2048xf32>
  %bias = "tfl.no_value"() {value} : () -> none
  %out = "tfl.transpose_conv"(%output_shape, %dq_weights, %arg0, %bias) {fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<4xi32>, tensor<4x2x2x2048xf32>, tensor<2x2x3x2048xf32>, none) -> tensor<2x3x2x2048xf32>
  func.return %out : tensor<2x3x2x2048xf32>

  // CHECK: "tfl.dequantize"
}

// -----

// CHECK-LABEL: foldQuantWeightsIntoEmbeddingLookup
func.func @foldQuantWeightsIntoEmbeddingLookup(%arg0: tensor<3xi32>) -> tensor<3x512xf32> {
  %q_weighs = "tfl.pseudo_qconst"() {qtype = tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>, value = dense<-76> : tensor<3074x512xi8>} : () -> tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>
  %dq_weights = "tfl.dequantize"(%q_weighs) : (tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>) -> tensor<3074x512xf32>
  %out = "tfl.embedding_lookup"(%arg0, %dq_weights) : (tensor<3xi32>, tensor<3074x512xf32>) -> tensor<3x512xf32>
  func.return %out : tensor<3x512xf32>

  // CHECK-NOT: "tfl.dequantize"
  // CHECK: "tfl.embedding_lookup"(%arg0, %0) : (tensor<3xi32>, tensor<3074x512x!quant.uniform<u8<1:255>:f32
}

// -----

// CHECK-LABEL: quantizeTFCustomOp
func.func @quantizeTFCustomOp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<!quant.uniform<i16:f32, 1.0>>} : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<!quant.uniform<i16:f32, 1.0>>) -> (tensor<f32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<!quant.uniform<i16:f32, 1.0>>} : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<!quant.uniform<i16:f32, 1.0>>) -> (tensor<f32>)
  %4 = "tfl.quantize"(%arg2) {qtype = tensor<!quant.uniform<i16:f32, 1.0>>} : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.0>>
  %5 = "tfl.dequantize"(%4) : (tensor<!quant.uniform<i16:f32, 1.0>>) -> (tensor<f32>)
  %6:4 = "tfl.custom_tf"(%1, %3, %5) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
      %7:4 = "tf.TFLite_Detection_PostProcess"(%arg3, %arg4, %arg5) {_output_quantized = true, _output_types = [f32, f32, f32, f32], _support_output_type_float_in_quantized_op = true} : (tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
      "tfl.yield"(%7#0, %7#1, %7#2, %7#3) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
    }) {_output_quantized = true, _output_types = [f32, f32, f32, f32], _support_output_type_float_in_quantized_op = true} : (tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
  return %6#0, %6#1, %6#2, %6#3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>

  // CHECK: %0 = "tfl.quantize"(%arg0) <{qtype = tensor<!quant.uniform<i16:f32, 1.000000e+00>>}> : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.000000e+00>>
  // CHECK: %1 = "tfl.quantize"(%arg1) <{qtype = tensor<!quant.uniform<i16:f32, 1.000000e+00>>}> : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.000000e+00>>
  // CHECK: %2 = "tfl.quantize"(%arg2) <{qtype = tensor<!quant.uniform<i16:f32, 1.000000e+00>>}> : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.000000e+00>>
  // CHECK: %3:4 = "tfl.custom_tf"(%0, %1, %2)
  // CHECK: (tensor<!quant.uniform<i16:f32, 1.000000e+00>>, tensor<!quant.uniform<i16:f32, 1.000000e+00>>, tensor<!quant.uniform<i16:f32, 1.000000e+00>>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  // CHECK: return %3#0, %3#1, %3#2, %3#3 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}
