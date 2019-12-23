// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize -tfl-numeric-verify | FileCheck --check-prefix=DEBUG %s

// CHECK-LABEL: QuantizeFloatConst
func @QuantizeFloatConst() -> tensor<f32> {
  %0 = constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<f32>
  return %2 : tensor<f32>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<0> : tensor<2x2xi8>}
// CHECK:  %[[dq:.*]] = "tfl.dequantize"(%[[cst]])
// CHECK:  return %[[dq]] : tensor<f32>
}

// CHECK-LABEL: QuantizeDenseFloatConst
func @QuantizeDenseFloatConst() -> tensor<2x2xf32> {
  %0 = constant dense<[[-0.1, 1.0], [1.0, 3.0]]> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<{{\[\[}}0, -1], {{\[}}-1, -1]]> : tensor<2x2xi8>}
// CHECK:  %[[dq:.*]] = "tfl.dequantize"(%[[cst]])
// CHECK:  return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: QuantizeSplatFloatConst
func @QuantizeSplatFloatConst() -> tensor<2x2xf32> {
  %0 = constant dense<3.0> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}
// CHECK:  %[[dq:.*]] = "tfl.dequantize"(%[[cst]])
// CHECK:  return %[[dq]] : tensor<2x2xf32>
}

// CHECK-LABEL: DequantizeAndQuantize
func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}
// CHECK:  return %[[cst]] : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// CHECK-LABEL: QuantizeConv2D
// DEBUG-LABEL: QuantizeConv2D
func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %w = constant dense<-1.0> : tensor<32x3x3x3xf32>
  %3 = "tfl.quantize"(%w) {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 2 : i32, dilation_w_factor = 3 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi8>}
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// DEBUG: %[[wt:.*]] = constant dense<-1.000000e+00> : tensor<32x3x3x3xf32>
// DEBUG: %[[bias:.*]] = constant dense<-1.23697901> : tensor<32xf32>
// DEBUG: %[[act:.*]] = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
// DEBUG: %[[f_conv:.*]] = "tfl.conv_2d"(%[[act]], %[[wt]], %[[bias]])
// DEBUG: %[[q_conv:.*]] = "tfl.conv_2d"
// DEBUG: "tfl.NumericVerify"(%[[q_conv]], %[[f_conv]]) {tolerance = 1.000000e-01 : f32}
// DEBUG: return %[[q_conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
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

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[cst1]], %[[cst0]]) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}
// CHECK: return %[[conv]]
}

// CHECK-LABEL: QuantizeFullyConnected
func @QuantizeFullyConnected(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<-1.23697901> : tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>} : () -> tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x12xf32>
  %5 = "tfl.fully_connected"(%2, %4, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}
// CHECK: %[[cst_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>}
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[cst_1]], %[[cst_0]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}
// CHECK: return %[[fc]]
}

// CHECK-LABEL: QuantizeNoBiasFullyConnected
func @QuantizeNoBiasFullyConnected(%arg0: tensor<3x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>, %arg2: none) -> tensor<3x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<3x!quant.uniform<u8:f32, 1.0>>) -> tensor<3xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<3x3xf32>
  %2 = "tfl.fully_connected"(%0, %1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<3xf32>, tensor<3x3xf32>, none) -> tensor<3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<3x!quant.uniform<u8:f32, 1.0>>} : (tensor<3xf32>) -> tensor<3x!quant.uniform<u8:f32, 1.0>>
  return %3 : tensor<3x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2)
// CHECK: return %[[fc]]
}

// CHECK-LABEL: QuantizeAveragePool2D
func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  return %1 : tensor<1x1x1x16xf32>

// CHECK: %[[avgp:.*]] = "tfl.average_pool_2d"(%arg0)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[avgp]]) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %[[dq]] : tensor<1x1x1x16xf32>
}

// CHECK-LABEL: QuantizeReshape2D
func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = constant dense<[1, 36, 16]> : tensor<3xi32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  return %1 : tensor<1x36x16xf32>

// CHECK: %[[rs:.*]] = "tfl.reshape"(%arg0, %{{.*}})
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[rs]]) : (tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32>
// CHECK: return %[[dq]] : tensor<1x36x16xf32>
}

// CHECK-LABEL: QuantizeSoftmax
func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %1 : tensor<1x6x6x16xf32>

// CHECK: %[[sm:.*]] = "tfl.softmax"(%arg0)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[sm]]) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x6x6x16xf32>
// CHECK: return %[[dq]] : tensor<1x6x6x16xf32>
}

// CHECK-LABEL: QuantizeLogistic
func @QuantizeLogistic(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %1 : tensor<1x6x6x16xf32>

// CHECK: %[[lg:.*]] = "tfl.logistic"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[lg]]) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>)
// CHECK: return %[[dq]]
}

// CHECK-LABEL: QuantizeAdd
func @QuantizeAdd(tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>> {
^bb0(%arg0: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, %arg1: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>) -> tensor<1x56x56x24xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24xf32>
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<1x56x56x24xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>} : (tensor<1x56x56x24xf32>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
  return %3 : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>

// CHECK: %[[add:.*]] = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>)
// CHECK: return %[[add]] : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
}

// CHECK-LABEL: QuantizeConcat
func @QuantizeConcat(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %[[cc:.*]] = "tfl.concatenation"(%[[q1]], %[[q0]]) {axis = 0 : i32, fused_activation_function = "NONE"}
// CHECK: return %[[cc]] : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeConcatRequantize
func @QuantizeConcatRequantize(tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>, %arg1: tensor<1x2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<1x2x!quant.uniform<u8:f32, 2.000000e+00:128>>) -> tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: %[[cc:.*]] = "tfl.concatenation"(%[[q0]], %[[q1]]) {axis = 0 : i32, fused_activation_function = "NONE"}
// CHECK: return %[[cc]] : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// CHECK-LABEL: QuantizeMaxPool2D
func @QuantizeMaxPool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.max_pool_2d"(%0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  return %1 : tensor<1x1x1x16xf32>

// CHECK: %[[mp:.*]] = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[mp]]) : (tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32>
// CHECK: return %[[dq]] : tensor<1x1x1x16xf32>
}

// CHECK-LABEL: QuantizeSplit
// DEBUG-LABEL: QuantizeSplit
func @QuantizeSplit(%arg: tensor<4x!quant.uniform<u8:f32, 1.0>>, %cst: tensor<i32>) -> (tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %1:2 = "tfl.split"(%cst, %0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %2 = "tfl.quantize"(%1#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.quantize"(%1#1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  return %2, %3 : tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[sp:.*]]:2 = "tfl.split"(%arg1, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>)
// CHECK: return %[[sp]]#0, %[[sp]]#1

// DEUBG: %[[f_split:.*]]:2 = "tfl.split"
// DEUBG: %[[q_split:.*]]:2 = "tfl.split"
// DEUBG: "tfl.NumericVerify"(%[[q_split]]#1, %[[f_split]]#1) {tolerance = 1.000000e-01 : f32}
// DEUBG: "tfl.NumericVerify"(%[[q_split]]#0, %[[f_split]]#0) {tolerance = 1.000000e-01 : f32}
}

// CHECK-LABEL: QuantizeSplitUnusedResults
func @QuantizeSplitUnusedResults(%arg: tensor<4x!quant.uniform<u8:f32, 1.0>>, %cst: tensor<i32>)
  -> (tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %1:4 = "tfl.split"(%cst, %0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>,tensor<2xf32>, tensor<2xf32>)
  %2 = "tfl.quantize"(%1#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.quantize"(%1#1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  return %2, %3 : tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[sp:.*]]:4 = "tfl.split"(%arg1, %arg0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>)
// CHECK: return %[[sp]]#0, %[[sp]]#1
}

// CHECK-LABEL: QuantizeShape
func @QuantizeShape(%arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>,
                    %arg2: tensor<*x!quant.uniform<u8:f32, 1.0>>,
                    %arg3: tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1xi32>,tensor<?xi32>,tensor<3xi32>) {
  %1 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<*x!quant.uniform<u8:f32, 1.0>>) -> tensor<*xf32>
  %3 = "tfl.dequantize"(%arg3) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> tensor<?x?x4xf32>
  %4 = "tfl.shape"(%1) : (tensor<4xf32>) -> tensor<1xi32>
  %5 = "tfl.shape"(%2) : (tensor<*xf32>) -> tensor<?xi32>
  %6 = "tfl.shape"(%3) : (tensor<?x?x4xf32>) -> tensor<3xi32>
  return %4, %5, %6 : tensor<1xi32>, tensor<?xi32>, tensor<3xi32>

// CHECK: %[[s1:.*]] = "tfl.shape"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<1xi32>
// CHECK-NEXT: %[[s2:.*]] = "tfl.shape"(%arg1) : (tensor<*x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<?xi32>
// CHECK-NEXT: %[[s3:.*]] = "tfl.shape"(%arg2) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<3xi32>
// CHECK-NEXT: %[[s1]], %[[s2]], %[[s3]] : tensor<1xi32>, tensor<?xi32>, tensor<3xi32>
}

// CHECK-LABEL: QuantizeMultipleUsers
func @QuantizeMultipleUsers(%arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1xi32>,tensor<1xi32>) {
  %1 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2 = "tfl.shape"(%1) : (tensor<4xf32>) -> tensor<1xi32>
  return %2, %2 : tensor<1xi32>, tensor<1xi32>

// CHECK: %[[s1:.*]] = "tfl.shape"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<1xi32>
// CHECK-NEXT: %[[s1]], %[[s1]] : tensor<1xi32>, tensor<1xi32>
}

// CHECK-LABEL: NotQuantizePow
// DEBUG-LABEL: NotQuantizePow
func @NotQuantizePow(%arg0: tensor<4x!quant.uniform<u8:f32, 1.0>>,
                     %arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<4x!quant.uniform<u8:f32, 1.0>>) {
  %1 = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %3 = "tfl.pow"(%1, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>

  return %4 : tensor<4x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%arg0)
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%arg1)
// CHECK-NEXT: %[[pow:.*]] = tfl.pow %[[dq1]], %[[dq2]]
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[pow]])
// CHECK-NEXT: return %[[q]]

// DEBUG-NOT: "tfl.NumericVerify"
}
