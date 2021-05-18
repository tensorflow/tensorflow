// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize -tfl-quantize -tfl-numeric-verify -tfl-log-if-failed | FileCheck --check-prefix=DEBUG %s
// RUN: tf-opt %s -tfl-quantize -tfl-legacy-quantize | FileCheck --check-prefix=LEGACY %s

// CHECK-LABEL: QuantizeFloatConst
func @QuantizeFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<0> : tensor<2x2xi8>}
// CHECK:  return %[[cst]]
}

// CHECK-LABEL: QuantizeDenseFloatConst
func @QuantizeDenseFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = constant dense<[[-0.1, 1.0], [1.0, 3.0]]> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<{{\[\[}}0, -1], {{\[}}-1, -1]]> : tensor<2x2xi8>}
// CHECK:  return %[[cst]]
}

// CHECK-LABEL: QuantizeSplatFloatConst
func @QuantizeSplatFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = constant dense<3.0> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}
// CHECK:  return %[[cst]]
}

// CHECK-LABEL: NotQuantizeFloatConst
func @NotQuantizeFloatConst() -> tensor<2x2xf32> {
  %0 = constant dense<-0.1> : tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %2 = "tfl.dequantize"(%1) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>

// CHECK:  %[[cst:.*]] = constant dense<-1.000000e-01> : tensor<2x2xf32>
// CHECK:  return %[[cst]] : tensor<2x2xf32>
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
  %5 = "tfl.conv_2d"(%2, %4, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi8>}
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>

// DEBUG-DAG: %[[wt:.*]] = constant dense<-1.000000e+00> : tensor<32x3x3x3xf32>
// DEBUG-DAG: %[[bias:.*]] = constant dense<-1.23697901> : tensor<32xf32>
// DEBUG: %[[act:.*]] = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
// DEBUG: %[[f_conv:.*]] = "tfl.conv_2d"(%[[act]], %[[wt]], %[[bias]])
// DEBUG: %[[q_conv:.*]] = "tfl.conv_2d"
// DEBUG: "tfl.NumericVerify"(%[[q_conv]], %[[f_conv]]) {log_if_failed = true, tolerance = 5.000000e+00 : f32}
// DEBUG: return %[[q_conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// CHECK-LABEL: QuantizeDepthwiseConv2D
// DEBUG-LABEL: QuantizeDepthwiseConv2D
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
// DEBUG-LABEL: QuantizeFullyConnected
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

// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>, volatile}
// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>, volatile}
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

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>, volatile}
// CHECK: %[[q0:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}
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
// DEUBG: "tfl.NumericVerify"(%[[q_split]]#1, %[[f_split]]#1) {tolerance = 5.000000e+00 : f32}
// DEUBG: "tfl.NumericVerify"(%[[q_split]]#0, %[[f_split]]#0) {tolerance = 5.000000e+00 : f32}
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
func @QuantizeShape(%arg0: tensor<*x!quant.uniform<u8:f32, 1.0>>,
                    %arg1: tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<?xi32>,tensor<3xi32>) {
  %2 = "tfl.dequantize"(%arg0) : (tensor<*x!quant.uniform<u8:f32, 1.0>>) -> tensor<*xf32>
  %3 = "tfl.dequantize"(%arg1) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.0>>) -> tensor<?x?x4xf32>
  %5 = "tfl.shape"(%2) : (tensor<*xf32>) -> tensor<?xi32>
  %6 = "tfl.shape"(%3) : (tensor<?x?x4xf32>) -> tensor<3xi32>
  return %5, %6 : tensor<?xi32>, tensor<3xi32>

// CHECK: %[[s2:.*]] = "tfl.shape"(%arg0) : (tensor<*x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<?xi32>
// CHECK-NEXT: %[[s3:.*]] = "tfl.shape"(%arg1) : (tensor<?x?x4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<3xi32>
// CHECK-NEXT: %[[s2]], %[[s3]] : tensor<?xi32>, tensor<3xi32>
}

// CHECK-LABEL: QuantizeMultipleUsers
func @QuantizeMultipleUsers(%arg1: tensor<?x!quant.uniform<u8:f32, 1.0>>) -> (tensor<1xi32>,tensor<1xi32>) {
  %1 = "tfl.dequantize"(%arg1) : (tensor<?x!quant.uniform<u8:f32, 1.0>>) -> tensor<?xf32>
  %2 = "tfl.shape"(%1) : (tensor<?xf32>) -> tensor<1xi32>
  return %2, %2 : tensor<1xi32>, tensor<1xi32>

// CHECK: %[[s1:.*]] = "tfl.shape"(%arg0) : (tensor<?x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<1xi32>
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

// CHECK-LABEL: QuantizeCustomTfOp
// DEBUG-LABEL: QuantizeCustomTfOp
func @QuantizeCustomTfOp(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
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

// CHECK: %4 = "tfl.custom_tf"(%arg0, %arg1, %arg2, %arg3) ( {
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):  // no predecessors
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {_tfl_quant_trait = "fully_quantizable", device = ""} :
// CHECK-SAME: (tensor<128x128x!quant.uniform<u8:f32, 1.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 2.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 4.000000e-01:127>>, tensor<1xi32>)
// CHECK-SAME: -> tensor<128x128x!quant.uniform<u8:f32, 2.000000e-01:125>>
}


// CHECK-LABEL: NotQuantizeCustomTfOp
// DEBUG-LABEL: NotQuantizeCustomTfOp
func @NotQuantizeCustomTfOp(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
    %arg1: tensor<1x!quant.uniform<u8:f32, 0.2:127>>, %arg2: tensor<1x!quant.uniform<u8:f32, 0.4:127>>,
    %arg3: tensor<1xi32>) -> (tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>) -> tensor<128x128xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x!quant.uniform<u8:f32, 0.2:127>>) -> tensor<1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<1x!quant.uniform<u8:f32, 0.4:127>>) -> tensor<1xf32>
  %3 = "tfl.custom_tf"(%0, %1, %2, %arg3) ( {
  ^bb0(%a1: tensor<128x128xf32>, %a2: tensor<1xf32>, %a3: tensor<1xf32>, %a4: tensor<1xi32>):  // no predecessors
    %4 = "tf.LayerNorm"(%a1, %a2, %a3, %a4) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
   "tfl.yield"(%4) : (tensor<128x128xf32>) -> ()
  }) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>} : (tensor<128x128xf32>) -> tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
  return %4 : tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>

// CHECK: "tfl.custom_tf"
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):  // no predecessors
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
}

// CHECK-LABEL: NotQuantizableValues
func @NotQuantizableValues(%arg0: tensor<1x!tf.string>) -> (tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x!tf.string>, tensor<1xi32>) {
  %0:3 = "tfl.custom_tf"(%arg0) ( {
  ^bb0(%arg1: tensor<1x!tf.string>):  // no predecessors
    %1:3 = "tf.SequenceStringProjection"(%arg1) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf.string>, tensor<1xi32>)
    "tfl.yield"(%1#0, %1#1, %1#2) : (tensor<1x?x16xf32>, tensor<1x!tf.string>, tensor<1xi32>) -> ()
  }) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf.string>, tensor<1xi32>)
  %2 = "tfl.quantize"(%0#0) {qtype = tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>} : (tensor<1x?x16xf32>) -> tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>
  return %2, %0#1, %0#2 : tensor<1x?x16x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x!tf.string>, tensor<1xi32>

// CHECK: "tfl.custom_tf"(%arg0) ( {
// CHECK-NEXT: ^bb0(%arg1: tensor<1x!tf.string>):  // no predecessors
// CHECK-NEXT:   "tf.SequenceStringProjection"(%arg1) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf.string>) -> (tensor<1x?x16xf32>, tensor<1x!tf.string>, tensor<1xi32>)
// CHECK-NEXT:   "tfl.yield"
// CHECK: }) {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x!tf.string>) -> (tensor<1x?x16x!quant.uniform<u8:f32, 1.000000e-01:128>>, tensor<1x!tf.string>, tensor<1xi32>)
}

// Checks that legacy path correctly handles asymmetric quantized values.
// LEGACY-LABEL: CheckLegacyQuantizeAdd
func @CheckLegacyQuantizeAdd() -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>> {
  %cst = constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
  %0 = "tfl.quantize"(%cst) {qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>
  return %0 : tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>

// LEGACY:  "tfl.pseudo_qconst"() {qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, value = dense<{{\[\[}}-1, 127]]> : tensor<1x2xi8>}
}

// DEBUG-LABEL: CheckNumericVerifyMultipleUsers
func @CheckNumericVerifyMultipleUsers(%arg0: tensor<1x5x5x3xf32>) -> tensor<1x5x5x3xf32> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>, volatile} : (tensor<1x5x5x3xf32>) -> tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x5x5x3xf32>
  %2 = "tfl.average_pool_2d"(%1) {filter_height = 5 : i32, filter_width = 5 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 5 : i32, stride_w = 5 : i32} : (tensor<1x5x5x3xf32>) -> tensor<1x1x1x3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x1x1x3x!quant.uniform<i8:f32, 0.1>>, volatile} : (tensor<1x1x1x3xf32>) -> tensor<1x1x1x3x!quant.uniform<i8:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<1x1x1x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x1x1x3xf32>
  %5 = "tfl.add"(%1, %4) {fused_activation_function = "NONE"} : (tensor<1x5x5x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x5x5x3xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>, volatile} : (tensor<1x5x5x3xf32>) -> tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>
  %7 = "tfl.dequantize"(%6) : (tensor<1x5x5x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<1x5x5x3xf32>
  return %7 : tensor<1x5x5x3xf32>
// DEBUG: %[[q:.*]] = "tfl.quantize"(%arg0)
// DEBUG: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// DEBUG: "tfl.average_pool_2d"(%[[dq]])
// DEBUG: "tfl.average_pool_2d"(%[[q]])
}
