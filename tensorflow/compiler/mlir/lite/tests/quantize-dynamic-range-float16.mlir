// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-float16-quantization" -tfl-quantize="enable-dynamic-range-quantization=true" | FileCheck --check-prefix=CHECK %s

// CHECK-LABEL: QuantizeUnidirectionalLstm
func.func @QuantizeUnidirectionalLstm(%arg0: tensor<1x2x3xf32>) -> (tensor<1x2x3xf32>) {
  %1 = "tfl.pseudo_const"() {value = dense<[[0.1]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.2]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[[0.3]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %4 = "tfl.pseudo_const"() {value = dense<[[0.4]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[0.5]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[[0.6]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %7 = "tfl.pseudo_const"() {value = dense<[[0.7]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[[0.8]]> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %9 = "tfl.no_value"() {value} : () -> none
  %10 = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %11 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  %recurrent_input = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %16 = "tfl.unidirectional_sequence_lstm"(
    %arg0,
    %1, %2, %3, %4,
    %5, %6, %7, %8,
    %9, %9, %9,
    %10, %11,
    %10, %10,
    %9, %9,
    %recurrent_input, %cell_input,
    %9, %9, %9, %9) {
      cell_clip = 1.000000e+01 : f32,
      fused_activation_function = "TANH",
      proj_clip = 0.000000e+00 : f32,
      time_major = false} : (
        tensor<1x2x3xf32>,
        tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
        tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
        none, none, none,
        tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>,
        none, none,
        tensor<1x3xf32>, tensor<1x3xf32>,
        none, none, none, none) -> tensor<1x2x3xf32>
  %17 = "quantfork.stats"(%16) {layerStats = dense<[-0.1, 0.1]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  func.return %17 : tensor<1x2x3xf32>

  // CHECK: %[[NONE:.*]] = "tfl.no_value"() {value} : () -> none
  // CHECK: %[[DQ_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1xf16>) -> tensor<1x1xf32>
  // CHECK: %[[DQ_9:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3xf16>) -> tensor<3xf32>
  // CHECK: %[[DQ_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3xf16>) -> tensor<3xf32>
  // CHECK: %[[DQ_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3xf16>) -> tensor<1x3xf32>
  // CHECK: %[[DQ_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3xf16>) -> tensor<1x3xf32>
  // CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(
  // CHECK-SAME: %arg0,
  // CHECK-SAME: %[[DQ_1]], %[[DQ_2]], %[[DQ_3]], %[[DQ_4]],
  // CHECK-SAME: %[[DQ_5]], %[[DQ_6]], %[[DQ_7]], %[[DQ_8]],
  // CHECK-SAME: %[[NONE]], %[[NONE]], %[[NONE]],
  // CHECK-SAME: %[[DQ_9]], %[[DQ_10]], %[[DQ_9]], %[[DQ_9]],
  // CHECK-SAME: %[[NONE]], %[[NONE]],
  // CHECK-SAME: %[[DQ_11]], %[[DQ_12]],
  // CHECK-SAME: %[[NONE]], %[[NONE]], %[[NONE]], %[[NONE]]) {
  // CHECK-SAME: cell_clip = 1.000000e+01 : f32,
  // CHECK-SAME: fused_activation_function = "TANH",
  // CHECK-SAME: proj_clip = 0.000000e+00 : f32,
  // CHECK-SAME: time_major = false} : (
  // CHECK-SAME: tensor<1x2x3xf32>,
  // CHECK-SAME: tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
  // CHECK-SAME: tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
  // CHECK-SAME: none, none, none,
  // CHECK-SAME: tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>,
  // CHECK-SAME: none, none,
  // CHECK-SAME: tensor<1x3xf32>, tensor<1x3xf32>,
  // CHECK-SAME: none, none, none, none)
  // CHECK-SAME: -> tensor<1x2x3xf32>
}
