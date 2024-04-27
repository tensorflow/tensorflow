// RUN: tf-opt %s -tfl-prepare-quantize="quantize-signed=true post-training-quantize=true activation-number-of-bits=16" -cse | FileCheck %s

// CHECK-LABEL: QuantizeUnidirectionalLstmFullPerTensor
func.func @QuantizeUnidirectionalLstmFullPerTensor(%arg0: tensor<1x2x3xf32>) -> (tensor<1x2x3xf32>) {
  %input = "quantfork.stats"(%arg0) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
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
  %recurrent_stats = "quantfork.stats"(%recurrent_input) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %cell_stats = "quantfork.stats"(%cell_input) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %16 = "tfl.unidirectional_sequence_lstm"(
    %input,
    %1, %2, %3, %4,
    %5, %6, %7, %8,
    %9, %9, %9,
    %10, %11,
    %10, %10,
    %9, %9,
    %recurrent_stats, %cell_stats,
    %9, %9, %9, %9) {
      asymmetric_quantize_inputs = false,
      cell_clip = 1.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<0x!quant.calibrated<f32<0.0:1.0>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<0xf32>,
      input_to_forget_intermediate = tensor<0xf32>,
      input_to_input_intermediate = tensor<0xf32>,
      input_to_output_intermediate = tensor<0xf32>,
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

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x3x!quant.uniform<i16<-32767:32767>:f32, 3.0518509475997192E-5>>) -> tensor<1x2x3xf32>
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 7.8740158653634745E-4>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0015748031730726949>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0031496063461453898>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.003937007874015748>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0047244096365500624>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0055118109297564652>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0062992126922907796>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_9:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 2.4030322780124744E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 4.8060645560249487E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 7.2090970130772759E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 9.6121291120498974E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3x!quant.uniform<i16:f32, 3.0518043793392844E-5:-1>>) -> tensor<1x3xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3x!quant.uniform<i16:f32, 3.0517578125E-5>>) -> tensor<1x3xf32>
// CHECK: %[[lstm:.*]] = "tfl.unidirectional_sequence_lstm"(
// CHECK-SAME: %[[input_0]],
// CHECK-SAME: %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]],
// CHECK-SAME: %[[input_5]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[input_9]], %[[input_9]], %[[input_9]],
// CHECK-SAME: %[[input_10]], %[[input_11]], %[[input_12]], %[[input_13]],
// CHECK-SAME: %[[input_9]], %[[input_9]],
// CHECK-SAME: %[[input_14]], %[[input_15]],
// CHECK-SAME: %[[input_9]], %[[input_9]], %[[input_9]], %[[input_9]]) {
// CHECK-SAME: asymmetric_quantize_inputs = false,
// CHECK-SAME: cell_clip = 1.000000e+01 : f32,
// CHECK-SAME: effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8:f32, {{.*}}>>,
// CHECK-SAME: fused_activation_function = "TANH",
// CHECK-SAME: input_to_cell_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_forget_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_input_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_output_intermediate = tensor<0xf32>,
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
// CHECK: "tfl.quantize"(%[[lstm]]) {qtype = tensor<1x2x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<i16:f32, {{.*}}>>

}

// CHECK-LABEL: QuantizeUnidirectionalLstmFullPerAxis
func.func @QuantizeUnidirectionalLstmFullPerAxis(%arg0: tensor<1x2x3xf32>) -> (tensor<1x2x3xf32>) {
  %input = "quantfork.stats"(%arg0) {
    layerStats = dense<[0.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
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
  %recurrent_stats = "quantfork.stats"(%recurrent_input) {layerStats = dense<[0.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %cell_input = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %cell_stats = "quantfork.stats"(%cell_input) {
    layerStats = dense<[0.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 1 : i64
  } : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %16 = "tfl.unidirectional_sequence_lstm"(
    %input,
    %1, %2, %3, %4,
    %5, %6, %7, %8,
    %9, %9, %9,
    %10, %11,
    %10, %10,
    %9, %9,
    %recurrent_stats, %cell_stats,
    %9, %9, %9, %9) {
      asymmetric_quantize_inputs = false,
      cell_clip = 1.000000e+01 : f32,
      effective_hidden_scale_intermediate = tensor<0x!quant.calibrated<f32<0.0:1.0>>>,
      fused_activation_function = "TANH",
      input_to_cell_intermediate = tensor<0xf32>,
      input_to_forget_intermediate = tensor<0xf32>,
      input_to_input_intermediate = tensor<0xf32>,
      input_to_output_intermediate = tensor<0xf32>,
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
  %17 = "quantfork.stats"(%16) {
    layerStats = dense<[0.0, 1.0]> : tensor<2xf32>,
    axisStats = dense<[
      [-1.0, 1.0],
      [-8.0, 8.0],
      [-0.5, 0.5]
    ]> : tensor<3x2xf32>, axis = 2 : i64
  } : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  func.return %17 : tensor<1x2x3xf32>

// CHECK-DAG: %[[input_0:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x2x3x!quant.uniform<i16<-32767:32767>:f32, {{3.0518509475997192E-5}}>>) -> tensor<1x2x3xf32>
// CHECK-DAG: %[[input_1:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 7.8740158653634745E-4>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_2:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0015748031730726949>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_3:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0023622048182750312>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_4:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0031496063461453898>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_5:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.003937007874015748>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_6:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0047244096365500624>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_7:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0055118109297564652>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_8:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x1x!quant.uniform<i8<-127:127>:f32, 0.0062992126922907796>>) -> tensor<1x1xf32>
// CHECK-DAG: %[[input_9:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK-DAG: %[[input_10:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 2.4030322780124744E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_11:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 4.8060645560249487E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_12:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 7.2090970130772759E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_13:.*]] = "tfl.dequantize"({{.*}}) : (tensor<3x!quant.uniform<i32:f32, 9.6121291120498974E-8>>) -> tensor<3xf32>
// CHECK-DAG: %[[input_14:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3x!quant.uniform<i16:f32, 3.0518043793392844E-5:-1>>) -> tensor<1x3xf32>
// CHECK-DAG: %[[input_15:.*]] = "tfl.dequantize"({{.*}}) : (tensor<1x3x!quant.uniform<i16:f32, 3.0517578125E-5>>) -> tensor<1x3xf32>
// CHECK: %31 = "tfl.unidirectional_sequence_lstm"(
// CHECK-SAME: %[[input_0]],
// CHECK-SAME: %[[input_1]], %[[input_2]], %[[input_3]], %[[input_4]],
// CHECK-SAME: %[[input_5]], %[[input_6]], %[[input_7]], %[[input_8]],
// CHECK-SAME: %[[input_9]], %[[input_9]], %[[input_9]],
// CHECK-SAME: %[[input_10]], %[[input_11]], %[[input_12]], %[[input_13]],
// CHECK-SAME: %[[input_9]], %[[input_9]],
// CHECK-SAME: %[[input_14]], %[[input_15]],
// CHECK-SAME: %[[input_9]], %[[input_9]], %[[input_9]], %[[input_9]]) {
// CHECK-SAME: asymmetric_quantize_inputs = false,
// CHECK-SAME: cell_clip = 1.000000e+01 : f32, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8:f32, {{.*}}>>,
// CHECK-SAME: fused_activation_function = "TANH",
// CHECK-SAME: input_to_cell_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_forget_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_input_intermediate = tensor<0xf32>,
// CHECK-SAME: input_to_output_intermediate = tensor<0xf32>, proj_clip = 0.000000e+00 : f32, time_major = false} : (
// CHECK-SAME: tensor<1x2x3xf32>,
// CHECK-SAME: tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
// CHECK-SAME: tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>,
// CHECK-SAME: none, none, none,
// CHECK-SAME: tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>,
// CHECK-SAME: none, none,
// CHECK-SAME: tensor<1x3xf32>, tensor<1x3xf32>,
// CHECK-SAME: none, none, none, none)
// CHECK-SAME: -> tensor<1x2x3xf32>
// CHECK: %32 = "tfl.quantize"(%31) {qtype = tensor<1x2x3x!quant.uniform<i16:f32:2, {{{.*}},{{.*}},{{.*}}}>>, volatile} : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<i16:f32:2, {{{.*}},{{.*}},{{.*}}}>>

}

// CHECK-LABEL: QuantizeFixedOutputRangeInterfaceOpSoftmax
func.func @QuantizeFixedOutputRangeInterfaceOpSoftmax(%arg0: tensor<1x1xf32>) -> (tensor<1x1xf32>) {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %2 : tensor<1x1xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[sm:.*]] = "tfl.softmax"(%[[dq1]]) {{{.*}}} : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[sm]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>) -> tensor<1x1xf32>
}

// CHECK-LABEL: QuantizeFixedOutputRangeInterfaceOpL2Normalization
func.func @QuantizeFixedOutputRangeInterfaceOpL2Normalization(%arg0: tensor<1x1xf32>) -> (tensor<1x1xf32>) {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.l2_normalization"(%0) {fused_activation_function = "NONE"} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %2 : tensor<1x1xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[l2:.*]] = "tfl.l2_normalization"(%[[dq1]]) {{{.*}}} : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[l2]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>) -> tensor<1x1xf32>
}

// CHECK-LABEL: QuantizeFixedOutputRangeInterfaceOpLogistic
func.func @QuantizeFixedOutputRangeInterfaceOpLogistic(%arg0: tensor<1x1xf32>) -> (tensor<1x1xf32>) {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.logistic"(%0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %2 : tensor<1x1xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[lo:.*]] = "tfl.logistic"(%[[dq1]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[lo]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, 1.52587890625E-5:-32768>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, 1.52587890625E-5:-32768>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x1x!quant.uniform<i16:f32, 1.52587890625E-5:-32768>>) -> tensor<1x1xf32>
}

// CHECK-LABEL: QuantizeFixedOutputRangeInterfaceOpTanh
func.func @QuantizeFixedOutputRangeInterfaceOpTanh(%arg0: tensor<1x1xf32>) -> (tensor<1x1xf32>) {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %1 = "tfl.tanh"(%0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %2 : tensor<1x1xf32>

// CHECK: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[ta:.*]] = "tfl.tanh"(%[[dq1]]) : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[ta]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x1x!quant.uniform<i16:f32, 3.0517578125E-5>>) -> tensor<1x1xf32>
}

// CHECK-LABEL: QuantizeReshapeOp
func.func @QuantizeReshapeOp(%arg0: tensor<1x1x3xf32>) -> (tensor<1x3xf32>) {
  %1 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1x3xf32>) -> tensor<1x1x3xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[-1, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.reshape"(%1, %2) : (tensor<1x1x3xf32>, tensor<2xi32>) -> tensor<1x3xf32>
  %4 = "quantfork.stats"(%3) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %4 : tensor<1x3xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<[-1, 3]> : tensor<2xi32>
// CHECK-NEXT: %[[q1:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1x3xf32>) -> tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1x3xf32>
// CHECK-NEXT: %[[rs:.*]] = "tfl.reshape"(%[[dq1]], %[[cst]]) : (tensor<1x1x3xf32>, tensor<2xi32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[rs]]) {qtype = tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NEXT: return %[[dq2]] : tensor<1x3xf32>
}

// CHECK-LABEL: QuantizeFullyConnectedOp
func.func @QuantizeFullyConnectedOp(%arg0: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
  %1 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[[0.1, 0.1, 0.1]]> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %3 = "tfl.pseudo_const"() {value = dense<[0.1]> : tensor<1xf32>} : () -> tensor<1xf32>
  %4 = "tfl.fully_connected"(%1, %2, %3) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %5 = "quantfork.stats"(%4) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %5 : tensor<1x1xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-NEXT: %[[q1:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<1x!quant.uniform<i32:f32, {{.*}}>>, volatile} : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1xf32>
// CHECK-NEXT: %[[cst_0:.*]] = arith.constant dense<{{.*}}> : tensor<1x3xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[cst_0]]) {qtype = tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>, volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[q3:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq3:.*]] = "tfl.dequantize"(%[[q3]]) : (tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%[[dq3]], %[[dq2]], %[[dq1]]) {{{.*}}} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q4:.*]] = "tfl.quantize"(%[[fc]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq4:.*]] = "tfl.dequantize"(%[[q4]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: return %[[dq4]] : tensor<1x1xf32>
}

// CHECK-LABEL: QuantizeReshapeAndFullyConnectedOp
func.func @QuantizeReshapeAndFullyConnectedOp(%arg0: tensor<1x1x3xf32>) -> (tensor<1x1xf32>) {
  %1 = "quantfork.stats"(%arg0) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1x3xf32>) -> tensor<1x1x3xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[-1, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.reshape"(%1, %2) : (tensor<1x1x3xf32>, tensor<2xi32>) -> tensor<1x3xf32>
  %4 = "quantfork.stats"(%3) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[[0.1, 0.1, 0.1]]> : tensor<1x3xf32>} : () -> tensor<1x3xf32>
  %6 = "tfl.pseudo_const"() {value = dense<[0.1]> : tensor<1xf32>} : () -> tensor<1xf32>
  %7 = "tfl.fully_connected"(%4, %5, %6) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %8 = "quantfork.stats"(%7) {layerStats = dense<[-1.0, 1.0]> : tensor<2xf32>} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %8 : tensor<1x1xf32>

// CHECK: %[[cst:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-NEXT: %[[q1:.*]] = "tfl.quantize"(%[[cst]]) {qtype = tensor<1x!quant.uniform<i32:f32, {{.*}}>>, volatile} : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, {{.*}}>>
// CHECK-NEXT: %[[dq1:.*]] = "tfl.dequantize"(%[[q1]]) : (tensor<1x!quant.uniform<i32:f32, {{.*}}>>) -> tensor<1xf32>
// CHECK-NEXT: %[[cst_0:.*]] = arith.constant dense<{{.*}}> : tensor<1x3xf32>
// CHECK-NEXT: %[[q2:.*]] = "tfl.quantize"(%[[cst_0]]) {qtype = tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>, volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>
// CHECK-NEXT: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]]) : (tensor<1x3x!quant.uniform<i8<-127:127>:f32:0, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[cst_1:.*]] = arith.constant dense<[-1, 3]> : tensor<2xi32>
// CHECK-NEXT: %[[q3:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1x3xf32>) -> tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq3:.*]] = "tfl.dequantize"(%[[q3]]) : (tensor<1x1x3x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1x3xf32>
// CHECK-NEXT: %[[rs:.*]] = "tfl.reshape"(%[[dq3]], %[[cst_1]]) : (tensor<1x1x3xf32>, tensor<2xi32>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[q4:.*]] = "tfl.quantize"(%[[rs]]) {qtype = tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq4:.*]] = "tfl.dequantize"(%[[q4]]) : (tensor<1x3x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x3xf32>
// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%[[dq4]], %[[dq2]], %[[dq1]]) {{{.*}}} : (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT: %[[q5:.*]] = "tfl.quantize"(%[[fc]]) {qtype = tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>, volatile} : (tensor<1x1xf32>) -> tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>
// CHECK-NEXT: %[[dq5:.*]] = "tfl.dequantize"(%[[q5]]) : (tensor<1x1x!quant.uniform<i16:f32, {{.*}}>>) -> tensor<1x1xf32>
// CHECK-NEXT: return %[[dq5]] : tensor<1x1xf32>
}
