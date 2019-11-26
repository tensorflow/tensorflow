// RUN: tf-opt -tfl-split-merged-operands %s | FileCheck %s

func @testSingleLstm(%arg0: tensor<4 x f32>) -> tensor<4xf32> {
  // CHECK-LABEL: testSingleLstm
  // CHECK:  %[[CST_0:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[CST_1:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[LSTM:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %[[CST_0]], %[[CST_1]], %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

func @testMultipleLstms(%arg0: tensor<4 x f32>) -> tensor<4xf32> {
  // CHECK-LABEL: testMultipleLstms
  // CHECK:  %[[CST_0:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[CST_1:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[LSTM_1:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %[[CST_0]], %[[CST_1]], %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK:  %[[CST_2:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[CST_3:.*]] = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK:  %[[LSTM_2:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%[[LSTM_1]], %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %[[CST_2]], %[[CST_3]], %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %1 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "tfl.unidirectional_sequence_lstm"(%1, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
