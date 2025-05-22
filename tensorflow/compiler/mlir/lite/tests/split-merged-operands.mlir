// RUN: litert-opt -tfl-split-merged-operands %s | FileCheck %s

func.func @testSingleLstm(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
  // CHECK-LABEL: testSingleLstm
  // CHECK-DAG:  %[[CST_0:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK-DAG:  %[[CST_1:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK:  %[[LSTM:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %[[CST_0]], %[[CST_1]], %arg0, %arg0, %arg0, %arg0) <{fused_activation_function = "NONE", time_major = true}> : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>

  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32> loc("Const")
  %1 = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %1 : tensor<4x4x4xf32>
}

func.func @testMultipleLstms(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
  // CHECK-LABEL: testMultipleLstms
  // CHECK-DAG:  %[[CST_0:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK-DAG:  %[[CST_1:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK:  %[[LSTM_1:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %[[CST_0]], %[[CST_1]], %arg0, %arg0, %arg0, %arg0) <{fused_activation_function = "NONE", time_major = true}> : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  // CHECK-DAG:  %[[CST_2:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK-DAG:  %[[CST_3:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
  // CHECK:  %[[LSTM_2:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%[[LSTM_1]], %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %[[CST_2]], %[[CST_3]], %arg0, %arg0, %arg0, %arg0) <{fused_activation_function = "NONE", time_major = true}> : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>

  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32> loc("Const")
  %1 = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  %2 = "tfl.unidirectional_sequence_lstm"(%1, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %0, %0, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %2 : tensor<4x4x4xf32>
}

func.func @testSingleLstmFloat16(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
  // CHECK-LABEL: testSingleLstm
  // CHECK-DAG:  %[[CST_0:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf16>}> : () -> tensor<4x4xf16>
  // CHECK-DAG:  %[[CST_1:.*]] = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4x4xf16>}> : () -> tensor<4x4xf16>
  // CHECK-DAG:  %[[DQ_0:.*]] = "tfl.dequantize"(%[[CST_0]]) : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK-DAG:  %[[DQ_1:.*]] = "tfl.dequantize"(%[[CST_1]]) : (tensor<4x4xf16>) -> tensor<4x4xf32>
  // CHECK:  %[[LSTM:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %[[DQ_0]], %[[DQ_1]], %arg0, %arg0, %arg0, %arg0) <{fused_activation_function = "NONE", time_major = true}> : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>

  %0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4x4xf16>} : () -> tensor<4x4xf16> loc("Const")
  %1 = "tfl.dequantize"(%0) : (tensor<4x4xf16>) -> tensor<4x4xf32>
  %2 = "tfl.unidirectional_sequence_lstm"(%arg2, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg1, %arg0, %arg1, %1, %1, %arg0, %arg0, %arg0, %arg0) {fused_activation_function = "NONE", time_major = true} : (tensor<4x4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %2 : tensor<4x4x4xf32>
}
