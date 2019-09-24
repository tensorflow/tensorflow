// RUN: tf-opt -tfl-legalize-ophint-func-op %s  -split-input-file | FileCheck %s

module {
  // CHECK-LABEL: func @testConvertUnidirectionalSequenceRNN
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<1x3xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<1x3xf32>)
  func @testConvertUnidirectionalSequenceRNN(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x4xf32> {
    // CHECK:  %[[CST:.*]] = constant dense<0.000000e+00> : tensor<1x4xf32>
    // CHECK:  %[[CST_0:.*]] = constant dense<0.000000e+00> : tensor<4xf32>
    // CHECK:  %[[CST_1:.*]] = constant dense<0.000000e+00> : tensor<4x3xf32>
    // CHECK:  %[[CST_2:.*]] = constant dense<0.000000e+00> : tensor<4x4xf32>
    // CHECK:  %[[PACKED_INPUT:[a-z0-9]*]] = "tfl.pack"(%[[ARG_0]], %[[ARG_1]]) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x1x3xf32>
    // CHECK:  %[[FUSED_OUTPUT:[a-z0-9]*]] = "tfl.unidirectional_sequence_rnn"(%[[PACKED_INPUT]], %[[CST_1]], %[[CST_2]], %[[CST_0]], %[[CST]]) {fused_activation_function = "TANH", time_major = true} : (tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>) -> tensor<2x1x4xf32>
    // CHECK:  %[[UNPACK:[0-9]*]]:2 = "tfl.unpack"(%[[FUSED_OUTPUT]]) {axis = 0 : i32, num = 2 : i32} : (tensor<2x1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>)

    %cst = constant dense<0.000000e+00> : tensor<1x4xf32>
    %cst0 = constant dense<0.000000e+00> : tensor<4xf32>
    %cst1 = constant dense<0.000000e+00> : tensor<4x3xf32>
    %cst2 = constant dense<0.000000e+00> : tensor<4x4xf32>
    %2 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x1x3xf32>
    %3 = call @a9211722c23011e9875cdc4a3e957995(%2, %cst1, %cst2, %cst0, %cst) : (tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>) -> tensor<2x1x4xf32>
    %4:2 = "tfl.unpack"(%3) {axis = 0 : i32, num = 2 : i32} : (tensor<2x1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>)
    return %4#0 : tensor<1x4xf32>
  }
  func @a9211722c23011e9875cdc4a3e957995(tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>) -> tensor<2x1x4xf32>
  attributes  {_tflite_function_name = "UnidirectionalSequenceRnn"}
}

// -----

module {
  // CHECK-LABEL: func @testConvertUnidirectionalSequenceLSTM
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<1x3xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<1x3xf32>)
  func @testConvertUnidirectionalSequenceLSTM(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x4xf32> {
    // CHECK:  %[[CST:.*]] = constant dense<0.000000e+00> : tensor<4x4xf32>
    // CHECK:  %[[CST_0:.*]] = constant dense<0.000000e+00> : tensor<4x4xf32>
    // CHECK:  %[[CST_1:.*]] = constant dense<0.000000e+00> : tensor<4x4xf32>
    // CHECK:  %[[CST_2:.*]] = constant dense<0.000000e+00> : tensor<4x4xf32>
    // CHECK:  %[[CST_3:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
    // CHECK:  %[[CST_4:.*]] = constant dense<0.000000e+00> : tensor<4x3xf32>
    // CHECK:  %[[CST_5:.*]] = constant dense<0.000000e+00> : tensor<4x3xf32>
    // CHECK:  %[[CST_6:.*]] = constant dense<0.000000e+00> : tensor<4x3xf32>
    // CHECK:  %[[CST_7:.*]] = constant dense<0.000000e+00> : tensor<4x3xf32>
    // CHECK:  %[[CST_8:.*]] = constant dense<0.000000e+00> : tensor<4xf32>
    // CHECK:  %[[CST_9:.*]] = constant dense<0.000000e+00> : tensor<1x4xf32>
    // CHECK:  %[[PACKED_INPUT:[a-z0-9]*]] = "tfl.pack"(%[[ARG_0]], %[[ARG_1]]) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x1x3xf32>
    // CHECK:  %[[CST_10:.*]] = constant unit
    // CHECK:  %[[FUSED_OUTPUT:[a-z0-9]*]] = "tfl.unidirectional_sequence_lstm"(%[[PACKED_INPUT]], %[[CST_6]], %[[CST_5]], %[[CST_4]], %[[CST_7]], %[[CST_1]], %[[CST_0]], %[[CST]], %[[CST_2]], %[[CST_10]], %[[CST_10]], %[[CST_10]], %[[CST_8]], %[[CST_3]], %[[CST_8]], %[[CST_8]], %[[CST_10]], %[[CST_10]], %[[CST_9]], %[[CST_9]], %[[CST_10]], %[[CST_10]], %[[CST_10]], %[[CST_10]]) {fused_activation_function = "TANH", time_major = true} : (tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, none, none, none, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, none, none, tensor<1x4xf32>, tensor<1x4xf32>, none, none, none, none) -> tensor<2x1x4xf32>
    // CHECK:  %[[UNPACK:[0-9]*]]:2 = "tfl.unpack"(%[[FUSED_OUTPUT]]) {axis = 0 : i32, num = 2 : i32} : (tensor<2x1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>)

    %cst = constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_0 = constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_1 = constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_2 = constant dense<0.000000e+00> : tensor<4x4xf32>
    %cst_3 = constant dense<1.000000e+00> : tensor<4xf32>
    %cst_4 = constant dense<0.000000e+00> : tensor<4x3xf32>
    %cst_5 = constant dense<0.000000e+00> : tensor<4x3xf32>
    %cst_6 = constant dense<0.000000e+00> : tensor<4x3xf32>
    %cst_7 = constant dense<0.000000e+00> : tensor<4x3xf32>
    %cst_8 = constant dense<0.000000e+00> : tensor<4xf32>
    %cst_9 = constant dense<0.000000e+00> : tensor<1x4xf32>
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<2x1x3xf32>
    %1:2 = call @a7addbdad08811e9b52cdc4a3e957995(%0, %cst_6, %cst_5, %cst_4, %cst_7, %cst_1, %cst_0, %cst, %cst_2, %cst_8, %cst_3, %cst_8, %cst_8, %cst_9, %cst_9) : (tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<2x1x4xf32>)
    %2:2 = "tfl.unpack"(%1#1) {axis = 0 : i32, num = 2 : i32} : (tensor<2x1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>)
    return %2#1 : tensor<1x4xf32>
  }
  func @a7addbdad08811e9b52cdc4a3e957995(tensor<2x1x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<2x1x4xf32>)
  attributes  {_tflite_function_input_index = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 18 : i32, 19 : i32], _tflite_function_name = "UnidirectionalSequenceLstm"}
}
