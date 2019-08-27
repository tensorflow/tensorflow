// RUN: tf-opt -tfl-legalize-ophint-func-op %s | FileCheck %s

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
