// RUN: tac-translate -input-mlir -output-mlir -device-specs=NNAPI %s -o - 2>&1 | FileCheck %s

module {
  // CHECK-LABEL: main
  func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
    // CHECK:  [[VAL_0:%.*]] = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
    // CHECK:  [[VAL_1:%.*]] = tfl.mul [[VAL_0]], [[VAL_0]] {fused_activation_function = "NONE"} : tensor<4xf32
  }

  // CHECK-LABEL: pack
  func @pack(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    return %0 : tensor<2x1xf32>
    // CHECK: %[[VAL_0:.*]] = arith.constant dense<[2, 1]> : tensor<2xi32>
    // CHECK: %[[CONCAT:.*]] = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    // CHECK: %[[VAL_1:.*]] = "tfl.reshape"(%[[CONCAT]], %[[VAL_0]]) : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
    // CHECK: return %[[VAL_1]]
  }
}
