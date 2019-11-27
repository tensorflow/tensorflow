// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

module {
  func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
    // CHECK:    [[VAL_1:%.*]] = pred[4] parameter(0)
    // CHECK:    [[VAL_2:%.*]] = pred[4] parameter(1)
    %0 = xla_hlo.xor %arg0, %arg1 : tensor<4xi1>
    // CHECK:   ROOT [[VAL_3:%.*]] = pred[4] xor(pred[4] [[VAL_1]], pred[4] [[VAL_2]])
    return %0 : tensor<4xi1>
  }
}
