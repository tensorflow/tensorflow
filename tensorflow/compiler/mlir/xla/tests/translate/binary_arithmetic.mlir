// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

module {
  func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
    // CHECK: [[VAL_1:%.*]] = s32[4] parameter(0)
    // CHECK: [[VAL_2:%.*]] = s32[4] parameter(1)
    // CHECK:  [[ATAN2:%.*]] = s32[4] atan2(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
    %0 = xla_hlo.atan2 %arg0, %arg1 : tensor<4xi32>

    // CHECK:  [[SHL:%.*]] = s32[4] shift-left(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
    %1 = xla_hlo.shift_left %arg0, %arg1 : tensor<4xi32>

    // CHECK:  [[SHRA:%.*]] = s32[4] shift-right-arithmetic(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
    %2 = xla_hlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>

    // CHECK:  [[SHRL:%.*]] = s32[4] shift-right-logical(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
    %3 = xla_hlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>

    // CHECK-LABEL:  ROOT
    // CHECK-SAME:  [[VAL_7:%.*]] = (s32[4], s32[4], s32[4], s32[4]) tuple(s32[4] [[ATAN2]], s32[4] [[SHL]], s32[4] [[SHRA]], s32[4] [[SHRL]])
    return %0, %1, %2, %3 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
  }
}
