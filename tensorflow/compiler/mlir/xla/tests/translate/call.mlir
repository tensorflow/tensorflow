// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = call @callee(%0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK:  [[CALLEE_1:%.*]] ([[ARG_1:.*]]: s32[4], [[ARG_2:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG_1]] = s32[4] parameter(0)
// CHECK:  %[[ARG_2]] = s32[4] parameter(1)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  s32[4] add(s32[4] %[[ARG_1]], s32[4] %[[ARG_2]])

// CHECK:  [[CALLEE_2:%.*]] ([[ARG_3:.*]]: s32[4], [[ARG_4:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG_3]] = s32[4] parameter(0)
// CHECK:  %[[ARG_4]] = s32[4] parameter(1)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  s32[4] add(s32[4] %[[ARG_3]], s32[4] %[[ARG_4]])

// CHECK:  ENTRY [[MAIN:%.*]] ([[ARG:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG]] = s32[4] parameter(0)
// CHECK:  [[CALL_OUT:%.*]] = s32[4] call(s32[4] %[[ARG]], s32[4] %[[ARG]]), to_apply=[[CALLEE_1]]
// CHECK-LABEL:  ROOT
// CHECK-SAME:  s32[4] call(s32[4] [[CALL_OUT]], s32[4] [[CALL_OUT]]), to_apply=[[CALLEE_2]]
