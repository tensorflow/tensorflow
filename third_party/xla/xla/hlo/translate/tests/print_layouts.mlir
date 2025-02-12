// RUN: hlo-translate -mlir-to-hlo -print-layouts %s | FileCheck %s --check-prefix CHECK

// CHECK-LABEL: main
// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((s32[3,3]{0,1}, pred[]), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:  [[GTE1:%.*]] = (s32[3,3]{0,1}, pred[]) get-tuple-element(((s32[3,3]{0,1}, pred[]), token[]) [[INFEED]]), index=0
// CHECK:  [[GTE2:%.*]] = s32[3,3]{0,1} get-tuple-element((s32[3,3]{0,1}, pred[]) [[GTE1]]), index=0
// CHECK:  [[GTE3:%.*]] = pred[] get-tuple-element((s32[3,3]{0,1}, pred[]) [[GTE1]]), index=1
// CHECK:  [[GTE4:%.*]] = token[] get-tuple-element(((s32[3,3]{0,1}, pred[]), token[]) [[INFEED]]), index=1
func.func @main(%arg0: !stablehlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token> {
  %0:3 = "stablehlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0, 1], [0]]} : (!stablehlo.token) -> (tensor<3x3xi32>, tensor<i1>, !stablehlo.token)
  %1 = "stablehlo.tuple"(%0#0, %0#1) : (tensor<3x3xi32>, tensor<i1>) -> tuple<tensor<3x3xi32>, tensor<i1>>
  %2 = "stablehlo.tuple"(%1, %0#2) : (tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
  func.return %2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
}
