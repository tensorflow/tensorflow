// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "xla_hlo.copy"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK: ENTRY %main
// CHECK: [[ARG:%.*]] = s32[2] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = s32[2] copy(s32[2] [[ARG]])
