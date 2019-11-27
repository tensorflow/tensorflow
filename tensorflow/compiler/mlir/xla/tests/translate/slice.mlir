// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// CHECK-LABEL:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK-LABEL:  ROOT
// CHECK-SAME:  s32[1,2] slice(s32[3,4] [[ARG]]), slice={[1:2:1], [0:4:2]}
