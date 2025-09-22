// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: f32[4] acos
  %0 = "mhlo.acos"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
