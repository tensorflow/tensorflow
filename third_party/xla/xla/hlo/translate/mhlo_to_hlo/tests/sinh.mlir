// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: f32[4] sinh
    %0 = "mhlo.sinh"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
    func.return %0 : tensor<4xf32>
  }
}
