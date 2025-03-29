// RUN: hlo-translate -mlir-to-hlo -split-input-file %s | FileCheck %s

// Validating chlo.op -> mhlo.op -> hlo.op conversion.

// CHECK-LABEL: main
func.func @main(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: f16[] parameter(0)
  // CHECK: erf
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}
