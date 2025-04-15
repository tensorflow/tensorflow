// RUN: hlo-translate -split-input-file -mlir-to-hlo %s | FileCheck %s --check-prefix CHECK
// RUN: hlo-translate -split-input-file -mlir-to-hlo -print-large-constants %s | FileCheck %s --check-prefix CHECK-PRINT-LARGE

func.func @main(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  // CHECK: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  // CHECK-PRINT-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  %0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  func.return %0 : tensor<10xi32>
}

// -----

func.func @main(%arg0: tensor<11xi32>) -> tensor<11xi32> {
  // CHECK: constant({...})
  // CHECK-PRINT-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  %0 = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<11xi32>
  func.return %0 : tensor<11xi32>
}
