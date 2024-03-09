// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --check-prefix CHECK-SMALL
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -print-large-constants %s | FileCheck %s --check-prefix CHECK-LARGE

func.func @main(%arg0: tensor<10xi32>) -> tensor<10xi32> {
  %0 = mhlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>
  func.return %0 : tensor<10xi32>
}

// CHECK-SMALL: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
// CHECK-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

// -----

func.func @main(%arg0: tensor<11xi32>) -> tensor<11xi32> {
  %0 = mhlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<11xi32>
  func.return %0 : tensor<11xi32>
}

// CHECK-SMALL: constant({...})
// CHECK-LARGE: constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
