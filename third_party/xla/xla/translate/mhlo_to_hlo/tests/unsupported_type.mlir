// RUN: not xla-translate -split-input-file -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

// CHECK: result #0 type is not supported
func.func @main() {
  %0 = arith.constant dense<1> : tensor<1xindex>
  func.return
}
