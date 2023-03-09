// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dim() -> index {
  %it = tensor.empty() : tensor<2x3x4xi32>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %it, %c1 : tensor<2x3x4xi32>
  return %dim : index
}

// CHECK-LABEL: @dim
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 3
