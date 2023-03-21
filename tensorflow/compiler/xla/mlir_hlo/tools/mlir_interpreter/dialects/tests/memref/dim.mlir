// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dim() -> index {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x50xf32>
  %c1 = arith.constant 1 : index
  %dim = memref.dim %alloc, %c1 : memref<10x50xf32>
  return %dim : index
}

// CHECK-LABEL: @dim
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 50
