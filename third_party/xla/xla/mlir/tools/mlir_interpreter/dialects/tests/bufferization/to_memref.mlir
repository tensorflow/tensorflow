// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @memref() -> memref<2xi16> {
  %cst = arith.constant dense<[42, 43]> : tensor<2xi16>
  %memref = bufferization.to_buffer %cst : tensor<2xi16> to memref<2xi16>
  return %memref : memref<2xi16>
}

// CHECK-LABEL: @memref
// CHECK{LITERAL}: [42, 43]
