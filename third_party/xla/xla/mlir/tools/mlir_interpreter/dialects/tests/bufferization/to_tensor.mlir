// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @tensor() -> tensor<2xi16> {
  %cst = arith.constant dense<[43, 44]> : tensor<2xi16>
  %memref = bufferization.to_memref %cst : tensor<2xi16> to memref<2xi16>
  %tensor = bufferization.to_tensor %memref : memref<2xi16> to tensor<2xi16>
  return %tensor : tensor<2xi16>
}

// CHECK-LABEL: @tensor
// CHECK{LITERAL}: [43, 44]
