// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @broadcast() -> (tensor<2x3xi32>, tensor<2x3xi32>) {
  %v = arith.constant dense<[1,2]> : tensor<2xi32>
  %init = tensor.empty() : tensor<2x3xi32>
  %bcast = linalg.broadcast
      ins(%v: tensor<2xi32>)
      outs(%init: tensor<2x3xi32>)
      dimensions = [1]
  func.return %init, %bcast : tensor<2x3xi32>, tensor<2x3xi32>
}

// CHECK-LABEL: @broadcast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0], [0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [2, 2, 2]]

func.func @bufferized() -> memref<2x3xi32> {
  %v = arith.constant dense<[1,2]> : memref<2xi32>
  %alloc = memref.alloc() : memref<2x3xi32>
  linalg.broadcast
      ins(%v: memref<2xi32>)
      outs(%alloc: memref<2x3xi32>)
      dimensions = [1]
  func.return %alloc : memref<2x3xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [2, 2, 2]]
