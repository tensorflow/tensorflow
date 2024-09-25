// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @fill() -> (tensor<2xi32>, tensor<2xi32>) {
  %c42 = arith.constant 42 : i32
  %init = tensor.empty() : tensor<2xi32>
  %fill = linalg.fill ins(%c42 : i32) outs(%init : tensor<2xi32>) -> tensor<2xi32>
  func.return %init, %fill : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: @fill
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [0, 0]
// CHECK-NEXT{LITERAL}: [42, 42]

func.func @bufferized() -> memref<2xi32> {
  %c42 = arith.constant 42 : i32
  %alloc = memref.alloc() : memref<2xi32>
  linalg.fill ins(%c42 : i32) outs(%alloc : memref<2xi32>)
  func.return %alloc : memref<2xi32>
}

// CHECK-LABEL: @bufferized
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [42, 42]
