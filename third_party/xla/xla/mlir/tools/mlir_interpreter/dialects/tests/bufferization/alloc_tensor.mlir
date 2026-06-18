// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @static() -> tensor<1x2x3xi32> {
  %t = bufferization.alloc_tensor() : tensor<1x2x3xi32>
  return %t : tensor<1x2x3xi32>
}

// CHECK-LABEL: @static
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[0, 0, 0], [0, 0, 0]]]

func.func @dynamic() -> tensor<?x1xi32> {
  %c4 = arith.constant 4 : index
  %t = bufferization.alloc_tensor(%c4) : tensor<?x1xi32>
  return %t : tensor<?x1xi32>
}

// CHECK-LABEL: @dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0], [0], [0], [0]]

func.func @copy() -> tensor<i32> {
  %c = arith.constant dense<123> : tensor<i32>
  %t = bufferization.alloc_tensor() copy(%c) : tensor<i32>
  return %t : tensor<i32>
}

// CHECK-LABEL: @copy
// CHECK-NEXT: Results
// CHECK-NEXT: 123
