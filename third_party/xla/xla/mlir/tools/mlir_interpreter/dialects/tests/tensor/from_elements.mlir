// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @from_elements() -> tensor<2x3xindex> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %ret = tensor.from_elements %c0, %c1, %c2, %c3, %c4, %c5 :  tensor<2x3xindex>
  return %ret : tensor<2x3xindex>
}

// CHECK-LABEL: @from_elements
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 1, 2], [3, 4, 5]]

func.func @empty() -> tensor<0xindex> {
  %ret = tensor.from_elements : tensor<0xindex>
  return %ret : tensor<0xindex>
}

// CHECK-LABEL: @empty
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: []
