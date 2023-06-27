// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @memref_element_cast() -> tensor<2xi8> {
  %c = arith.constant dense<[true, false]> : tensor<2xi1>
  %ret = "xla_cpu.memref_element_cast"(%c) : (tensor<2xi1>) -> tensor<2xi8>
  return %ret : tensor<2xi8>
}

// CHECK-LABEL: @memref_element_cast
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 0]
