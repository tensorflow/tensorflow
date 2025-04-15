// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @tensor() -> tensor<2xi16> {
  %cst = arith.constant dense<[42, 43]> : tensor<2xi16>
  return %cst : tensor<2xi16>
}

// CHECK-LABEL: @tensor
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 43]

func.func @tensor_splat() -> tensor<2xi32> {
  %cst = arith.constant dense<42> : tensor<2xi32>
  return %cst : tensor<2xi32>
}

// CHECK-LABEL: @tensor_splat
// CHECK-NEXT: Results
// CHECK-NEXT: [42, 42]

func.func @scalar() -> i1 {
  %cst = arith.constant true
  return %cst : i1
}

// CHECK-LABEL: @scalar
// CHECK-NEXT: Results
// CHECK-NEXT: true

func.func @vector() -> vector<2x3xi32> {
  %cst = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : vector<2x3xi32>
  return %cst : vector<2x3xi32>
}

// CHECK-LABEL: @vector
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x3xi32>: [[1, 2, 3], [4, 5, 6]]
