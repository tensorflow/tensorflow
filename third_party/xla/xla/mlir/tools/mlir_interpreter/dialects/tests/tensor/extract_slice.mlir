// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extract() -> tensor<1x2x1xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.extract_slice %cst[0, 0, 0][1, 2, 1][1, 1, 1] : tensor<1x2x3xi32> to tensor<1x2x1xi32>
  return %ret : tensor<1x2x1xi32>
}

// CHECK-LABEL: @extract
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[[1], [4]]]

func.func @rank_reduction() -> tensor<2xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %ret = tensor.extract_slice %cst[0, 0, 0][1, 2, 1][1, 1, 1] : tensor<1x2x3xi32> to tensor<2xi32>
  return %ret : tensor<2xi32>
}

// CHECK-LABEL: @rank_reduction
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 4]

func.func @rank_reduction_to_dynamic() -> tensor<?xi32> {
  %cst = arith.constant dense<[[[1, 2, 3], [4, 5, 6]]]> : tensor<1x2x3xi32>
  %c2 = arith.constant 2 : index
  %ret = tensor.extract_slice %cst[0, 0, 0][1, %c2, 1][1, 1, 1] : tensor<1x2x3xi32> to tensor<?xi32>
  return %ret : tensor<?xi32>
}

// CHECK-LABEL: @rank_reduction_to_dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [1, 4]

func.func @no_rank_reduction_to_dynamic() -> tensor<1x?xi32> {
  %cst = arith.constant dense<[[0], [1], [2], [3], [4]]> : tensor<5x1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ret = tensor.extract_slice %cst[%c2, %c0] [1, %c1] [1, 1]
    : tensor<5x1xi32> to tensor<1x?xi32>
  return %ret : tensor<1x?xi32>
}

// CHECK-LABEL: @no_rank_reduction_to_dynamic
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[2]]

func.func @extract_from_extract() -> tensor<1x1xi32> {
  %cst = arith.constant dense<[[0], [1], [2], [3], [4]]> : tensor<5x1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %extracted = tensor.extract_slice %cst[%c2, %c0] [1, %c1] [1, 1]
    : tensor<5x1xi32> to tensor<1x?xi32>
  %ret = tensor.extract_slice %extracted[%c0, %c0] [1, 1] [1, 1]
    : tensor<1x?xi32> to tensor<1x1xi32>
  return %ret : tensor<1x1xi32>
}

// CHECK-LABEL: @extract_from_extract
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[2]]
