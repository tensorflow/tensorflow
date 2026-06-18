// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @dot_general_2d() -> tensor<2x2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi32>
  %dot = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0],
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = []
    >
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %dot : tensor<2x2xi32>
}

// CHECK-LABEL: @dot_general_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[16, 19], [36, 43]]

func.func @dot_general_2d_2() -> tensor<2x2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[[4, 5], [6, 7]]> : tensor<2x2xi32>
  %dot = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1],
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = []
    >
  } : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %dot : tensor<2x2xi32>
}

// CHECK-LABEL: @dot_general_2d_2
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[19, 27], [28, 40]]

func.func @dot_general_2d_1d() -> tensor<2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0],
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = []
    >
  } : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %dot : tensor<2xi32>
}

// CHECK-LABEL: @dot_general_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [19, 28]

func.func @dot_general_batch_only() -> tensor<2x2xi32> {
  %lhs = mhlo.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %rhs = mhlo.constant dense<[4, 5]> : tensor<2xi32>
  %dot = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = [],
      lhs_batching_dimensions = [1],
      rhs_batching_dimensions = [0]
    >
  } : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %dot : tensor<2x2xi32>
}

// CHECK-LABEL: @dot_general_batch_only
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[4, 12], [10, 20]]