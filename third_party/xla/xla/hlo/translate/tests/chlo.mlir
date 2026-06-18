// RUN: hlo-translate -mlir-to-hlo -split-input-file %s | FileCheck %s

// Validating chlo.op -> mhlo.op -> hlo.op conversion.

// CHECK-LABEL: main
func.func @main(%arg : tensor<f16>) -> tensor<f16> {
  // CHECK: %[[ARG:.*]] = f16[] parameter(0)
  // CHECK: erf(%[[ARG]])
  %1 = "chlo.erf"(%arg) : (tensor<f16>) -> tensor<f16>
  func.return %1 : tensor<f16>
}

// -----

func.func @main(%arg : tensor<16x16xf32>) -> (tensor<16x8xf32>, tensor<16x8xi32>) {
  // CHECK: %[[ARG:.*]] = f32[16,16] parameter(0)
  // CHECK: (f32[16,8], s32[16,8]) topk(%[[ARG]]), k=8, largest=true
  %1:2 = chlo.top_k(%arg, k=8) : tensor<16x16xf32> -> (tensor<16x8xf32>, tensor<16x8xi32>)
  func.return %1#0, %1#1 : tensor<16x8xf32>, tensor<16x8xi32>
}

// -----

func.func @main(%lhs : tensor<2x11x5xf32>, %rhs : tensor<3x2x5x7xf32>, %group_sizes : tensor<3xi64>) -> tensor<2x11x7xf32> {
  // CHECK: ragged-dot
  %0 = "chlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
    ragged_dot_dimension_numbers = #chlo.ragged_dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [2],
      lhs_ragged_dimensions = [1],
      rhs_group_dimensions = [0]
    >,
    precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
  } : (tensor<2x11x5xf32>, tensor<3x2x5x7xf32>, tensor<3xi64>) -> tensor<2x11x7xf32>
  func.return %0 : tensor<2x11x7xf32>
}
