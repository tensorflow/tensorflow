// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

module @ragged_dot_non_contracting {
  func.func @main(%lhs : tensor<19x11x5xf32>, %rhs : tensor<3x5x7xf32>, %group_sizes : tensor<19x3xi64>) -> tensor<19x11x7xf32> {
    // CHECK: %[[ARG0:.+]] = f32[19,11,5] parameter(0)
    // CHECK: %[[ARG1:.+]] = f32[3,5,7] parameter(1)
    // CHECK: %[[ARG2:.+]] = s64[19,3] parameter(2)
    // CHECK: f32[19,11,7] ragged-dot(%[[ARG0]], %[[ARG1]], %[[ARG2]]), lhs_contracting_dims={2}, rhs_contracting_dims={1}, lhs_ragged_dims={1}, rhs_group_dims={0}
    %0 = "mhlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
      ragged_dot_dimension_numbers = #mhlo.ragged_dot<
        dot_dimension_numbers = <
          lhs_batching_dimensions = [],
          rhs_batching_dimensions = [],
          lhs_contracting_dimensions = [2],
          rhs_contracting_dimensions = [1]
        >,
        lhs_ragged_dimensions = [1],
        rhs_group_dimensions = [0]
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<19x11x5xf32>, tensor<3x5x7xf32>, tensor<19x3xi64>) -> tensor<19x11x7xf32>
    func.return %0 : tensor<19x11x7xf32>
  }
}

// -----

module @ragged_dot_contracting {
  func.func @main(%lhs : tensor<11x19x5xf32>, %rhs : tensor<19x5x7xf32>, %group_sizes : tensor<19x3xi64>) -> tensor<3x11x7xf32> {
    // CHECK: %[[ARG0:.+]] = f32[11,19,5] parameter(0)
    // CHECK: %[[ARG1:.+]] = f32[19,5,7] parameter(1)
    // CHECK: %[[ARG2:.+]] = s64[19,3] parameter(2)
    // CHECK: f32[3,11,7] ragged-dot(%[[ARG0]], %[[ARG1]], %[[ARG2]]), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}, lhs_ragged_dims={2}
    %0 = "mhlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
      ragged_dot_dimension_numbers = #mhlo.ragged_dot<
        dot_dimension_numbers = <
          lhs_batching_dimensions = [],
          rhs_batching_dimensions = [],
          lhs_contracting_dimensions = [1,2],
          rhs_contracting_dimensions = [0,1]
        >,
        lhs_ragged_dimensions = [2],
        rhs_group_dimensions = []
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<11x19x5xf32>, tensor<19x5x7xf32>, tensor<19x3xi64>) -> tensor<3x11x7xf32>
    func.return %0 : tensor<3x11x7xf32>
  }
}

// -----

module @ragged_dot_batch {
  func.func @main(%lhs : tensor<19x17x11x5xf32>, %rhs : tensor<19x17x5x7xf32>, %group_sizes : tensor<19x3xi64>) -> tensor<19x17x11x7xf32> {
    // CHECK: %[[ARG0:.+]] = f32[19,17,11,5] parameter(0)
    // CHECK: %[[ARG1:.+]] = f32[19,17,5,7] parameter(1)
    // CHECK: %[[ARG2:.+]] = s64[19,3] parameter(2)
    // CHECK: f32[19,17,11,7] ragged-dot(%[[ARG0]], %[[ARG1]], %[[ARG2]]), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}, lhs_ragged_dims={1}
    %0 = "mhlo.ragged_dot"(%lhs, %rhs, %group_sizes) {
      ragged_dot_dimension_numbers = #mhlo.ragged_dot<
        dot_dimension_numbers = <
          lhs_batching_dimensions = [0,1],
          rhs_batching_dimensions = [0,1],
          lhs_contracting_dimensions = [3],
          rhs_contracting_dimensions = [2]
        >,
        lhs_ragged_dimensions = [1],
        rhs_group_dimensions = []
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<19x17x11x5xf32>, tensor<19x17x5x7xf32>, tensor<19x3xi64>) -> tensor<19x17x11x7xf32>
    func.return %0 : tensor<19x17x11x7xf32>
  }
}
