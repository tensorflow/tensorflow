// RUN: mlir-hlo-opt %s --split-input-file --group-reduction-dimensions | \
// RUN: FileCheck %s

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --group-reduction-dimensions="prefer-columns-reductions=false" | \
// RUN: FileCheck %s --check-prefix=CHECK-ROW-RED

// CHECK-LABEL: @trailing_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x10x3x3xf32>
func.func @trailing_reduction(%arg : tensor<10x10x3x3xf32>) -> tensor<10x10xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3]{{\]}}
  // CHECK-SAME:     : tensor<10x10x3x3xf32> into tensor<100x9xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [1]
  // CHECK-SAME:     : (tensor<100x9xf32>, tensor<f32>) -> tensor<100xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1]{{\]}} : tensor<100xf32> into tensor<10x10xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<10x10x3x3xf32>, tensor<f32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: @leading_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30x4x5x6xf32>
func.func @leading_reduction(%arg : tensor<10x20x30x4x5x6xf32>)
    -> tensor<4x5x6xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %arg0
  // CHECK-SAME:     {{\[}}[0, 1, 2], [3, 4, 5]{{\]}}
  // CHECK-SAME:     : tensor<10x20x30x4x5x6xf32> into tensor<6000x120xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<6000x120xf32>, tensor<f32>) -> tensor<120xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:     : tensor<120xf32> into tensor<4x5x6xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 1, 2]
      : (tensor<10x20x30x4x5x6xf32>, tensor<f32>) -> tensor<4x5x6xf32>
  func.return %0 : tensor<4x5x6xf32>
}

// -----

// CHECK-LABEL: @unordered_reduction_dimensions
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30x4x5x6xf32>
func.func @unordered_reduction_dimensions(%arg : tensor<10x20x30x4x5x6xf32>)
    -> tensor<4x5x6xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %arg0
  // CHECK-SAME:     {{\[}}[0, 1, 2], [3, 4, 5]{{\]}}
  // CHECK-SAME:     : tensor<10x20x30x4x5x6xf32> into tensor<6000x120xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<6000x120xf32>, tensor<f32>) -> tensor<120xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:     : tensor<120xf32> into tensor<4x5x6xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 2, 1]
      : (tensor<10x20x30x4x5x6xf32>, tensor<f32>) -> tensor<4x5x6xf32>
  func.return %0 : tensor<4x5x6xf32>
}

// -----

// CHECK-LABEL: @reduction_to_rank1
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?xf32>
func.func @reduction_to_rank1(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0], [1, 2]{{\]}}
  // CHECK-SAME:     : tensor<?x?x?xf32> into tensor<?x?xf32>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.minimum across dimensions = [1]
  // CHECK-SAME:     : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK:      return %[[RESULT]]
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.reduce(%arg0 init: %0)
      applies mhlo.minimum across dimensions = [1, 2]
      : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @full_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30xf32>
func.func @full_reduction(%arg : tensor<10x20x30xf32>) -> tensor<f32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:     : tensor<10x20x30xf32> into tensor<6000xf32>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<6000xf32>, tensor<f32>) -> tensor<f32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 1, 2]
      : (tensor<10x20x30xf32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @inner_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<3x4x5x6x7x8xf32>
func.func @inner_reduction(%arg : tensor<3x4x5x6x7x8xf32>) -> tensor<3x4x7x8xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3], [4, 5]{{\]}}
  // CHECK-SAME:     : tensor<3x4x5x6x7x8xf32> into tensor<12x30x56xf32>
  // CHECK:      %[[CTED:.*]] = "mhlo.transpose"(%[[CED]])
  // CHECK-SAME:     {permutation = dense<[1, 0, 2]> : tensor<3xi64>}
  // CHECK-SAME:     : (tensor<12x30x56xf32>) -> tensor<30x12x56xf32>
  // CHECK:      %[[CTCED:.*]] = tensor.collapse_shape %[[CTED]]
  // CHECK-SAME:     {{\[}}[0], [1, 2]{{\]}}
  // CHECK-SAME:     : tensor<30x12x56xf32> into tensor<30x672xf32>
  // CHECK:      %[[CTCRED:.*]] = mhlo.reduce(%[[CTCED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<30x672xf32>, tensor<f32>) -> tensor<672xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CTCRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2, 3]{{\]}}
  // CHECK-SAME:     : tensor<672xf32> into tensor<3x4x7x8xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<3x4x5x6x7x8xf32>, tensor<f32>) -> tensor<3x4x7x8xf32>
  func.return %0 : tensor<3x4x7x8xf32>
}

// -----

// CHECK-LABEL: @non_consecutive_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<3x4x5x6x7x8xf32>
func.func @non_consecutive_reduction(%arg : tensor<3x4x5x6x7x8xf32>)
    -> tensor<5x6xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3], [4, 5]{{\]}}
  // CHECK-SAME:     : tensor<3x4x5x6x7x8xf32> into tensor<12x30x56xf32>
  // CHECK:      %[[CTED:.*]] = "mhlo.transpose"(%[[CED]])
  // CHECK-SAME:     {permutation = dense<[0, 2, 1]> : tensor<3xi64>}
  // CHECK-SAME:     : (tensor<12x30x56xf32>) -> tensor<12x56x30xf32>
  // CHECK:      %[[CTCED:.*]] = tensor.collapse_shape %[[CTED]]
  // CHECK-SAME:     {{\[}}[0, 1], [2]{{\]}}
  // CHECK-SAME:     : tensor<12x56x30xf32> into tensor<672x30xf32>
  // CHECK:      %[[CTCRED:.*]] = mhlo.reduce(%[[CTCED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<672x30xf32>, tensor<f32>) -> tensor<30xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CTCRED]]
  // CHECK-SAME:     {{\[}}[0, 1]{{\]}}
  // CHECK-SAME:     : tensor<30xf32> into tensor<5x6xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 1, 4, 5]
      : (tensor<3x4x5x6x7x8xf32>, tensor<f32>) -> tensor<5x6xf32>
  func.return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: @accept_dynamic_shape
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x?x3x3xf32>
func.func @accept_dynamic_shape(%arg : tensor<10x?x3x3xf32>) -> tensor<10x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3]{{\]}}
  // CHECK-SAME:     : tensor<10x?x3x3xf32> into tensor<?x9xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [1]
  // CHECK-SAME:     : (tensor<?x9xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1]{{\]}} : tensor<?xf32> into tensor<10x?xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<10x?x3x3xf32>, tensor<f32>) -> tensor<10x?xf32>
  func.return %0 : tensor<10x?xf32>
}

// -----

// CHECK-LABEL: @more_than_one_dyn_parallel_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x3x3xf32>
func.func @more_than_one_dyn_parallel_dim(%arg : tensor<?x?x3x3xf32>)
    -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG:  %[[C0_:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %[[ARG]], %[[C0_]]
  // CHECK-DAG:  %[[D1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK-DAG:  %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]]
  // CHECK-DAG:  %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3]{{\]}}
  // CHECK-SAME:     : tensor<?x?x3x3xf32> into tensor<?x9xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [1]
  // CHECK-SAME:     : (tensor<?x9xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK:      %[[RESULT:.*]] = mhlo.dynamic_reshape %[[CRED]], %[[SHAPE]]
  // CHECK-SAME:     : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<?x?x3x3xf32>, tensor<f32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @ignore_if_multiple_operands
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?x3x3xf32>, %[[ARG1:.*]]: tensor<?x?x3x3xf32>
func.func @ignore_if_multiple_operands(%arg0: tensor<?x?x3x3xf32>,
    %arg1: tensor<?x?x3x3xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[RESULTS:.*]]:2 = mhlo.reduce(%[[ARG0]] init: %[[C0]]),
  // CHECK-SAME:     (%[[ARG1]] init: %[[C0]]) across dimensions = [2, 3]
  // CHECK-SAME:     : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, tensor<f32>,
  // CHECK-SAME:     tensor<f32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK:        reducer(%[[E0:.*]]: tensor<f32>, %[[A0:arg4]]: tensor<f32>)
  // CHECK-SAME:       (%[[E1:.*]]: tensor<f32>, %[[A1:.*]]: tensor<f32>)
  // CHECK-DAG:    %[[A0_:.*]] = mhlo.add %[[E0]], %[[A0]]
  // CHECK-DAG:    %[[A1_:.*]] = mhlo.add %[[E1]], %[[A1]]
  // CHECK:        mhlo.return %[[A0_]], %[[A1_]]
  // CHECK:      return %[[RESULTS]]#0, %[[RESULTS]]#1
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1:2 = mhlo.reduce(%arg0 init: %0), (%arg1 init: %0)
      across dimensions = [2, 3]
      : (tensor<?x?x3x3xf32>, tensor<?x?x3x3xf32>, tensor<f32>, tensor<f32>)
      -> (tensor<?x?xf32>, tensor<?x?xf32>)
    reducer(%elem0: tensor<f32>, %acc0: tensor<f32>)
        (%elem1: tensor<f32>, %acc1: tensor<f32>) {
    %acc0_ = mhlo.add %elem0, %acc0 : tensor<f32>
    %acc1_ = mhlo.add %elem1, %acc1 : tensor<f32>
    "mhlo.return"(%acc0_, %acc1_) : (tensor<f32>, tensor<f32>) -> ()
  }
  func.return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @leading_one_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<1x1x10x3x3xf32>
func.func @leading_one_dims(%arg : tensor<1x1x10x3x3xf32>) -> tensor<1x3xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1, 2, 3], [4]{{\]}}
  // CHECK-SAME:     : tensor<1x1x10x3x3xf32> into tensor<30x3xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<30x3xf32>, tensor<f32>) -> tensor<3xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1]{{\]}}
  // CHECK-SAME:     : tensor<3xf32> into tensor<1x3xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 2, 3]
      : (tensor<1x1x10x3x3xf32>, tensor<f32>) -> tensor<1x3xf32>
  func.return %0 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: @trailing_one_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x3x3x1x1xf32>
func.func @trailing_one_dims(%arg : tensor<10x3x3x1x1xf32>) -> tensor<10x1x1xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0], [1, 2, 3, 4]{{\]}}
  // CHECK-SAME:     : tensor<10x3x3x1x1xf32> into tensor<10x9xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [1]
  // CHECK-SAME:     : (tensor<10x9xf32>, tensor<f32>) -> tensor<10xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:     : tensor<10xf32> into tensor<10x1x1xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [1, 2]
      : (tensor<10x3x3x1x1xf32>, tensor<f32>) -> tensor<10x1x1xf32>
  func.return %0 : tensor<10x1x1xf32>
}

// -----

// CHECK-LABEL: @inner_one_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x1x3x1x9xf32>
func.func @inner_one_dims(%arg : tensor<10x1x3x1x9xf32>) -> tensor<1x1x9xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1, 2, 3], [4]{{\]}}
  // CHECK-SAME:     : tensor<10x1x3x1x9xf32> into tensor<30x9xf32>
  // CHECK:      %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<30x9xf32>, tensor<f32>) -> tensor<9xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:     : tensor<9xf32> into tensor<1x1x9xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [0, 2]
      : (tensor<10x1x3x1x9xf32>, tensor<f32>) -> tensor<1x1x9xf32>
  func.return %0 : tensor<1x1x9xf32>
}

// -----

// CHECK-LABEL: @all_one_dims
// CHECK-SAME:  %[[ARG:.*]]: tensor<1x1x1x1x1xf32>
func.func @all_one_dims(%arg : tensor<1x1x1x1x1xf32>) -> tensor<1x1x1xf32> {
  // CHECK:      %[[RESULT:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1, 2], [3], [4]{{\]}}
  // CHECK-SAME:     : tensor<1x1x1x1x1xf32> into tensor<1x1x1xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [0, 2]
      : (tensor<1x1x1x1x1xf32>, tensor<f32>) -> tensor<1x1x1xf32>
  func.return %0 : tensor<1x1x1xf32>
}

// -----

// CHECK-LABEL: @all_one_dims_full_reduce
// CHECK-SAME:  %[[ARG:.*]]: tensor<1x1x1x1x1xf32>
func.func @all_one_dims_full_reduce(%arg : tensor<1x1x1x1x1xf32>) -> tensor<f32> {
  // CHECK:      %[[RESULT:.*]] = tensor.collapse_shape %[[ARG]] []
  // CHECK-SAME:     : tensor<1x1x1x1x1xf32> into tensor<f32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 1, 2, 3, 4]
      : (tensor<1x1x1x1x1xf32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @not_really_a_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x5x1x3xf32>
func.func @not_really_a_reduction(%arg : tensor<10x5x1x3xf32>)
    -> tensor<10x5x3xf32> {
  // CHECK:      %[[RESULT:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0], [1, 2], [3]{{\]}}
  // CHECK-SAME:     : tensor<10x5x1x3xf32> into tensor<10x5x3xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2]
      : (tensor<10x5x1x3xf32>, tensor<f32>) -> tensor<10x5x3xf32>
  func.return %0 : tensor<10x5x3xf32>
}

// -----

// CHECK-LABEL: @needs_transpose
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x11x12x13x14x15x16x17x18x19xf32>
func.func @needs_transpose(%arg : tensor<10x11x12x13x14x15x16x17x18x19xf32>)
    -> tensor<10x11x14x15x18x19xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]{{\]}}
  // CHECK-SAME:     : tensor<10x11x12x13x14x15x16x17x18x19xf32>
  // CHECK-SAME:     into tensor<110x156x210x272x342xf32>
  // CHECK:      %[[CTED:.*]] = "mhlo.transpose"(%[[CED]])
  // CHECK-SAME:     {permutation = dense<[1, 3, 0, 2, 4]> : tensor<5xi64>}
  // CHECK-SAME:     : (tensor<110x156x210x272x342xf32>)
  // CHECK-SAME:     -> tensor<156x272x110x210x342xf32>
  // CHECK:      %[[CTCED:.*]] = tensor.collapse_shape %[[CTED]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3, 4]{{\]}}
  // CHECK-SAME:     : tensor<156x272x110x210x342xf32>
  // CHECK-SAME:     into tensor<42432x7900200xf32>
  // CHECK:      %[[CTCRED:.*]] = mhlo.reduce(%[[CTCED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<42432x7900200xf32>, tensor<f32>)
  // CHECK-SAME:     -> tensor<7900200xf32>
  // CHECK:      %[[RESULT:.*]] = tensor.expand_shape %[[CTCRED]]
  // CHECK-SAME:     {{\[}}[0, 1, 2, 3, 4, 5]{{\]}}
  // CHECK-SAME:     : tensor<7900200xf32> into tensor<10x11x14x15x18x19xf32>
  // CHECK:      return %[[RESULT]] : tensor<10x11x14x15x18x19xf32>
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [2, 3, 6, 7]
      : (tensor<10x11x12x13x14x15x16x17x18x19xf32>, tensor<f32>)
      -> tensor<10x11x14x15x18x19xf32>
  func.return %0 : tensor<10x11x14x15x18x19xf32>
}

// CHECK-ROW-RED-LABEL: @needs_transpose
// CHECK-ROW-RED-SAME:  %[[ARG:.*]]: tensor<10x11x12x13x14x15x16x17x18x19xf32>

// CHECK-ROW-RED-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
// CHECK-ROW-RED:      %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
// CHECK-ROW-RED-SAME:     {{\[}}[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]{{\]}}
// CHECK-ROW-RED-SAME:     : tensor<10x11x12x13x14x15x16x17x18x19xf32>
// CHECK-ROW-RED-SAME:     into tensor<110x156x210x272x342xf32>
// CHECK-ROW-RED:      %[[CTED:.*]] = "mhlo.transpose"(%[[CED]])
// CHECK-ROW-RED-SAME:     {permutation = dense<[0, 2, 4, 1, 3]>
// CHECK-ROW-RED-SAME:     : tensor<5xi64>} : (tensor<110x156x210x272x342xf32>)
// CHECK-ROW-RED-SAME:     -> tensor<110x210x342x156x272xf32>
// CHECK-ROW-RED:      %[[CTCED:.*]] = tensor.collapse_shape %[[CTED]]
// CHECK-ROW-RED-SAME:     {{\[}}[0, 1, 2], [3, 4]{{\]}}
// CHECK-ROW-RED-SAME:     : tensor<110x210x342x156x272xf32>
// CHECK-ROW-RED-SAME:     into tensor<7900200x42432xf32>
// CHECK-ROW-RED:      %[[CTCRED:.*]] = mhlo.reduce(%[[CTCED]] init: %[[C0]])
// CHECK-ROW-RED-SAME:     applies mhlo.add across dimensions = [1]
// CHECK-ROW-RED-SAME:     : (tensor<7900200x42432xf32>, tensor<f32>)
// CHECK-ROW-RED-SAME:     -> tensor<7900200xf32>
// CHECK-ROW-RED:      %[[RESULT:.*]] = tensor.expand_shape %[[CTCRED]]
// CHECK-ROW-RED-SAME:     {{\[}}[0, 1, 2, 3, 4, 5]{{\]}} : tensor<7900200xf32>
// CHECK-ROW-RED-SAME:     into tensor<10x11x14x15x18x19xf32>
// CHECK-ROW-RED:      return %[[RESULT]] : tensor<10x11x14x15x18x19xf32>

// -----

// CHECK-LABEL: @needs_transpose_and_dynamic_reshape
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x11x12x13x14x15x16x17x18x?xf32>
func.func @needs_transpose_and_dynamic_reshape(
    %arg : tensor<?x11x12x13x14x15x16x17x18x?xf32>)
    -> tensor<?x11x14x15x18x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK-DAG:  %[[CI0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[CI11:.*]] = arith.constant 11
  // CHECK-DAG:  %[[CI14:.*]] = arith.constant 14
  // CHECK-DAG:  %[[CI15:.*]] = arith.constant 15
  // CHECK-DAG:  %[[CI18:.*]] = arith.constant 18
  // CHECK-DAG:  %[[CI9:.*]] = arith.constant 9
  // CHECK:      %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CI0]]
  // CHECK-SAME:     : tensor<?x11x12x13x14x15x16x17x18x?xf32>
  // CHECK:      %[[DIM9:.*]] = tensor.dim %[[ARG]], %[[CI9]]
  // CHECK-SAME:     : tensor<?x11x12x13x14x15x16x17x18x?xf32>
  // CHECK-DAG:  %[[SHAPE:.*]] = tensor.from_elements %[[DIM0]], %[[CI11]],
  // CHECK-SAME:     %[[CI14]], %[[CI15]], %[[CI18]], %[[DIM9]]
  // CHECK-DAG:  %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]{{\]}}
  // CHECK-SAME:     : tensor<?x11x12x13x14x15x16x17x18x?xf32>
  // CHECK-SAME:     into tensor<?x156x210x272x?xf32>
  // CHECK:      %[[CTED:.*]] = "mhlo.transpose"(%[[CED]])
  // CHECK-SAME:     {permutation = dense<[1, 3, 0, 2, 4]> : tensor<5xi64>}
  // CHECK-SAME:     : (tensor<?x156x210x272x?xf32>)
  // CHECK-SAME:     -> tensor<156x272x?x210x?xf32>
  // CHECK:      %[[CTCED:.*]] = tensor.collapse_shape %[[CTED]]
  // CHECK-SAME:     {{\[}}[0, 1], [2, 3, 4]{{\]}}
  // CHECK-SAME:     : tensor<156x272x?x210x?xf32> into tensor<42432x?xf32>
  // CHECK:      %[[CTCRED:.*]] = mhlo.reduce(%[[CTCED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<42432x?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK-DAG:  %[[RESULT:.*]] = mhlo.dynamic_reshape %[[CTCRED]],
  // CHECK-SAME:     %[[SHAPE]]
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [2, 3, 6, 7]
      : (tensor<?x11x12x13x14x15x16x17x18x?xf32>, tensor<f32>)
      -> tensor<?x11x14x15x18x?xf32>
  func.return %0 : tensor<?x11x14x15x18x?xf32>
}

// -----

// CHECK-LABEL: @transpose_wo_collapse
// CHECK-SAME:  %[[ARG:.*]]: tensor<2x3x4xf32>
func.func @transpose_wo_collapse(%arg : tensor<2x3x4xf32>) -> tensor<3xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[TED:.*]] = "mhlo.transpose"(%[[ARG]])
  // CHECK-SAME:     {permutation = dense<[0, 2, 1]> : tensor<3xi64>}
  // CHECK-SAME:     : (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
  // CHECK:      %[[TCED:.*]] = tensor.collapse_shape %[[TED]]
  // CHECK-SAME:     {{\[}}[0, 1], [2]{{\]}}
  // CHECK-SAME:     : tensor<2x4x3xf32> into tensor<8x3xf32>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[TCED]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0]
  // CHECK-SAME:     : (tensor<8x3xf32>, tensor<f32>) -> tensor<3xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [0, 2]
      : (tensor<2x3x4xf32>, tensor<f32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: @requires_scalar_expansion
// CHECK-SAME:  %[[ARG:.*]]: tensor<1x1x?xi32>
func.func @requires_scalar_expansion(%arg0: tensor<1x1x?xi32>) -> tensor<1xi32> {
  // CHECK-DAG:   %[[C0:.*]] = mhlo.constant dense<1>
  // CHECK:       %[[CED:.*]] = tensor.collapse_shape %[[ARG]]
  // CHECK-SAME:      {{\[}}[0, 1, 2]{{\]}}
  // CHECK-SAME:      : tensor<1x1x?xi32> into tensor<?xi32>
  // CHECK:       %[[CRED:.*]] = mhlo.reduce(%[[CED]] init: %[[C0]])
  // CHECK-SAME:      applies mhlo.multiply across dimensions = [0]
  // CHECK-SAME:      : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
  // CHECK:       %[[RESULT:.*]] = tensor.expand_shape %[[CRED]] []
  // CHECK-SAME:      : tensor<i32> into tensor<1xi32>
  // CHECK:       return %[[RESULT]]
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0)
      applies mhlo.multiply across dimensions = [0, 2]
      : (tensor<1x1x?xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}
