// RUN: mlir-hlo-opt %s --split-input-file --group-reduction-dimensions | \
// RUN: FileCheck %s

// CHECK-LABEL: @trailing_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x10x3x3xf32>
func @trailing_reduction(%arg : tensor<10x10x3x3xf32>) -> tensor<10x10xf32> {
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
  return %0 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: @leading_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30x4x5x6xf32>
func @leading_reduction(%arg : tensor<10x20x30x4x5x6xf32>)
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
  return %0 : tensor<4x5x6xf32>
}

// -----

// CHECK-LABEL: @unordered_reduction_dimensions
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30x4x5x6xf32>
func @unordered_reduction_dimensions(%arg : tensor<10x20x30x4x5x6xf32>)
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
  return %0 : tensor<4x5x6xf32>
}

// -----

// CHECK-LABEL: @reduction_to_rank1
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?xf32>
func @reduction_to_rank1(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32> {
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
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @full_reduction
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x20x30xf32>
func @full_reduction(%arg : tensor<10x20x30xf32>) -> tensor<f32> {
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
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @dont_touch
// CHECK-SAME:  %[[ARG:.*]]: tensor<3x4x5x6x7x8xf32>
func @dont_touch(%arg : tensor<3x4x5x6x7x8xf32>) -> tensor<3x4x7x8xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[ARG]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [2, 3]
  // CHECK-SAME:     : (tensor<3x4x5x6x7x8xf32>, tensor<f32>)
  // CHECK-SAME:     -> tensor<3x4x7x8xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<3x4x5x6x7x8xf32>, tensor<f32>) -> tensor<3x4x7x8xf32>
  return %0 : tensor<3x4x7x8xf32>
}

// -----

// CHECK-LABEL: @also_dont_touch_non_consecutive
// CHECK-SAME:  %[[ARG:.*]]: tensor<3x4x5x6x7x8xf32>
func @also_dont_touch_non_consecutive(%arg : tensor<3x4x5x6x7x8xf32>)
    -> tensor<5x6xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[ARG]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [0, 1, 4, 5]
  // CHECK-SAME:     : (tensor<3x4x5x6x7x8xf32>, tensor<f32>) -> tensor<5x6xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0)
      applies mhlo.add across dimensions = [0, 1, 4, 5]
      : (tensor<3x4x5x6x7x8xf32>, tensor<f32>) -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: @accept_dynamic_shape
// CHECK-SAME:  %[[ARG:.*]]: tensor<10x?x3x3xf32>
func @accept_dynamic_shape(%arg : tensor<10x?x3x3xf32>) -> tensor<10x?xf32> {
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
  return %0 : tensor<10x?xf32>
}

// -----

// CHECK-LABEL: @reject_dynamic_shape
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x3x3xf32>
func @reject_dynamic_shape(%arg : tensor<?x?x3x3xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = mhlo.constant dense<0.000000e+00>
  // CHECK:      %[[RESULT:.*]] = mhlo.reduce(%[[ARG]] init: %[[C0]])
  // CHECK-SAME:     applies mhlo.add across dimensions = [2, 3]
  // CHECK-SAME:     : (tensor<?x?x3x3xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK:      return %[[RESULT]]
  %c0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = mhlo.reduce(%arg init: %c0) applies mhlo.add across dimensions = [2, 3]
      : (tensor<?x?x3x3xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @ignore_if_multiple_operands
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?x3x3xf32>, %[[ARG1:.*]]: tensor<?x?x3x3xf32>
func @ignore_if_multiple_operands(%arg0: tensor<?x?x3x3xf32>,
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
  // CHECK:        "mhlo.return"(%[[A0_]], %[[A1_]])
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
  return %1#0, %1#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
