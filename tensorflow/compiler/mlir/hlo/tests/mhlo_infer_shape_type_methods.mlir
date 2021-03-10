// RUN: mlir-hlo-opt --mhlo-test-infer-shaped-type-methods --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// -----
// CHECK-LABEL: @select
// CHECK-SAME: (%[[PRED:.*]]: tensor<2x?xi1>,
func @select(%pred : tensor<2x?xi1>, %a : tensor<2x?xf32>, %b : tensor<2x?xf32>)
    -> tensor<2xindex> {
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[DIM:.*]] = dim %[[PRED]], %[[C1]] : tensor<2x?xi1>
  // CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[C2]], %[[DIM]] : tensor<2xindex>
  // CHECK: return %[[SHAPE]] : tensor<2xindex>
  %0 = "mhlo.select"(%pred, %a, %b)
      : (tensor<2x?xi1>, tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%0)
      : (tensor<2x?xf32>) -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

// -----
// CHECK-LABEL: @compare
// CHECK-SAME: (%[[A:.*]]: tensor<2x?xf32>,
func @compare(%a : tensor<2x?xf32>, %b : tensor<2x?xf32>) -> tensor<2xindex> {
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[DIM:.*]] = dim %[[A]], %[[C1]] : tensor<2x?xf32>
  // CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[C2]], %[[DIM]] : tensor<2xindex>
  // CHECK: return %[[SHAPE]] : tensor<2xindex>
  %0 = "mhlo.compare"(%a, %b) {comparison_direction = "NE"}
      : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xi1>
  %1 = "mhlo_test.reify_return_type_shapes"(%0)
      : (tensor<2x?xi1>) -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

