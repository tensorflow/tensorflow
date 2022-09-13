// RUN: mlir-hlo-opt %s --scalarize --split-input-file | FileCheck %s

#map = affine_map<() -> ()>

func.func @zero_rank(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32>  {
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: func @zero_rank
// CHECK-SAME:    (%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>)
// CHECK-DAG:   %[[LHS_VAL:.*]] = tensor.extract %[[LHS]]
// CHECK-DAG:   %[[RHS_VAL:.*]] = tensor.extract %[[RHS]]
// CHECK:       %[[RES:.*]] = arith.addf %[[LHS_VAL]], %[[RHS_VAL]]
// CHECK:       %[[NEW_TENSOR_RES:.*]] = tensor.from_elements %[[RES]]
// CHECK:       return %[[NEW_TENSOR_RES]]

// -----


func.func @nonzero_rank(%lhs: tensor<1xf32>, %rhs: tensor<1x1xf32>)
    -> tensor<1x1x1xf32>  {
  %0 = linalg.init_tensor [1, 1, 1] : tensor<1x1x1xf32>
  %1 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%lhs, %rhs : tensor<1xf32>, tensor<1x1xf32>)
    outs(%0 : tensor<1x1x1xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x1xf32>
  return %1: tensor<1x1x1xf32>
}
// CHECK-LABEL: func @nonzero_rank
// CHECK-SAME:    (%[[LHS:.*]]: tensor<1xf32>, %[[RHS:.*]]: tensor<1x1xf32>)
// CHECK-DAG:     %[[LHS_VAL:.*]] = tensor.extract %[[LHS]]
// CHECK-DAG:     %[[RHS_VAL:.*]] = tensor.extract %[[RHS]]
// CHECK:         %[[RES:.*]] = arith.addf %[[LHS_VAL]], %[[RHS_VAL]]
// CHECK:         %[[NEW_TENSOR_RES:.*]] = tensor.from_elements %[[RES]]
// CHECK:         return %[[NEW_TENSOR_RES]]

// -----

#map = affine_map<() -> ()>

func.func @op_sequence(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32>  {
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<f32>

  %3 = linalg.init_tensor [] : tensor<f32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %1 : tensor<f32>, tensor<f32>)
    outs(%3 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.mulf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>

  %6 = linalg.init_tensor [] : tensor<f32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%1, %4 : tensor<f32>, tensor<f32>)
    outs(%6 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.divf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>

  return %7: tensor<f32>
}
// CHECK-LABEL: func @op_sequence
// CHECK-SAME:    (%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>)
// CHECK-DAG:   %[[LHS_VAL:.*]] = tensor.extract %[[LHS]]
// CHECK-DAG:   %[[RHS_VAL:.*]] = tensor.extract %[[RHS]]
// CHECK:       %[[RES:.*]] = arith.addf %[[LHS_VAL]], %[[RHS_VAL]]
// CHECK-DAG:   %[[LHS_VAL_:.*]] = tensor.extract %[[LHS]]
// CHECK:       %[[RES2:.*]] = arith.mulf %[[LHS_VAL_]], %[[RES]]
// CHECK:       %[[RES3:.*]] = arith.divf %[[RES]], %[[RES2]]
// CHECK:       %[[NEW_TENSOR_RES:.*]] = tensor.from_elements %[[RES3]]
// CHECK:       return %[[NEW_TENSOR_RES]]

// -----

#map = affine_map<() -> ()>

func.func @multiple_ops(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32>  {
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs : tensor<f32>, tensor<f32>)
    outs(%0 : tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4 : f32
    %3 = arith.mulf %2, %arg4 : f32
    linalg.yield %3 : f32
  } -> tensor<f32>
  return %1: tensor<f32>
}
// CHECK-LABEL: func @multiple_ops
// CHECK-SAME:    (%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>)
// CHECK-DAG:     %[[LHS_VAL:.*]] = tensor.extract %[[LHS]]
// CHECK-DAG:     %[[RHS_VAL:.*]] = tensor.extract %[[RHS]]
// CHECK:         %[[RES:.*]] = arith.addf %[[LHS_VAL]], %[[RHS_VAL]]
// CHECK:         %[[RES2:.*]] = arith.mulf %[[RES]], %[[RHS_VAL]]
// CHECK:         %[[NEW_TENSOR_RES:.*]] = tensor.from_elements %[[RES2]]
// CHECK:         return %[[NEW_TENSOR_RES]]
