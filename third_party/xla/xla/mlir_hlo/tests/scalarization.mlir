// RUN: mlir-hlo-opt %s --scalarize --split-input-file | FileCheck %s

#map = affine_map<() -> ()>

func.func @zero_rank(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32>  {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs: tensor<f32>, tensor<f32>)
    outs(%0: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4: f32
    linalg.yield %2: f32
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

func.func @linalg_index(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  %0 = tensor.empty() : tensor<1xf64>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    outs(%0 : tensor<1xf64>) {
  ^bb0(%arg1: f64):
    %2 = linalg.index 0 : index
    %3 = tensor.extract %arg0[%2] : tensor<1xf64>
    linalg.yield %3 : f64
  } -> tensor<1xf64>
  return %1 : tensor<1xf64>
}
// CHECK-LABEL: func @linalg_index
// CHECK-SAME:      (%[[ARG:.*]]: tensor<1xf64>)
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0
// CHECK-NEXT:    %[[ELEM:.*]] = tensor.extract %[[ARG]][%[[C0]]]
// CHECK-NEXT:    tensor.from_elements %[[ELEM]]

// -----


func.func @nonzero_rank(%lhs: tensor<1xf32>, %rhs: tensor<1x1xf32>)
    -> tensor<1x1x1xf32>  {
  %0 = tensor.empty() : tensor<1x1x1xf32>
  %1 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%lhs, %rhs: tensor<1xf32>, tensor<1x1xf32>)
    outs(%0: tensor<1x1x1xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4: f32
    linalg.yield %2: f32
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
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs: tensor<f32>, tensor<f32>)
    outs(%0: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4: f32
    linalg.yield %2: f32
  } -> tensor<f32>

  %3 = tensor.empty() : tensor<f32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %1: tensor<f32>, tensor<f32>)
    outs(%3: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.mulf %arg3, %arg4: f32
    linalg.yield %5: f32
  } -> tensor<f32>

  %6 = tensor.empty() : tensor<f32>
  %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%1, %4: tensor<f32>, tensor<f32>)
    outs(%6: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.divf %arg3, %arg4: f32
    linalg.yield %5: f32
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
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs: tensor<f32>, tensor<f32>)
    outs(%0: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4: f32
    %3 = arith.mulf %2, %arg4: f32
    linalg.yield %3: f32
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

// -----

func.func @outside_yield() -> tensor<1x1xi1>  {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<1x1xi1>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       outs(%0 : tensor<1x1xi1>) {
  ^bb0(%arg1: i1):
    linalg.yield %true : i1
  } -> tensor<1x1xi1>
  return %1: tensor<1x1xi1>
}

// CHECK-LABEL: func @outside_yield
// CHECK:         %[[CST:.*]] = arith.constant dense<true> : tensor<1x1xi1>
// CHECK:         return %[[CST]]

// -----

#map0 = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
func.func @extra_argument(%arg0: tensor<4xf64>, %arg2: tensor<i1>) -> tensor<f64> {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = tensor.empty() : tensor<f64>
  %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<f64>) -> tensor<f64>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%arg2, %arg0 : tensor<i1>, tensor<4xf64>) outs(%1 : tensor<f64>) {
  ^bb0(%arg3: i1, %arg4: f64, %arg5: f64):
    %3 = arith.cmpf une, %arg4, %arg4 : f64
    %4 = arith.select %3, %cst, %arg4 : f64
    %5 = arith.select %arg3, %4, %cst : f64
    %6 = arith.addf %arg5, %5 : f64
    linalg.yield %6 : f64
  } -> tensor<f64>
  return %2 : tensor<f64>
}

// CHECK-LABEL: func @extra_argument

// -----

func.func @scatter_f32_with_update_computation(%indices: tensor<1x2xindex>,
    %updates: tensor<1x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = thlo.scatter ins(%indices: tensor<1x2xindex>, %updates: tensor<1x?x?xf32>)
                    outs(%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %1 = arith.addf %in, %out: f32
      thlo.yield %1: f32
    }
  return %0: tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scatter_f32_with_update_computation(
// CHECK-SAME:      %[[INDICES:.*]]: tensor<1x2xindex>,
// CHECK-SAME:      %[[UPDATES:.*]]: tensor<1x?x?xf32>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2

// CHECK-DAG:  %[[UPDATES_DIM_1:.*]] = tensor.dim %[[UPDATES]], %[[C1]]
// CHECK-DAG:  %[[UPDATES_DIM_2:.*]] = tensor.dim %[[UPDATES]], %[[C2]]
// CHECK-DAG:  %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-DAG:  %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]

// Extract scatter indices from `indices` arg.
// CHECK-DAG:  %[[INDEX_0:.*]] = tensor.extract %[[INDICES]][%[[C0]],
// CHECK-DAG:  %[[INDEX_1:.*]] = tensor.extract %[[INDICES]][%[[C0]],

// CHECK-COUNT-7: arith.andi

// CHECK:      scf.if
// CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract_slice %[[INIT]][%[[INDEX_0]],
// CHECK-SAME:   %[[INDEX_1]]] [%[[UPDATES_DIM_1]], %[[UPDATES_DIM_2]]] [%[[C1]],
// CHECK-SAME:   %[[C1]]] : tensor<?x?xf32> to tensor<?x?xf32>

// CHECK-NEXT: %[[UPDATES_SLICE:.*]] = tensor.extract_slice %[[UPDATES]]

// CHECK-NEXT: %[[SUM:.*]] = linalg.reduce
// CHECK-SAME:   ins(%[[UPDATES_SLICE]] : tensor<1x?x?xf32>)
// CHECK-SAME:   outs(%[[EXTRACTED]] : tensor<?x?xf32>) dimensions = [0]
// CHECK-NEXT:   (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32) {
// CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
// CHECK-NEXT:     linalg.yield %[[ADD]] : f32
// CHECK-NEXT:   }

// CHECK-NEXT: %[[INSERTED:.*]] = tensor.insert_slice %[[SUM]] into %[[INIT]]
// CHECK-SAME:   [%[[INDEX_0]], %[[INDEX_1]]] [%[[UPDATES_DIM_1]],
// CHECK-SAME:   %[[UPDATES_DIM_2]]] [%[[C1]], %[[C1]]] : tensor<?x?xf32>
// CHECK-SAME:   into tensor<?x?xf32>
// CHECK-NEXT:       scf.yield %[[INSERTED]] : tensor<?x?xf32>
// CHECK-NEXT:   } else {
// CHECK-NEXT:       scf.yield %[[INIT]] : tensor<?x?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return

// -----

func.func @scatter_i64_no_update_computation(%indices: tensor<1x1xindex>,
                           %updates: tensor<1x1x3x4xi64>,
                           %init: tensor<3x3x4xi64>) -> tensor<3x3x4xi64> {
 %0 = thlo.scatter ins(%indices : tensor<1x1xindex>,
                       %updates : tensor<1x1x3x4xi64>)
                   outs(%init : tensor<3x3x4xi64>)
   (%arg5: i64, %arg6: i64) {
     thlo.yield %arg5 : i64
 }
 func.return %0 : tensor<3x3x4xi64>
}
// CHECK-LABEL:   func.func @scatter_i64_no_update_computation(
// CHECK-SAME:         %[[INDICES:.*]]: tensor<1x1xindex>,
// CHECK-SAME:         %[[UPDATES:.*]]: tensor<1x1x3x4xi64>,
// CHECK-SAME:         %[[INIT:.*]]: tensor<3x3x4xi64>) -> tensor<3x3x4xi64> {

// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C3:.*]] = arith.constant 3 : index

// CHECK:           %[[INDEX_0:.*]] = tensor.extract %[[INDICES]]{{\[}}%[[C0]],
// CHECK-SAME:      %[[C0]]] : tensor<1x1xindex>

// CHECK:         scf.if
// CHECK-NEXT:      %[[COLLAPSED:.*]] = tensor.collapse_shape %[[UPDATES]]
// CHECK-SAME:        [0, 1], [2], [3]]
// CHECK-SAME:        : tensor<1x1x3x4xi64> into tensor<1x3x4xi64>
// CHECK-NEXT:      %[[INSERTED:.*]] = tensor.insert_slice %[[COLLAPSED]] into
// CHECK-SAME:        %[[INIT]][%[[INDEX_0]], %[[C0]], %[[C0]]] [1, 3, 4]
// CHECK-SAME:        [%[[C1]], %[[C1]], %[[C1]]]
// CHECK-SAME:        : tensor<1x3x4xi64> into tensor<3x3x4xi64>
// CHECK-NEXT:      scf.yield %[[INSERTED]] : tensor<3x3x4xi64>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[INIT]] : tensor<3x3x4xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// -----

func.func @gather(%indices: tensor<1x2xindex>,
                  %operand: tensor<5x6x7xi64>,
                  %init: tensor<1x3xi64>) -> tensor<1x3xi64> {
 %0 = thlo.gather ins(%operand : tensor<5x6x7xi64>,
                      %indices : tensor<1x2xindex>)
                   outs(%init : tensor<1x3xi64>)
 func.return %0 : tensor<1x3xi64>
}

// CHECK-LABEL: func.func @gather(
//  CHECK-SAME:     %[[INDICES:.*]]: tensor<1x2xindex>
//  CHECK-SAME:     %[[OPERAND:.*]]: tensor<5x6x7xi64>
//  CHECK-SAME:     %[[INIT:.*]]: tensor<1x3xi64>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5
//   CHECK-DAG:   %[[INDEX0:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C0]]]
//   CHECK-DAG:   %[[INDEX1:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C1]]]
//   CHECK-DAG:   %[[CLAMPED_INDEX0:.*]] = arith.minsi %[[INDEX0]], %[[C2]]
//   CHECK-DAG:   %[[CLAMPED_INDEX0_:.*]] = arith.maxsi %[[CLAMPED_INDEX0]], %[[C0]]
//   CHECK-DAG:   %[[CLAMPED_INDEX1:.*]] = arith.minsi %[[INDEX1]], %[[C5]]
//   CHECK-DAG:   %[[CLAMPED_INDEX1_:.*]] = arith.maxsi %[[CLAMPED_INDEX1]], %[[C0]]
//       CHECK:    scf.for %[[J:.*]] = %[[C0]] to %[[C3]]
//       CHECK:      %[[OFFSET_J:.*]] = arith.addi %[[J]], %[[CLAMPED_INDEX0_]]

//       CHECK:      %[[VAL:.*]] = tensor.extract %[[OPERAND]]
//       CHECK-SAME:   [%[[OFFSET_J]], %[[CLAMPED_INDEX1_]], %[[C0]]]
//       CHECK-NEXT: %[[UPDATED:.*]] = tensor.insert %[[VAL]]
//       CHECK:      scf.yield %[[UPDATED]]

// -----

func.func @fold_from_elements_into_insert_slice(%elem: f32,
    %out: tensor<8x2xf32>) -> tensor<8x2xf32>  {
  %elem_tensor = tensor.from_elements %elem : tensor<1x1xf32>
  %updated = tensor.insert_slice %elem_tensor into %out[0, 1] [1, 1] [1, 1]
    : tensor<1x1xf32> into tensor<8x2xf32>

  func.return %updated: tensor<8x2xf32>
}
// CHECK-LABEL: func @fold_from_elements_into_insert_slice
// CHECK-SAME:      %[[ELEM:.*]]: f32, %[[OUT:.*]]: tensor<8x2xf32>

// CHECK:         %[[UPDATE:.*]] = tensor.insert %[[ELEM]] into %[[OUT]]
// CHECK-NEXT:    return %[[UPDATE]]

// -----

func.func @dynamic_broadcast_in_dim(%arg : tensor<1x1xf32>,
                                    %init: tensor<1x1x1xf32>)
                                    -> tensor<1x1x1xf32>  {
  %0 = thlo.dynamic_broadcast_in_dim ins(%arg : tensor<1x1xf32>)
                                     outs(%init : tensor<1x1x1xf32>)
                                     broadcast_dimensions = [0, 2]
  func.return %0 : tensor<1x1x1xf32>
}
// CHECK-LABEL: @dynamic_broadcast_in_dim(
// CHECK-SAME:      %[[ARG:.*]]: tensor<1x1xf32>, %[[INIT:.*]]: tensor<1x1x1xf32>)
// CHECK:                 %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[ELEM:.*]] = tensor.extract %[[ARG]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[UPDATED:.*]] = tensor.from_elements %[[ELEM]]

// -----

func.func @concatenate(
  %arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>,
  %arg2: tensor<?x?x?xf32>, %init: tensor<?x1x?xf32>) -> tensor<?x1x?xf32> {
  %cat = thlo.concatenate
    ins(%arg0: tensor<?x?x?xf32>,
        %arg1: tensor<?x?x?xf32>,
        %arg2: tensor<?x?x?xf32>)
    outs(%init: tensor<?x1x?xf32>)
    dimension = 1
  func.return %cat : tensor<?x1x?xf32>
}

// CHECK-LABEL: func @concatenate(
// CHECK-SAME:      %[[ARG_0:[0-9a-zA-Z]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:      %[[ARG_1:[0-9a-zA-Z]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:      %[[ARG_2:[0-9a-zA-Z]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:      %[[INIT:[0-9a-zA-Z]*]]: tensor<?x1x?xf32>)

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2

// CHECK-DAG:   %[[DIM0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-DAG:   %[[DIM2:.*]] = tensor.dim %[[INIT]], %[[C2]]


// Extract elements from arg0 is it's not empty.
// CHECK-NEXT:  %[[DIM_ARG_0:.*]] = tensor.dim %[[ARG_0]], %[[C1]]
// CHECK-NEXT:  %[[CMP_0:.*]] = arith.cmpi ne, %[[DIM_ARG_0]], %[[C0]]
// CHECK:       %[[RESULT:.*]] = scf.if %[[CMP_0]]
// CHECK:         %[[MAT_0:.*]] = tensor.extract_slice %[[ARG_0]]
// CHECK-SAME:        [0, 0, 0] [%[[DIM0]], 1, %[[DIM2]]] [1, 1, 1]
// CHECK:         %[[RES_0:.*]] = tensor.insert_slice %[[MAT_0]] into %[[INIT]]
// CHECK-NEXT:    scf.yield %[[RES_0]]
// CHECK-NEXT:  } else {

// Else check arg1 and extracts element if it's not empty.
// CHECK-NEXT:    %[[DIM_ARG_1:.*]] = tensor.dim %[[ARG_1]], %[[C1]]
// CHECK-NEXT:    %[[CMP_1:.*]] = arith.cmpi ne, %[[DIM_ARG_1]], %[[C0]]
// CHECK-NEXT:    %[[RESULT_1:.*]] = scf.if %[[CMP_1]]
// CHECK-NEXT:      %[[MAT_1:.*]] = tensor.extract_slice %[[ARG_1]]
// CHECK-SAME:        [0, 0, 0] [%[[DIM0]], 1, %[[DIM2]]] [1, 1, 1]
// CHECK-NEXT:      %[[RES_1:.*]] = tensor.insert_slice %[[MAT_1]] into %[[INIT]]
// CHECK-NEXT:      scf.yield %[[RES_1]]
// CHECK-NEXT:    } else {

// Otherwise extract elements from arg2, because arg0 and arg1 are empty.
// CHECK-NEXT:      %[[MAT_2:.*]] = tensor.extract_slice %[[ARG_2]]
// CHECK-SAME:        [0, 0, 0] [%[[DIM0]], 1, %[[DIM2]]] [1, 1, 1]
// CHECK-NEXT:      %[[RES_2:.*]] = tensor.insert_slice %[[MAT_2]] into %[[INIT]]
// CHECK-NEXT:      scf.yield %[[RES_2]]
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.yield %[[RESULT_1]]
// CHECK-NEXT:  }

// CHECK-NEXT:  return %[[RESULT]] : tensor<?x1x?xf32>

// -----

func.func @linalg_map(%lhs : tensor<1x1xf32>,
                      %rhs: tensor<1x1xf32>,
                      %init: tensor<1x1xf32>)
                                    -> tensor<1x1xf32>  {
      %add = linalg.map
          ins(%lhs, %rhs : tensor<1x1xf32>, tensor<1x1xf32>)
          outs(%init: tensor<1x1xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
      func.return %add : tensor<1x1xf32>
}

// CHECK-LABEL: @linalg_map(
// CHECK-SAME:      %[[LHS:.*]]: tensor<1x1xf32>, %[[RHS:.*]]: tensor<1x1xf32>, %[[INIT:.*]]: tensor<1x1xf32>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[L_ELEM:.*]] = tensor.extract %[[LHS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[R_ELEM:.*]] = tensor.extract %[[RHS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[L_ELEM]], %[[R_ELEM]]
// CHECK-NEXT:      tensor.from_elements %[[ADD]]

// -----

func.func @linalg_reduce(%ins: tensor<1x1x1xf32>,
                         %outs: tensor<1x1xf32>)
                                    -> tensor<1x1xf32>  {
      %reduce = linalg.reduce
          ins(%ins: tensor<1x1x1xf32>)
          outs(%outs: tensor<1x1xf32>)
          dimensions = [1]
          (%in: f32, %out: f32) {
            %0 = arith.addf %in, %out: f32
            linalg.yield %0: f32
          }
      func.return %reduce : tensor<1x1xf32>
}

// CHECK-LABEL: @linalg_reduce(
// CHECK-SAME:      %[[INS:.*]]: tensor<1x1x1xf32>, %[[OUTS:.*]]: tensor<1x1xf32>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[L_ELEM:.*]] = tensor.extract %[[INS]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[R_ELEM:.*]] = tensor.extract %[[OUTS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[L_ELEM]], %[[R_ELEM]]
// CHECK-NEXT:      tensor.from_elements %[[ADD]]

// -----

func.func @linalg_transpose(%ins: tensor<1x1xf32>,
                            %outs: tensor<1x1xf32>)
                                    -> tensor<1x1xf32>  {
      %transpose = linalg.transpose
          ins(%ins: tensor<1x1xf32>)
          outs(%outs: tensor<1x1xf32>)
          permutation = [1, 0]
      func.return %transpose : tensor<1x1xf32>
}

// CHECK-LABEL: @linalg_transpose(
// CHECK-SAME:      %[[INS:.*]]: tensor<1x1xf32>, %[[OUTS:.*]]: tensor<1x1xf32>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[EXTRACTED:.*]] = tensor.extract %[[INS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      tensor.from_elements %[[EXTRACTED]]

// -----

func.func @linalg_matmul(%lhs: tensor<1x1xf32>,
                         %rhs: tensor<1x1xf32>,
                         %out : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<1x1xf32>, tensor<1x1xf32>)
      outs(%out : tensor<1x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: @linalg_matmul(
// CHECK-SAME:      %[[LHS:.*]]: tensor<1x1xf32>, %[[RHS:.*]]: tensor<1x1xf32>, %[[OUT:.*]]: tensor<1x1xf32>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[LHS_ELEM:.*]] = tensor.extract %[[LHS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[RHS_ELEM:.*]] = tensor.extract %[[RHS]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[OUT_ELEM:.*]] = tensor.extract %[[OUT]][%[[C0]], %[[C0]]]
// CHECK-NEXT:      %[[MUL:.*]] = arith.mulf %[[LHS_ELEM]], %[[RHS_ELEM]]
// CHECK-NEXT:      %[[ADD:.*]] = arith.addf %[[OUT_ELEM]], %[[MUL]]
// CHECK-NEXT:       tensor.from_elements %[[ADD]]

// -----

func.func @thlo_reverse(%arg : tensor<1x1xf32>, %init: tensor<1x1xf32>)
    -> tensor<1x1xf32> {
  %0 = thlo.reverse ins(%arg : tensor<1x1xf32>)
        outs(%init : tensor<1x1xf32>)
        reverse_dimensions = [0, 1]
  func.return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: @thlo_reverse(
//  CHECK-SAME: %[[ARG:.*]]: tensor<1x1xf32>, %[[INIT:.*]]: tensor<1x1xf32>)
//       CHECK:   return %[[ARG]]

// -----

func.func @ite_1d(%arg0: i1, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>)
    -> tensor<1xf32> {
  %0 = scf.if %arg0 -> (tensor<1xf32>) {
    scf.yield %arg2 : tensor<1xf32>
  } else {
    scf.yield %arg1 : tensor<1xf32>
  }
  return %0 : tensor<1xf32>
}

// CHECK:     func.func @ite_1d(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<1xf32>, %[[ARG2:.*]]: tensor<1xf32>)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[IF:.*]] = scf.if %[[ARG0]] -> (f32)
// CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[ARG2]][%[[C0]]]
// CHECK:         scf.yield %[[EXTRACTED]] : f32
// CHECK:       else
// CHECK:         %[[EXTRACTED_0:.*]] = tensor.extract %[[ARG1]][%[[C0]]]
// CHECK:         scf.yield %[[EXTRACTED_0]] : f32
// CHECK:       %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[IF]]
// CHECK:       return %[[FROM_ELEMENTS]]

// -----

func.func @ite_2d(%arg0: i1, %arg1: tensor<1x1xf32>, %arg2: tensor<1x1xf32>)
    -> tensor<1x1xf32> {
  %0 = scf.if %arg0 -> (tensor<1x1xf32>) {
    scf.yield %arg2 : tensor<1x1xf32>
  } else {
    scf.yield %arg1 : tensor<1x1xf32>
  }
  return %0 : tensor<1x1xf32>
}

// CHECK:     func.func @ite_2d(%[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<1x1xf32>, %[[ARG2:.*]]: tensor<1x1xf32>)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[IF:.*]] = scf.if %[[ARG0]] -> (f32)
// CHECK:         %[[EXTRACTED:.*]] = tensor.extract %[[ARG2]][%[[C0]], %[[C0]]]
// CHECK:         scf.yield %[[EXTRACTED]] : f32
// CHECK:       else
// CHECK:         %[[EXTRACTED_0:.*]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]]]
// CHECK:         scf.yield %[[EXTRACTED_0]] : f32
// CHECK:       %[[FROM_ELEMENTS:.*]] = tensor.from_elements %[[IF]]
// CHECK:       return %[[FROM_ELEMENTS]]


// -----

func.func @scalarize_for_op(%initValue: f32, %input: tensor<10xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %initTensor = tensor.from_elements %initValue : tensor<1x1xf32>

  %sum = scf.for %i = %c0 to %c10 step %c1
      iter_args(%acc = %initTensor) -> (tensor<1x1xf32>) {
    %input_elem = tensor.extract %input[%i] : tensor<10xf32>

    %acc_elem = tensor.extract %acc[%c0, %c0] : tensor<1x1xf32>
    %add = arith.addf %acc_elem, %input_elem : f32
    %from_elements = tensor.from_elements %add : tensor<1x1xf32>

    scf.yield %from_elements : tensor<1x1xf32>
  }
  %sum_elem = tensor.extract %sum[%c0, %c0] : tensor<1x1xf32>
  func.return %sum_elem : f32
}
// CHECK-LABEL: @scalarize_for_op

// CHECK:      scf.for %[[I:[a-z0-9]+]] =
// CHECK-NEXT:   %[[ELEM:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<10xf32>
// CHECK-NEXT:   %[[ADD:.*]] = arith.addf %{{.*}}, %[[ELEM]] : f32
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return


