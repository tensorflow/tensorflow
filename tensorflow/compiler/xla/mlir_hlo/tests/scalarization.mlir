// RUN: mlir-hlo-opt %s --scalarize --split-input-file | FileCheck %s

#map = affine_map<() -> ()>

func.func @zero_rank(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32>  {
  %0 = linalg.init_tensor []: tensor<f32>
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


func.func @nonzero_rank(%lhs: tensor<1xf32>, %rhs: tensor<1x1xf32>)
    -> tensor<1x1x1xf32>  {
  %0 = linalg.init_tensor [1, 1, 1]: tensor<1x1x1xf32>
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
  %0 = linalg.init_tensor []: tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %rhs: tensor<f32>, tensor<f32>)
    outs(%0: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %2 = arith.addf %arg3, %arg4: f32
    linalg.yield %2: f32
  } -> tensor<f32>

  %3 = linalg.init_tensor []: tensor<f32>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []}
    ins(%lhs, %1: tensor<f32>, tensor<f32>)
    outs(%3: tensor<f32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %5 = arith.mulf %arg3, %arg4: f32
    linalg.yield %5: f32
  } -> tensor<f32>

  %6 = linalg.init_tensor []: tensor<f32>
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
  %0 = linalg.init_tensor []: tensor<f32>
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
  %0 = linalg.init_tensor [1, 1]: tensor<1x1xi1>
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
  %0 = linalg.init_tensor [] : tensor<f64>
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

func.func @scatter_i32_i64(%indices: tensor<1x2xi32>, %updates: tensor<1xi64>,
                           %init: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = thlo.scatter ins(%indices: tensor<1x2xi32>, %updates: tensor<1xi64>)
                    outs(%init: tensor<?x?xi64>)
    (%in: i64, %out: i64) {
      %1 = arith.addi %in, %out: i64
      thlo.yield %1: i64
    }
  return %0: tensor<?x?xi64>
}
// CHECK-LABEL: func @scatter_i32_i64(
// CHECK-SAME:      %[[INDICES:.*]]: tensor<1x2xi32>,
// CHECK-SAME:      %[[UPDATES:.*]]: tensor<1xi64>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<?x?xi64>) -> tensor<?x?xi64> {

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index

// CHECK:       %[[UPD_ELEM:.*]] = tensor.extract %[[UPDATES]][%[[C0]]]

// CHECK:       %[[I0_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C0]]]
// CHECK:       %[[I0:.*]] = arith.index_cast %[[I0_INT]] : i32 to index

// CHECK:       %[[I1_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C1]]]
// CHECK:       %[[I1:.*]] = arith.index_cast %[[I1_INT]] : i32 to index

// CHECK:         %[[RESULT:.*]] = scf.if
// CHECK:           %[[CUR_ELEM:.*]] = tensor.extract %[[INIT]][%[[I0]], %[[I1]]]
// CHECK:           %[[COMBINED:.*]] = arith.addi %[[UPD_ELEM]], %[[CUR_ELEM]]
// CHECK:           %[[UPDATED_INIT:.*]] = tensor.insert %[[COMBINED]] into %[[INIT]]
// CHECK:         } else {
// CHECK:           scf.yield %[[INIT]]
// CHECK:         }
// CHECK:         return %[[RESULT]]

// -----

func.func @scatter_i32_f32(%indices: tensor<1x2xi32>, %updates: tensor<1xf32>,
                           %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = thlo.scatter ins(%indices: tensor<1x2xi32>, %updates: tensor<1xf32>)
                    outs(%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %1 = arith.addf %in, %out: f32
      thlo.yield %1: f32
    }
  return %0: tensor<?x?xf32>
}
// CHECK-LABEL: func @scatter_i32_f32(
// CHECK-SAME:      %[[INDICES:.*]]: tensor<1x2xi32>,
// CHECK-SAME:      %[[UPDATES:.*]]: tensor<1xf32>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index

// CHECK:         %[[UPD_ELEM:.*]] = tensor.extract %[[UPDATES]][%[[C0]]]

// CHECK:         %[[I0_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C0]]]
// CHECK:         %[[I0:.*]] = arith.index_cast %[[I0_INT]] : i32 to index

// CHECK:         %[[I1_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C1]]]
// CHECK:         %[[I1:.*]] = arith.index_cast %[[I1_INT]] : i32 to index

// CHECK:         %[[RESULT:.*]] = scf.if
// CHECK:           %[[CUR_ELEM:.*]] = tensor.extract %[[INIT]][%[[I0]], %[[I1]]]
// CHECK:           %[[COMBINED:.*]] = arith.addf %[[UPD_ELEM]], %[[CUR_ELEM]]
// CHECK:           %[[UPDATED_INIT:.*]] = tensor.insert %[[COMBINED]] into %[[INIT]]
// CHECK:         } else {
// CHECK:           scf.yield %[[INIT]]
// CHECK:         }
// CHECK:         return %[[RESULT]]

// -----

func.func @scatter_2d_indices(%indices: tensor<1xi32>, %updates: tensor<f32>,
                              %init: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = thlo.scatter ins(%indices: tensor<1xi32>, %updates: tensor<f32>)
                    outs(%init: tensor<?x?x?xf32>)
    (%in: f32, %out: f32) {
      %1 = arith.addf %in, %out: f32
      thlo.yield %1: f32
    }
  return %0: tensor<?x?x?xf32>
}
// CHECK-LABEL:   func @scatter_2d_indices(
// CHECK-SAME:        %[[INDICES:.*]]: tensor<1xi32>,
// CHECK-SAME:        %[[UPDATES:.*]]: tensor<f32>,
// CHECK-SAME:        %[[INIT:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index

// CHECK:         %[[UPD_ELEM:.*]] = tensor.extract %[[UPDATES]][]

// CHECK:         %[[I0_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]]]
// CHECK:         %[[I0:.*]] = arith.index_cast %[[I0_INT]] : i32 to index

// CHECK:         %[[I1_INT:.*]] = tensor.extract %[[INDICES]][%[[C1]]]
// CHECK:         %[[I1:.*]] = arith.index_cast %[[I1_INT]] : i32 to index

// CHECK:         %[[I2_INT:.*]] = tensor.extract %[[INDICES]][%[[C2]]]
// CHECK:         %[[I2:.*]] = arith.index_cast %[[I2_INT]] : i32 to index

// CHECK:         %[[RESULT:.*]] = scf.if
// CHECK:           %[[CUR_ELEM:.*]] = tensor.extract %[[INIT]][%[[I0]], %[[I1]], %[[I2]]]
// CHECK:           %[[COMBINED:.*]] = arith.addf %[[UPD_ELEM]], %[[CUR_ELEM]]
// CHECK:           %[[UPDATED_INIT:.*]] = tensor.insert %[[COMBINED]] into %[[INIT]]
// CHECK:         } else {
// CHECK:           scf.yield %[[INIT]]
// CHECK:         }
// CHECK:         return %[[RESULT]]

// -----

func.func @scatter_small_vector_dim(%indices: tensor<1x1x2xi32>,
    %updates: tensor<1x1xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = thlo.scatter ins(%indices: tensor<1x1x2xi32>, %updates: tensor<1x1xf32>)
                    outs(%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %1 = arith.maxf %in, %out: f32
      thlo.yield %1: f32
    }
  return %0: tensor<?x?xf32>
}
// CHECK-LABEL: func @scatter_small_vector_dim(
// CHECK-SAME:      %[[INDICES:.*]]: tensor<1x1x2xi32>,
// CHECK-SAME:      %[[UPDATES:.*]]: tensor<1x1xf32>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {

// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index

// CHECK:         %[[UPD_ELEM:.*]] = tensor.extract %[[UPDATES]][%[[C0]], %[[C0]]]

// CHECK:         %[[I0_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:         %[[I0:.*]] = arith.index_cast %[[I0_INT]] : i32 to index

// CHECK:         %[[I1_INT:.*]] = tensor.extract %[[INDICES]][%[[C0]], %[[C0]], %[[C1]]]
// CHECK:         %[[I1:.*]] = arith.index_cast %[[I1_INT]] : i32 to index

// CHECK:         %[[RESULT:.*]] = scf.if
// CHECK:           %[[CUR_ELEM:.*]] = tensor.extract %[[INIT]][%[[I0]], %[[I1]]]
// CHECK:           %[[COMBINED:.*]] = arith.maxf %[[UPD_ELEM]], %[[CUR_ELEM]]
// CHECK:           %[[UPDATED_INIT:.*]] = tensor.insert %[[COMBINED]] into %[[INIT]]
// CHECK:         } else {
// CHECK:           scf.yield %[[INIT]]
// CHECK:         }
// CHECK:         return %[[RESULT]]

// -----

func.func @fold_extract_from_elements_into_gml_st(%in: tensor<8x2xf32>,
    %out: tensor<8x2xf32>) -> tensor<8x2xf32>  {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %space = gml_st.space [8, 2] : !gml_st.tile<8x2>
  %copy = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c8, %c2) step (%c1, %c1) {
    %tile = gml_st.tile %space [%i, %j] [1, 1] [1, 1]
      : !gml_st.tile<8x2> to !gml_st.tile<1x1>

    %in_sub = gml_st.materialize %in[%tile]
      : tensor<8x2xf32>[!gml_st.tile<1x1>] to tensor<1x1xf32>

    %elem = tensor.extract %in_sub[%c0, %c0] : tensor<1x1xf32>

    %out_sub = tensor.from_elements %elem : tensor<1x1xf32>

    gml_st.set_yield %out_sub into %out[%tile]
      : tensor<1x1xf32> into tensor<8x2xf32>[!gml_st.tile<1x1>]
  } : tensor<8x2xf32>
  func.return %copy: tensor<8x2xf32>
}
// CHECK-LABEL: func @fold_extract_from_elements_into_gml_st

// CHECK:       = gml_st.tile
// CHECK-NEXT:  %[[ELEM:.*]] = gml_st.materialize
// CHECK-SAME:    : tensor<8x2xf32>[!gml_st.tile<1x1>] to f32

// CHECK-NEXT:  gml_st.set_yield %[[ELEM]]
// CHECK-SAME:    : f32 into tensor<8x2xf32>[!gml_st.tile<1x1>]
