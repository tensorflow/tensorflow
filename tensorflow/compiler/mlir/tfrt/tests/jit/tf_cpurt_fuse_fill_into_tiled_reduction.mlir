// RUN: tf-tfrt-opt -tf-cpurt-fuse-fill-into-tiled-reduction %s \
// RUN: --split-input-file |\
// RUN: FileCheck %s

#map0 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module  {
  func @reduce_row_sum_2d(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %1 = linalg.init_tensor [%0] : tensor<?xf32>
    %fill = linalg.fill(%cst, %1) : f32, tensor<?xf32> -> tensor<?xf32>
    %3 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %4 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %5 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%3, %4) step (%c4, %c4)
        ins (%lhs_ = %lhs: tensor<?x?xf32>, %rhs_ = %rhs: tensor<?x?xf32>)
        outs (%fill_ = %fill: tensor<?xf32>)
        iterators["parallel", "reduction"] {
      %6 = affine.min #map0(%i)[%3]
      %7 = affine.min #map0(%j)[%4]
      %8 = tensor.extract_slice %lhs_[%i, %j] [%6, %7] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
      %9 = affine.min #map0(%i)[%3]
      %10 = affine.min #map0(%j)[%4]
      %11 = tensor.extract_slice %rhs_[%i, %j] [%9, %10] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
      %12 = affine.min #map0(%i)[%3]
      %13 = tensor.extract_slice %fill_[%i] [%12] [1]
        : tensor<?xf32> to tensor<?xf32>
      %14 = linalg.generic {
              indexing_maps = [#map1, #map1, #map2],
              iterator_types = ["parallel", "reduction"]}
              ins(%8, %11 : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%13 : tensor<?xf32>) {
      ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
        %16 = arith.mulf %arg7, %arg8 : f32
        %17 = arith.addf %16, %arg9 : f32
        linalg.yield %17 : f32
      } -> tensor<?xf32>
      %15 = tensor.insert_slice %14 into %fill_[%i] [%12] [1]
        : tensor<?xf32> into tensor<?xf32>
      linalg.yield %15 : tensor<?xf32>
    }
    return %5 : tensor<?xf32>
  }
}
// CHECK-LABEL: func @reduce_row_sum_2d(
// CHECK-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?xf32>

// CHECK-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index

// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D:.*]]
// CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[CLONE:.*]] = linalg.init_tensor [%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[FILL:.*]] = linalg.fill(%[[C0_F32]], %[[INIT]])
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]

// CHECK:      linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// CHECK-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_1D]],
// CHECK-SAME:         %[[CLONE_:.*]] = %[[CLONE]]: [[TY_1D]])

// CHECK:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]], %[[J]]]
// CHECK:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][%[[I]], %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]]]
// CHECK:      %[[CLONE_SUB:.*]] = tensor.extract_slice %[[CLONE_]][%[[I]]]

// CHECK:      %[[FILL_SUB:.*]] = linalg.fill(%[[C0_F32]], %[[CLONE_SUB]])

// CHECK:      %[[SUM_OF_PROD_SUB:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]] : [[TY_2D]], [[TY_2D]])
// CHECK-SAME:   outs(%[[FILL_SUB]] : [[TY_1D]])
// CHECK:          mulf
// CHECK:          addf
// CHECK-NEXT:     linalg.yield

// CHECK:      %[[ACC:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[SUM_OF_PROD_SUB]] : [[TY_1D]])
// CHECK-SAME:   outs(%[[OUT_SUB]] : [[TY_1D]]) {
// CHECK-NOT:      mulf
// CHECK:          addf
// CHECK-NEXT:     linalg.yield

// CHECK:      %[[CLONE_UPDATE:.*]] = tensor.insert_slice %[[SUM_SUB:.*]] into %[[CLONE_]]
// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[ACC:.*]] into %[[OUT_]]
// CHECK:      linalg.yield %[[UPDATE]], %[[CLONE_UPDATE]] : [[TY_1D]], [[TY_1D]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module  {
  func @reduce_row_sum_2d_static(%in: tensor<8x16xf32>) -> tensor<8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = linalg.init_tensor [8] : tensor<8xf32>
    %fill = linalg.fill(%cst, %0) : f32, tensor<8xf32> -> tensor<8xf32>
    %2 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4)
           ins (%in_ = %in: tensor<8x16xf32>)
           outs (%fill_ = %fill: tensor<8xf32>)
           iterators["parallel", "reduction"] {
      %3 = tensor.extract_slice %in_[%i, %j] [4, 4] [1, 1]
        : tensor<8x16xf32> to tensor<4x4xf32>
      %4 = tensor.extract_slice %fill_[%i] [4] [1]
        : tensor<8xf32> to tensor<4xf32>
      %5 = linalg.generic {
             indexing_maps = [#map0, #map1],
             iterator_types = ["parallel", "reduction"]}
             ins(%3 : tensor<4x4xf32>)
             outs(%4 : tensor<4xf32>) {
      ^bb0(%arg5: f32, %arg6: f32):
        %7 = arith.addf %arg5, %arg6 : f32
        linalg.yield %7 : f32
      } -> tensor<4xf32>
      %6 = tensor.insert_slice %5 into %fill_[%i] [4] [1]
        : tensor<4xf32> into tensor<8xf32>
      linalg.yield %6 : tensor<8xf32>
    }
    return %2 : tensor<8xf32>
  }
}
// CHECK-LABEL: func @reduce_row_sum_2d_static
// CHECK: linalg.tiled_loop
// CHECK:   tensor.insert_slice
// CHECK:   tensor.insert_slice

// -----

#map0 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module  {
  func @reduce_column_sum_2d(%in: tensor<?x?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.dim %in, %c0 : tensor<?x?xf32>
    %1 = linalg.init_tensor [%0] : tensor<?xf32>
    %fill = linalg.fill(%cst, %1) : f32, tensor<?xf32> -> tensor<?xf32>
    %3 = tensor.dim %in, %c0 : tensor<?x?xf32>
    %4 = tensor.dim %in, %c1 : tensor<?x?xf32>
    %5 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%3, %4) step (%c4, %c4)
        ins (%in_ = %in: tensor<?x?xf32>)
        outs (%fill_ = %fill: tensor<?xf32>)
        iterators["reduction", "parallel"] {
      %6 = affine.min #map0(%i)[%3]
      %7 = affine.min #map0(%j)[%4]
      %8 = tensor.extract_slice %in_[%i, %j] [%6, %7] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
      %9 = affine.min #map0(%j)[%4]
      %10 = tensor.extract_slice %fill_[%j] [%9] [1]
        : tensor<?xf32> to tensor<?xf32>
      %11 = linalg.generic {
          indexing_maps = [#map1, #map2],
          iterator_types = ["reduction", "parallel"]}
          ins(%8 : tensor<?x?xf32>)
          outs(%10 : tensor<?xf32>) {
      ^bb0(%arg5: f32, %arg6: f32):
        %13 = arith.addf %arg5, %arg6 : f32
        linalg.yield %13 : f32
      } -> tensor<?xf32>
      %12 = tensor.insert_slice %11 into %fill_[%j] [%9] [1]
        : tensor<?xf32> into tensor<?xf32>
      linalg.yield %12 : tensor<?xf32>
    }
    return %5 : tensor<?xf32>
  }
}
// CHECK-LABEL: func @reduce_column_sum_2d
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xf32>) -> tensor<?xf32>

// CHECK-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index

// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D:.*]]
// CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[CLONE:.*]] = linalg.init_tensor [%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[FILL:.*]] = linalg.fill(%[[C0_F32]], %[[INIT]])
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : [[TY_2D]]

// CHECK:      linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[IN_:.*]] = %[[INPUT]]: [[TY_2D]])
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_1D]],
// CHECK-SAME:         %[[CLONE_:.*]] = %[[CLONE]]: [[TY_1D]])

// CHECK:      %[[IN_SUB:.*]] = tensor.extract_slice %[[IN_]][%[[I]], %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[J]]]
// CHECK:      %[[CLONE_SUB:.*]] = tensor.extract_slice %[[CLONE_]][%[[J]]]

// CHECK:      %[[FILL_SUB:.*]] = linalg.fill(%[[C0_F32]], %[[CLONE_SUB]])

// CHECK:      %[[SUM_SUB:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[IN_SUB]] : [[TY_2D]])
// CHECK-SAME:   outs(%[[FILL_SUB]] : [[TY_1D]])
// CHECK:          addf
// CHECK-NEXT:     linalg.yield


// CHECK:      %[[ACC:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[SUM_SUB]] : [[TY_1D]])
// CHECK-SAME:   outs(%[[OUT_SUB]] : [[TY_1D]]) {
// CHECK:          addf
// CHECK-NEXT:     linalg.yield

// CHECK:      %[[CLONE_UPDATE:.*]] = tensor.insert_slice %[[SUM_SUB:.*]] into %[[CLONE_]]
// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[ACC:.*]] into %[[OUT_]]
// CHECK:      linalg.yield %[[UPDATE]], %[[CLONE_UPDATE]] : [[TY_1D]], [[TY_1D]]
