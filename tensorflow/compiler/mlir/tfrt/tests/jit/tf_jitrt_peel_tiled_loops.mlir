// RUN: tf-tfrt-opt %s -allow-unregistered-dialect -split-input-file \
// RUN: -tf-jitrt-peel-tiled-loops -cse -canonicalize | FileCheck %s

#map0 = affine_map<(d0) -> (8, -d0 + 102401)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @tanh_1d(%arg0: memref<102401xf32>) -> memref<102401xf32> {
  %c102401 = arith.constant 102401 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<102401xf32>
  gml_st.loop (%arg1) = (%c0) to (%c102401) step (%c8)
      ins (%arg2 = %arg0: memref<102401xf32>)
      outs (%arg3 = %0: memref<102401xf32>) {
    %1 = affine.min #map0(%arg1)
    %2 = memref.subview %arg2[%arg1] [%1] [1]
        : memref<102401xf32> to memref<?xf32, #map1>
    %3 = memref.subview %arg3[%arg1] [%1] [1]
        : memref<102401xf32> to memref<?xf32, #map1>
    %4 = vector.transfer_read %2[%c0], %cst
        : memref<?xf32, #map1>, vector<8xf32>
    %5 = math.tanh %4 : vector<8xf32>
    vector.transfer_write %5, %3[%c0] : vector<8xf32>, memref<?xf32, #map1>
    memref.copy %3, %3 : memref<?xf32, #map1> to memref<?xf32, #map1>
    gml_st.yield
  }
  func.return %0 : memref<102401xf32>
}

// CHECK-DAG:  #[[$MAP:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @tanh_1d

// CHECK:       gml_st.loop
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<8xf32, strided<[1], offset: ?>>
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<8xf32, strided<[1], offset: ?>>

// CHECK:       gml_st.loop
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<?xf32, #[[$MAP]]>
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<?xf32, #[[$MAP]]>

// -----

func.func @tanh_3d(%d0: index, %d1: index, %d2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  gml_st.loop (%arg1 ,%arg2, %arg3) = (%c0, %c0, %c0)
    to (%d0, %d1, %d2) step (%c8, %c1, %c8)
    ins () outs () {
    "prevent.dce"() : () -> ()
    gml_st.yield
  }
  func.return
}

// CHECK-LABEL: func @tanh_3d(
// CHECK-SAME:    %[[D0:[a-z0-9]+]]: index, %[[D1:[a-z0-9]+]]: index,
// CHECK-SAME:    %[[D2:[a-z0-9]+]]: index) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index

// CHECK-DAG:     %[[SPLIT0:.*]] = affine.apply{{.*}}%[[D0]]
// CHECK-DAG:     %[[SPLIT2:.*]] = affine.apply{{.*}}%[[D2]]

// CHECK:     gml_st.loop{{.*}}(%[[C0]], %[[C0]], %[[C0]])
// CHECK-SAME:  to (%[[SPLIT0]], %arg1, %[[SPLIT2]])
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// CHECK:     gml_st.loop{{.*}}(%[[C0]], %[[C0]], %[[SPLIT2]])
// CHECK-SAME:  to (%[[SPLIT0]], %arg1, %arg2)
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// CHECK:     gml_st.loop{{.*}}(%[[SPLIT0]], %[[C0]], %[[C0]])
// CHECK-SAME:  to (%arg0, %arg1, %arg2)
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// -----

func.func @reduce_column_sum_2d_dynamic(%in: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index

  %dim_X = tensor.dim %in, %c0 : tensor<?x?xf32>
  %dim_Y = tensor.dim %in, %c1 : tensor<?x?xf32>

  %1 = tensor.empty(%dim_Y) : tensor<?xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  %5 = gml_st.loop (%i, %j) = (%c0, %c0) to (%dim_Y, %dim_X)
         step (%c4, %c4)
         ins (%in_ = %in: tensor<?x?xf32>, %cst_ = %cst: f32)
         outs (%out_ = %2: tensor<?xf32>)
         iterators[#gml_st.iterator_type<parallel>,
                   #gml_st.iterator_type<reduction>] {
    %6 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%j)[%dim_X]
    %9 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%dim_Y]

    %8 = tensor.extract_slice %in_[%j, %i] [%6, %9] [1, 1]
           : tensor<?x?xf32> to tensor<?x?xf32>
    %11 = tensor.extract_slice %out_[%i] [%9] [1]
           : tensor<?xf32> to tensor<?xf32>

    %12 = linalg.fill ins(%cst_ : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
    %13 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                             affine_map<(d0, d1) -> (d0)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%8 : tensor<?x?xf32>)
            outs(%12 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %14 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%13 : tensor<?xf32>)
            outs(%11 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %15 = tensor.insert_slice %14 into %out_[%i] [%9] [1]
            : tensor<?xf32> into tensor<?xf32>
    gml_st.yield %15 : tensor<?xf32>
  }
  func.return %5 : tensor<?xf32>
}

// CHECK-LABEL: func @reduce_column_sum_2d_dynamic

// CHECK:       linalg.fill
// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<4xf32>

// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<4xf32>

// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>

// -----

func.func @reduce_row_sum_2d_dynamic(%in: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %dim_X = tensor.dim %in, %c0 : tensor<?x?xf32>
  %dim_Y = tensor.dim %in, %c1 : tensor<?x?xf32>

  %1 = tensor.empty(%dim_X) : tensor<?xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  %5 = gml_st.loop (%i, %j) = (%c0, %c0) to (%dim_X, %dim_Y)
         step (%c4, %c4)
         ins (%in_ = %in: tensor<?x?xf32>, %cst_ = %cst: f32)
         outs (%out_ = %2: tensor<?xf32>)
         iterators[#gml_st.iterator_type<parallel>,
                   #gml_st.iterator_type<reduction>] {
    %6 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%dim_X]
    %7 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%j)[%dim_Y]

    %8 = tensor.extract_slice %in_[%i, %j] [%6, %7] [1, 1]
           : tensor<?x?xf32> to tensor<?x?xf32>
    %11 = tensor.extract_slice %out_[%i] [%6] [1]
           : tensor<?xf32> to tensor<?xf32>
    %12 = linalg.fill ins(%cst_ : f32) outs(%11 : tensor<?xf32>) -> tensor<?xf32>
    %13 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%8 : tensor<?x?xf32>)
            outs(%12 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %14 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%13 : tensor<?xf32>)
            outs(%11 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %15 = tensor.insert_slice %14 into %out_[%i] [%6] [1]
            : tensor<?xf32> into tensor<?xf32>
    gml_st.yield %15 : tensor<?xf32>
  }
  func.return %5 : tensor<?xf32>
}

// CHECK-LABEL: func @reduce_row_sum_2d_dynamic

// CHECK:       linalg.fill
// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<4xf32>

// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<4xf32>

// CHECK:       gml_st.loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>

// -----

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_3 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = gml_st.parallel (%arg2, %arg3) = (%c0, %c0) to (%dim_1, %dim_3) step (%c8, %c4) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg2)[%dim_1]
    %4 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg3)[%dim_3]
    %5 = gml_st.tile [%arg2, 0] [%3, %dim_2] [1, 1] : !gml_st.tile<?x?>
    %6 = gml_st.materialize %arg0[%5] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %7 = gml_st.tile [0, %arg3] [%dim_2, %4] [1, 1] : !gml_st.tile<?x?>
    %8 = gml_st.materialize %arg1[%7] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %9 = gml_st.tile [%arg2, %arg3] [%3, %4] [1, 1] : !gml_st.tile<?x?>
    %10 = gml_st.materialize %1[%9] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %dim_4 = tensor.dim %6, %c0 : tensor<?x?xf32>
    %dim_5 = tensor.dim %6, %c1 : tensor<?x?xf32>
    %dim_6 = tensor.dim %8, %c1 : tensor<?x?xf32>
    %11 = gml_st.for (%arg4) = (%c0) to (%dim_5) step (%c2) outs (%arg5 = %10: tensor<?x?xf32>) {
      %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 2)>(%arg4)[%dim_5]
      %13 = gml_st.tile [0, %arg4] [%dim_4, %12] [1, 1] : !gml_st.tile<?x?>
      %14 = gml_st.materialize %6[%13] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
      %15 = gml_st.tile [%arg4, 0] [%12, %dim_6] [1, 1] : !gml_st.tile<?x?>
      %16 = gml_st.materialize %8[%15] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
      %17 = gml_st.tile [0, 0] [%dim_4, %dim_6] [1, 1] : !gml_st.tile<?x?>
      %18 = gml_st.materialize %arg5[%17] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
      %19 = linalg.matmul ins(%14, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
      gml_st.set_yield %19 into %arg5[%17] : tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
    } : tensor<?x?xf32>
    gml_st.set_yield %11 into %1[%9] : tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-DAG:  #[[$MAP_MAIN_PAR_I:.*]] = affine_map<()[s0] -> ((s0 floordiv 8) * 8)>
// CHECK-DAG:  #[[$MAP_MAIN_PAR_J:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>

// CHECK-LABEL: func @matmul(
// CHECK-SAME:    %[[LHS:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[RHS:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[LHS_ROW:.*]] = tensor.dim %[[LHS]], %[[C0]]
// CHECK-DAG:     %[[RHS_COL:.*]] = tensor.dim %[[RHS]], %[[C1]]
// CHECK-DAG:     %[[MAIN_PAR_I_UB:.*]] = affine.apply #[[$MAP_MAIN_PAR_I]]()[%[[LHS_ROW]]]
// CHECK-DAG:     %[[MAIN_PAR_J_UB:.*]] = affine.apply #[[$MAP_MAIN_PAR_J]]()[%[[RHS_COL]]]

// CHECK:         %[[MAIN_PAR:.*]] = gml_st.parallel (
// CHECK-SAME:         %[[MAIN_PAR_I:.*]], %[[MAIN_PAR_J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:         to (%[[MAIN_PAR_I_UB]], %[[MAIN_PAR_J_UB]])

// CHECK:            %[[MAIN_PAR_MAIN_FOR:.*]] = gml_st.for (
// CHECK-SAME:           %[[MAIN_PAR_MAIN_FOR_K:.*]]) = (%[[C0]])
// CHECK:              %[[MAIN_PAR_MAIN_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_MATMUL]] {{.*}} : tensor<8x4xf32> into tensor<8x4xf32>[!gml_st.tile<8x4>]

// CHECK:            %[[MAIN_PAR_REM_FOR:.*]] = gml_st.for (
// CHECK-SAME:           outs ({{.*}} = %[[MAIN_PAR_MAIN_FOR]]
// CHECK:              %[[MAIN_PAR_REM_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[MAIN_PAR_REM_FOR_MATMUL]] {{.*}} : tensor<8x4xf32> into tensor<8x4xf32>[!gml_st.tile<8x4>]

// CHECK:            gml_st.set_yield %[[MAIN_PAR_REM_FOR]]

// CHECK:         %[[REM_PAR_RHS_COL:.*]] = gml_st.parallel (
// CHECK-SAME:         %[[REM_PAR_RHS_COL_I:.*]], %[[REM_PAR_RHS_COL_J:.*]]) = (%[[C0]], %[[MAIN_PAR_J_UB]])

// CHECK:            %[[REM_PAR_RHS_COL_MAIN_FOR:.*]] = gml_st.for (
// CHECK:              %[[REM_PAR_RHS_COL_MAIN_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[REM_PAR_RHS_COL_MAIN_FOR_MATMUL]]

// CHECK:            %[[REM_PAR_RHS_COL_REM_FOR:.*]] = gml_st.for (
// CHECK-SAME:           outs ({{.*}} = %[[REM_PAR_RHS_COL_MAIN_FOR]]
// CHECK:              %[[REM_PAR_RHS_COL_REM_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[REM_PAR_RHS_COL_REM_FOR_MATMUL]]

// CHECK:            gml_st.set_yield %[[REM_PAR_RHS_COL_REM_FOR]]

// CHECK:         %[[REM_PAR_LHS_ROW:.*]] = gml_st.parallel (
// CHECK-SAME:         %[[REM_PAR_LHS_ROW_I:.*]], %[[REM_PAR_LHS_ROW_J:.*]]) = (%[[MAIN_PAR_I_UB]], %[[C0]])

// CHECK:            %[[REM_PAR_LHS_ROW_MAIN_FOR:.*]] = gml_st.for (
// CHECK:              %[[REM_PAR_LHS_ROW_MAIN_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[REM_PAR_LHS_ROW_MAIN_FOR_MATMUL]]

// CHECK:            %[[REM_PAR_LHS_ROW_REM_FOR:.*]] = gml_st.for (
// CHECK-SAME:           outs ({{.*}} = %[[REM_PAR_LHS_ROW_MAIN_FOR]]:
// CHECK:              %[[REM_PAR_LHS_ROW_REM_FOR_MATMUL:.*]] = linalg.matmul ins({{.*}})
// CHECK-NEXT:         gml_st.set_yield %[[REM_PAR_LHS_ROW_REM_FOR_MATMUL]]

// CHECK:            gml_st.set_yield %[[REM_PAR_LHS_ROW_REM_FOR]]
