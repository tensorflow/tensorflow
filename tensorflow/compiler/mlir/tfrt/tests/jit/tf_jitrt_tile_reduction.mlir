// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -tf-jitrt-tile-reduction="reduction-2d-tile-sizes=4,4 reduction-vector-size=8 reduction-1d-tile-size=16" \
// RUN: | FileCheck %s

func.func @reduce_row_sum_2d(%lhs: tensor<?x?xf32>,
                        %rhs: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0] : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?xf32>) -> tensor<?xf32>
  %sum_of_prod = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %o : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum_of_prod : tensor<?xf32>
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
// CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[LHS]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]] : [[TY_2D]]

// CHECK:      gml_st.loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: [[TY_2D]],
// CHECK-SAME:        %[[RHS_:.*]] = %[[RHS]]: [[TY_2D]])
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_1D]])

// CHECK:      %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]], %[[J]]]
// CHECK:      %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][%[[I]], %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]]]

// CHECK:      %[[SUM_SUB:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[LHS_SUB]], %[[RHS_SUB]] : [[TY_2D]], [[TY_2D]])
// CHECK-SAME:   outs(%[[OUT_SUB]] : [[TY_1D]])
// CHECK:          mulf
// CHECK:          addf
// CHECK-NEXT:     linalg.yield

// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[SUM_SUB]] into %[[OUT_]]
// CHECK-NEXT: gml_st.yield %[[UPDATE]] : [[TY_1D]]

// -----

func.func @reduce_row_sum_2d_static(%input: tensor<8x16xf32>) -> tensor<8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<8x16xf32>

  %init = linalg.init_tensor [8] : tensor<8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<8xf32>) -> tensor<8xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<8x16xf32>)
    outs(%fill : tensor<8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<8xf32>
  func.return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @reduce_row_sum_2d_static
// CHECK: gml_st.loop
// CHECK:   tensor.insert_slice

// -----

func.func @reduce_column_sum_2d(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0] : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>],
    iterator_types = ["reduction", "parallel"]}
    ins(%input : tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum : tensor<?xf32>
}
// CHECK-LABEL: func @reduce_column_sum_2d
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xf32>) -> tensor<?xf32>

// CHECK-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index

// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D:.*]]
// CHECK:      %[[INIT:.*]] = linalg.init_tensor [%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : [[TY_2D]]

// CHECK:      gml_st.loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[IN_:.*]] = %[[INPUT]]: [[TY_2D]])
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_1D]])

// CHECK:      %[[IN_SUB:.*]] = tensor.extract_slice %[[IN_]][%[[I]], %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[J]]]

// CHECK:      %[[SUM_SUB:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[IN_SUB]] : [[TY_2D]])
// CHECK-SAME:   outs(%[[OUT_SUB]] : [[TY_1D]])
// CHECK:          addf
// CHECK-NEXT:     linalg.yield

// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[ACC:.*]] into %[[OUT_]]
// CHECK:      gml_st.yield %[[UPDATE]] : [[TY_1D]]

// -----

func.func @abs(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %abs = math.abs %in: f32
    linalg.yield %abs : f32
  } -> tensor<?x?xf32>
  func.return %sum : tensor<?x?xf32>
}
// CHECK-LABEL: func @abs
// CHECK-NOT: gml_st.loop

// -----

func.func @reduce_sum_1d(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?xf32>

  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<f32>) -> tensor<f32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
    outs(%fill : tensor<f32>) {
  ^bb0(%l: f32, %r: f32, %out: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %out : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  func.return %sum : tensor<f32>
}

// CHECK-LABEL: func @reduce_sum_1d(
// CHECK-SAME:    %[[LHS:.*]]: tensor<?xf32>, %[[RHS:.*]]: tensor<?xf32>)
     // CHECK-DAG: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
     // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
     // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index

     // CHECK: %[[INIT:.*]] = linalg.init_tensor [] : tensor<f32>
     // CHECK: %[[FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[INIT]]
     // CHECK: %[[INPUT_SIZE:.*]] = tensor.dim %[[LHS]], %[[C0]]

     // CHECK: %[[TMP_INIT:.*]] = linalg.init_tensor [8] : tensor<8xf32>
     // CHECK: %[[TMP_FILL:.*]] = linalg.fill ins(%[[C0_F32]]{{.*}}outs(%[[TMP_INIT]]
     // CHECK: %[[TILABLE_UB:.*]] = affine.apply
// CHECK-SAME:  %[[INPUT_SIZE]]
     // CHECK: %[[TMP_SUM:.*]] = gml_st.loop (%[[I:.*]]) = (%[[C0]])
// CHECK-SAME:   to (%[[TILABLE_UB]]) step (%[[C16]])
// CHECK-SAME:   ins (%[[LHS_:.*]] = %[[LHS]]: tensor<?xf32>,
// CHECK-SAME:        %[[RHS_:.*]] = %[[RHS]]: tensor<?xf32>)
// CHECK-SAME:   outs (%[[TMP_INIT_:.*]] = %[[TMP_FILL]]: tensor<8xf32>)

     // CHECK: %[[LHS_SUB:.*]] = tensor.extract_slice %[[LHS_]][%[[I]]]
     // CHECK: %[[LHS_RESHAPE:.*]] = tensor.expand_shape %[[LHS_SUB]]
// CHECK-SAME:   {{\[\[}}0, 1]]
// CHECK-SAME:   : tensor<16xf32> into tensor<2x8xf32>

     // CHECK: %[[RHS_SUB:.*]] = tensor.extract_slice %[[RHS_]][%[[I]]]
     // CHECK: %[[RHS_RESHAPE:.*]] = tensor.expand_shape %[[RHS_SUB]]
// CHECK-SAME:   {{\[\[}}0, 1]]
// CHECK-SAME:   : tensor<16xf32> into tensor<2x8xf32>

     // CHECK: %[[SUM_OF_PROD:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[LHS_RESHAPE]], %[[RHS_RESHAPE]]
// CHECK-SAME:       tensor<2x8xf32>, tensor<2x8xf32>)
// CHECK-SAME:   outs(%[[TMP_INIT_]] : tensor<8xf32>) {
     // CHECK:   ^bb0(%[[L:.*]]: f32, %[[R:.*]]: f32, %[[O:.*]]: f32):
     // CHECK:     %[[MUL:.*]] = arith.mulf %[[L]], %[[R]] : f32
     // CHECK:     %[[ADD:.*]] = arith.addf %[[MUL]], %[[O]] : f32
     // CHECK:       linalg.yield %[[ADD]] : f32
     // CHECK:     } -> tensor<8xf32>
     // CHECK:   gml_st.yield %[[SUM_OF_PROD]] : tensor<8xf32>
     // CHECK: }
     // CHECK: %[[HORIZONTAL_REDUCE:.*]] = linalg.generic
// CHECK-SAME: ins(%[[TMP_SUM]] : tensor<8xf32>) outs(%[[FILL]] : tensor<f32>)
//  CHECK-NOT:  mulf
//      CHECK:  addf

     // CHECK: gml_st.loop (%[[K:.*]]) = (%[[TILABLE_UB]])
// CHECK-SAME:   to (%[[INPUT_SIZE]]) step (%[[C16]])
     // CHECK: linalg.generic
//      CHECK:  mulf
//      CHECK:  addf
