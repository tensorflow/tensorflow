// RUN: tf-tfrt-opt -tf-cpurt-codegen-reduction %s --split-input-file |\
// RUN: FileCheck %s

func @reduce_row_sum_2d(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0] : tensor<?xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<?xf32> -> tensor<?xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}
// CHECK-LABEL: func @reduce_row_sum_2d(
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
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]]]
// CHECK:      %[[CLONE_SUB:.*]] = tensor.extract_slice %[[CLONE_]][%[[I]]]

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

// -----

func @reduce_row_sum_2d_static(%input: tensor<8x16xf32>) -> tensor<8xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<8x16xf32>

  %init = linalg.init_tensor [8] : tensor<8xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<8xf32> -> tensor<8xf32>
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
  return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @reduce_row_sum_2d_static
// CHECK: linalg.tiled_loop
// CHECK:   tensor.insert_slice
// CHECK:   tensor.insert_slice

// -----

func @reduce_column_sum_2d(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<?x?xf32>

  %init = linalg.init_tensor [%0] : tensor<?xf32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<?xf32> -> tensor<?xf32>
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
  return %sum : tensor<?xf32>
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

// -----

func @abs(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
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
  return %sum : tensor<?x?xf32>
}
// CHECK-LABEL: func @abs
// CHECK-NOT: linalg.tiled_loop

// -----

func @reduce_sum_1d(%input: tensor<?xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %input, %c0 : tensor<?xf32>

  %init = linalg.init_tensor [] : tensor<f32>
  %fill = linalg.fill(%cst, %init) : f32, tensor<f32> -> tensor<f32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]}
    ins(%input : tensor<?xf32>)
    outs(%fill : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %add = arith.addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<f32>
  return %sum : tensor<f32>
}

// CHECK-LABEL: func @reduce_sum_1d(
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?xf32>) -> tensor<f32> {
     // CHECK: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
     // CHECK: %[[C0:.*]] = arith.constant 0 : index
     // CHECK: %[[C8:.*]] = arith.constant 8 : index

     // CHECK: %[[INIT:.*]] = linalg.init_tensor [] : tensor<f32>
     // CHECK: %[[FILL:.*]] = linalg.fill(%[[C0_F32]], %[[INIT]])
     // CHECK: %[[INPUT_SIZE:.*]] = tensor.dim %[[INPUT]], %[[C0]]

     // CHECK: %[[TMP_INIT:.*]] = linalg.init_tensor [8] : tensor<8xf32>
     // CHECK: %[[TMP_FILL:.*]] = linalg.fill(%[[C0_F32]], %[[TMP_INIT]])
     // CHECK: %[[TMP_SUM:.*]] = linalg.tiled_loop (%[[I:.*]]) = (%[[C0]])
// CHECK-SAME:   to (%[[INPUT_SIZE]]) step (%[[C8]])
// CHECK-SAME:   ins (%[[INPUT_:.*]] = %[[INPUT]]: tensor<?xf32>)
// CHECK-SAME:   outs (%[[TMP_INIT_:.*]] = %[[TMP_FILL]]: tensor<8xf32>)

     // CHECK: %[[IN_SUB:.*]] = tensor.extract_slice %[[INPUT_]][%[[I]]]
     // CHECK: %[[PAD:.*]] = linalg.pad_tensor %[[IN_SUB]]
     // CHECK: %[[RESHAPE:.*]] = linalg.tensor_expand_shape %[[PAD]]
// CHECK-SAME:   {{\[\[}}0, 1]]
// CHECK-SAME:   : tensor<8xf32> into tensor<1x8xf32>

     // CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME:   ins(%[[RESHAPE]] : tensor<1x8xf32>)
// CHECK-SAME:   outs(%[[TMP_INIT_]] : tensor<8xf32>) {
     // CHECK:   ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32):
     // CHECK:     %[[ADD:.*]] = arith.addf %[[A]], %[[B]] : f32
     // CHECK:       linalg.yield %[[ADD]] : f32
     // CHECK:     } -> tensor<8xf32>
     // CHECK:   linalg.yield %[[SUM]] : tensor<8xf32>
     // CHECK: }
     // CHECK: linalg.generic
// CHECK-SAME: ins(%[[TMP_SUM]] : tensor<8xf32>) outs(%[[FILL]] : tensor<f32>)
