// RUN: tf-tfrt-opt -tf-cpurt-codegen-reduction %s --split-input-file | FileCheck %s

func @reduce_column_sum_2d(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
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
    %add = addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}
// CHECK-LABEL: func @reduce_column_sum_2d(
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x?xf32>) -> tensor<?xf32>

// CHECK-DAG:  %[[C0_F32:.*]] = constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = constant 0 : index
// CHECK-DAG:  %[[C4:.*]] = constant 4 : index
// CHECK-DAG:  %[[C1:.*]] = constant 1 : index

// CHECK:      %[[DIM_0:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D:.*]]
// CHECK:      %[[INIT:.*]] = linalg.init_tensor {{\[}}%[[DIM_0]]] : [[TY_1D:.*]]
// CHECK:      %[[FILL:.*]] = linalg.fill(%[[C0_F32]], %[[INIT]])
// CHECK:      %[[DIM_0_:.*]] = tensor.dim %[[INPUT]], %[[C0]] : [[TY_2D]]
// CHECK:      %[[DIM_1:.*]] = tensor.dim %[[INPUT]], %[[C1]] : [[TY_2D]]

// CHECK:      linalg.tiled_loop (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:   to (%[[DIM_0_]], %[[DIM_1]]) step (%[[C4]], %[[C4]])
// CHECK-SAME:   ins (%[[IN_:.*]] = %[[INPUT]]: [[TY_2D]],
// CHECK-SAME:                      %[[C0_F32_:.*]] = %[[C0_F32]]: f32)
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[FILL]]: [[TY_1D]])

// CHECK:      %[[IN_SUB:.*]] = tensor.extract_slice %[[IN_]][%[[I]], %[[J]]]
// CHECK:      %[[OUT_SUB:.*]] = tensor.extract_slice %[[OUT_]][%[[I]]]

// CHECK:      %[[FILL_SUB:.*]] = linalg.fill(%[[C0_F32_]], %[[OUT_SUB]])

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

// CHECK:      %[[UPDATE:.*]] = tensor.insert_slice %[[ACC:.*]] into %[[OUT_]]
// CHECK:      linalg.yield %[[UPDATE]] : [[TY_1D]]

// -----

func @reduce_column_sum_2d_static(%input: tensor<8x16xf32>) -> tensor<8xf32> {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
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
    %add = addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<8xf32>
  return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @reduce_column_sum_2d_static
// CHECK: linalg.tiled_loop
// CHECK:   %[[LOCAL_INIT:.*]] = linalg.init_tensor [4] : tensor<4xf32>
// CHECK:   linalg.fill(%{{.*}}, %[[LOCAL_INIT]]) : f32, tensor<4xf32>

// -----

func @reduce_row_sum_2d(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
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
    %add = addf %in, %out : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}
// CHECK-LABEL: func @reduce_row_sum_2d
// CHECK-NOT: linalg.tiled_loop

// -----

func @abs(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
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
    %abs = absf %in: f32
    linalg.yield %abs : f32
  } -> tensor<?x?xf32>
  return %sum : tensor<?x?xf32>
}
// CHECK-LABEL: func @abs
// CHECK-NOT: linalg.tiled_loop
