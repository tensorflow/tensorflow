// RUN: mlir-hlo-opt %s -split-input-file \
// RUN: -test-gml-st-loop-tiling="tile-sizes=2,3,4 distribution-types=block_x,block_y,none" \
// RUN: | FileCheck %s

func.func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_tensors
// CHECK-SAME: (%[[ARG_0:.*]]: [[TY:.*]], %[[ARG_1:.*]]: [[TY]],
// CHECK-SAME: %[[ARG_2:.*]]: [[TY]]) -> [[TY]] {

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index

// CHECK: %[[ARG_0_X:.*]] = tensor.dim %[[ARG_0]], %[[C0]] : [[TY]]
// CHECK: %[[ARG_0_Y:.*]] = tensor.dim %[[ARG_0]], %[[C1]] : [[TY]]
// CHECK: %[[ARG_1_Y:.*]] = tensor.dim %[[ARG_1]], %[[C1]] : [[TY]]

// CHECK: %{{.*}} = gml_st.loop (%[[I:.*]], %[[J:.*]], %[[K:.*]]) =
// CHECK-SAME: (%[[C0]], %[[C0]], %[[C0]])
// CHECK-SAME: to (%[[ARG_0_X]], %[[ARG_1_Y]], %[[ARG_0_Y]])
// CHECK-SAME: step (%[[C2]], %[[C3]], %[[C4]])
// CHECK-SAME: ins (%[[A0:.*]] = %[[ARG_0]]: [[TY]], %[[A1:.*]] = %[[ARG_1]]: [[TY]])
// CHECK-SAME: outs (%[[A2:.*]] = %[[ARG_2]]: [[TY]])
// CHECK-SAME: iterators["parallel", "parallel", "reduction"]
// CHECK-SAME: distribution["block_x", "block_y", "none"] {

// CHECK: %[[SUB_ARG_0:.*]] = tensor.extract_slice %[[A0]][%[[I]], %[[K]]]
// CHECK: %[[SUB_ARG_1:.*]] = tensor.extract_slice %[[A1]][%[[K]], %[[J]]]
// CHECK: %[[SUB_ARG_2:.*]] = tensor.extract_slice %[[A2]][%[[I]], %[[J]]]

// CHECK: %[[PROD:.*]] = linalg.matmul ins(%[[SUB_ARG_0]], %[[SUB_ARG_1]]
// CHECK-SE: outs(%[[SUB_ARG_2]] : [[TY]]) -> [[TY]]

// CHECK: %[[O:.*]] = tensor.insert_slice %[[PROD]] into %[[A2]][%[[I]], %[[J]]]
// CHECK: gml_st.yield %[[O]] : [[TY]]

// -----

func.func @generic_op_tensors(
  %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = linalg.init_tensor [%0, %1, %2] : tensor<?x?x?xf32>
  %4 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d2, d1, d0)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%arg2 : f32, %arg3: f32, %arg4: f32):
      %5 = arith.addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?x?xf32>
  func.return %4 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @generic_op_tensors(
// CHECK-SAME:    %[[ARG_0:.*]]: [[TY:.*]],
// CHECK-SAME:    %[[ARG_1:.*]]: [[TY]]) -> [[TY]] {

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index

// CHECK:     %[[INIT:.*]] = linalg.init_tensor
// CHECK:     %[[ARG_0_X:.*]] = tensor.dim %[[ARG_0]], %[[C0]] : [[TY]]
// CHECK:     %[[ARG_0_Y:.*]] = tensor.dim %[[ARG_0]], %[[C1]] : [[TY]]
// CHECK:     %[[ARG_0_Z:.*]] = tensor.dim %[[ARG_0]], %[[C2]] : [[TY]]

// CHECK:     %{{.*}} = gml_st.loop (%{{.*}}, %{{.*}}, %{{.*}}) =
// CHECK-SAME: (%[[C0]], %[[C0]], %[[C0]])
// CHECK-SAME: to (%[[ARG_0_X]], %[[ARG_0_Y]], %[[ARG_0_Z]])
// CHECK-SAME: step (%[[C2]], %[[C3]], %[[C4]])
// CHECK-SAME: ins (%{{.*}} = %[[ARG_0]]: [[TY]], %{{.*}} = %[[ARG_1]]: [[TY]])
// CHECK-SAME: outs (%{{.*}} = %[[INIT]]: [[TY]])
// CHECK-SAME: distribution["block_x", "block_y", "none"] {
