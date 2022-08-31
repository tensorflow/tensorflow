// TODO(jreiffers): Remove -cse below once the duplicate init_tensor instruction
// is fixed.

// RUN: mlir-hlo-opt %s --split-input-file --gml-deprecated-tiling=tile-sizes=[256,512] \
// RUN:     --gml-deprecated-fusion --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-TILE

// RUN: mlir-hlo-opt %s --split-input-file --gml-deprecated-tiling=tile-sizes=[1,1] \
// RUN:     --gml-deprecated-fusion --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-POINT

func.func @pointwise(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %generic = linalg.generic {
     indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1, d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
     ],
     iterator_types = ["parallel", "parallel"]}
     ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
     outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %sum = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %sum : f32
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

// CHECK-TILE-LABEL: @pointwise
// CHECK-TILE-SAME:  %[[ARG0:.*]]:{{.*}}%[[ARG1:.*]]:
// CHECK-TILE:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-TILE:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-TILE:       %[[OUTPUT_TILE:.*]] = gml_st.tile %{{.*}} [%[[I]], %[[J]]]
// CHECK-TILE:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0]][%[[OUTPUT_TILE]]]
// CHECK-TILE:       %[[RHS_TILE:.*]] = gml_st.transpose_dims %[[OUTPUT_TILE]], [1, 0]
// CHECK-TILE:       %[[ARG1_MAT:.*]] = gml_st.materialize %[[ARG1]][%[[RHS_TILE]]]
// CHECK-TILE:       %[[INIT_MAT:.*]] = gml_st.materialize %[[INIT]][%[[OUTPUT_TILE]]]
// CHECK-TILE:       %[[OUT:.*]] = linalg.generic {
// CHECK-TILE-SAME:      ins(%[[ARG0_MAT]], %[[ARG1_MAT]]
// CHECK-TILE-SAME:      outs(%[[INIT_MAT]]
// CHECK-TILE:       gml_st.set_yield %[[OUT]] into %[[INIT]][%[[OUTPUT_TILE]]]

// CHECK-POINT-LABEL: @pointwise
// CHECK-POINT-SAME:  %[[ARG0:.*]]:{{.*}}%[[ARG1:.*]]:
// CHECK-POINT:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-POINT:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-POINT:       %[[OUTPUT_POINT:.*]] = gml_st.point %{{.*}} [%[[I]], %[[J]]]
// CHECK-POINT:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0]][%[[OUTPUT_POINT]]]
// CHECK-POINT:       %[[ARG1_POINT:.*]] = gml_st.transpose_dims %[[OUTPUT_POINT]], [1, 0]
// CHECK-POINT:       %[[ARG1_MAT:.*]] = gml_st.materialize %[[ARG1]][%[[ARG1_POINT]]]
// CHECK-POINT:       %[[OUT:.*]] = arith.addf %[[ARG0_MAT]], %[[ARG1_MAT]]
// CHECK-POINT:       gml_st.set_yield %[[OUT]] into %[[INIT]][%[[OUTPUT_POINT]]]

// -----

func.func @broadcast(%arg0: tensor<?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %init = linalg.init_tensor [%d0, %d0] : tensor<?x?xf32>
  %generic = linalg.generic {
     indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
     ],
     iterator_types = ["parallel", "parallel"]}
     ins(%arg0 : tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %_: f32):
    linalg.yield %lhs_scalar : f32
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

// CHECK-TILE-LABEL: @broadcast
// CHECK-TILE-SAME:  %[[ARG0:.*]]:
// CHECK-TILE:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-TILE:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-TILE:       %[[D0_SIZE:.*]] = arith.minsi {{.*}}, %c256
// CHECK-TILE:       %[[OUTPUT_TILE:.*]] = gml_st.tile %{{.*}} [%[[I]], %[[J]]]
// CHECK-TILE:       %[[INPUT_SPACE:.*]] = gml_st.space [%[[D0_SIZE]]]
// CHECK-TILE:       %[[INPUT_TILE:.*]] = gml_st.tile %[[INPUT_SPACE]] [%[[I]]]
// CHECK-TILE-SAME:  [%[[D0_SIZE]]] [%c1]
// CHECK-TILE:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0]][%[[INPUT_TILE]]]
// CHECK-TILE:       %[[INIT_MAT:.*]] = gml_st.materialize %[[INIT]][%[[OUTPUT_TILE]]]
// CHECK-TILE:       %[[OUT:.*]] = linalg.generic
// CHECK-TILE-SAME:      ins(%[[ARG0_MAT]]
// CHECK-TILE-SAME:      outs(%[[INIT_MAT]]
// CHECK-TILE:       gml_st.set_yield %[[OUT]] into %[[INIT]][%[[OUTPUT_TILE]]]

// CHECK-POINT-LABEL: @broadcast
// CHECK-POINT-SAME:  %[[ARG0:.*]]:
// CHECK-POINT:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-POINT:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-POINT:       %[[OUTPUT_POINT:.*]] = gml_st.point %{{.*}} [%[[I]], %[[J]]]
// CHECK-POINT:       %[[INPUT_SPACE:.*]] = gml_st.space [1]
// CHECK-POINT:       %[[INPUT_POINT:.*]] = gml_st.point %[[INPUT_SPACE]] [%[[I]]]
// CHECK-POINT:       %[[OUT:.*]] = gml_st.materialize %[[ARG0]][%[[INPUT_POINT]]]
// CHECK-POINT:       gml_st.set_yield %[[OUT]] into %[[INIT]][%[[OUTPUT_POINT]]]

// -----

func.func @reduction(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %generic = linalg.generic {
     indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
     ],
     iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%arg0 : tensor<?x?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %acc: f32):
    %sum = arith.addf %lhs_scalar, %acc : f32
    linalg.yield %sum : f32

  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

// CHECK-TILE-LABEL: @reduction
// CHECK-TILE-SAME:  %[[ARG0:.*]]:
// CHECK-TILE:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-TILE:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-TILE:       %[[OUTPUT_TILE:.*]] = gml_st.tile %3 [%[[I]], %[[J]]]
// CHECK-TILE-SAME:  [%[[D0_SIZE:.*]], %[[D1_SIZE:.*]]] [1, 1]
// CHECK-TILE:       %[[D2_SIZE:.*]] = tensor.dim %[[ARG0]], %c2
// CHECK-TILE:       %[[INPUT_SPACE:.*]] = gml_st.space [%[[D0_SIZE]], %[[D1_SIZE]],
// CHECK-TILE-SAME:  %[[D2_SIZE]]]
// CHECK-TILE:       %[[INPUT_TILE:.*]] = gml_st.tile %[[INPUT_SPACE]]
// CHECK-TILE-SAME:  [%[[I]], %[[J]], 0]
// CHECK-TILE-SAME:  [%[[D0_SIZE]], %[[D1_SIZE]], %[[D2_SIZE]]]
// CHECK-TILE-SAME:  [%c1, %c1, 1]
// CHECK-TILE:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0]][%[[INPUT_TILE]]]
// CHECK-TILE:       %[[INIT_MAT:.*]] = gml_st.materialize %[[INIT]][%[[OUTPUT_TILE]]]
// CHECK-TILE:       %[[OUT:.*]] = linalg.generic
// CHECK-TILE-SAME:      ins(%[[ARG0_MAT]]
// CHECK-TILE-SAME:      outs(%[[INIT_MAT]]
// CHECK-TILE:       gml_st.set_yield %[[OUT]] into %[[INIT]][%[[OUTPUT_TILE]]]

// CHECK-POINT-LABEL: @reduction
// CHECK-POINT-SAME:  %[[ARG0:.*]]:
// CHECK-POINT:       %[[INIT:.*]] = linalg.init_tensor
// CHECK-POINT:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-POINT:       %[[OUTPUT_POINT:.*]] = gml_st.point {{.*}} [%[[I]], %[[J]]]
// CHECK-POINT:       %[[REDUCTION_SIZE:.*]] = tensor.dim %[[ARG0]], %c2
// CHECK-POINT:       %[[INPUT_TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]], 0]
// CHECK-POINT-SAME:  [1, 1, %[[REDUCTION_SIZE]]] [1, 1, 1]
// CHECK-POINT:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0]][%[[INPUT_TILE]]]
// CHECK-POINT:       %[[INIT_MAT_SCALAR:.*]] = gml_st.materialize %[[INIT]][%[[OUTPUT_POINT]]]
// CHECK-POINT:       %[[INIT_TENSOR:.*]] = tensor.from_elements %[[INIT_MAT_SCALAR]]
// CHECK-POINT:       %[[OUT:.*]] = linalg.generic
// CHECK-POINT-SAME:      ins(%[[ARG0_MAT]] :
// CHECK-POINT-SAME:      outs(%[[INIT_TENSOR]] :
// CHECK-POINT:       %[[OUT_SCALAR:.*]] = tensor.extract %[[OUT]][%c0, %c0]
// CHECK-POINT:       gml_st.set_yield %[[OUT_SCALAR]] into %[[INIT]][%[[OUTPUT_POINT]]]

// -----

func.func @broadcast_reduction(%arg0: tensor<?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %init = linalg.init_tensor [%d0, %d0] : tensor<?x?xf32>
  %generic = linalg.generic {
     indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
     ],
     iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%arg0, %arg1 : tensor<?xf32>, tensor<?x?x?xf32>)
     outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %acc: f32):
    %sum1 = arith.addf %lhs_scalar, %acc : f32
    %sum2 = arith.addf %rhs_scalar, %sum1 : f32
    linalg.yield %sum2 : f32
  } -> tensor<?x?xf32>
  return %generic : tensor<?x?xf32>
}

// CHECK-TILE-LABEL: @broadcast_reduction
// CHECK-TILE:       %[[OUTPUT_TILE:.*]] = gml_st.tile {{.*}} [%[[I:.*]], %[[J:.*]]] [%[[D0_SIZE:.*]], %[[D1_SIZE:.*]]] [
// CHECK-TILE:       %[[ARG0_TILE:.*]] = gml_st.tile {{.*}} [%[[I]]] [%[[D0_SIZE]]]
// CHECK-TILE:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0:.*]][%[[ARG0_TILE]]]
// CHECK-TILE:       %[[ARG1_TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]], 0] [%[[D0_SIZE]], %[[D1_SIZE]]
// CHECK-TILE:       %[[ARG1_MAT:.*]] = gml_st.materialize %[[ARG1:.*]][%[[ARG1_TILE]]]
// CHECK-TILE:       %[[OUT:.*]] = linalg.generic
// CHECK-TILE-SAME:      ins(%[[ARG0_MAT]], %[[ARG1_MAT]]
// CHECK-TILE:       gml_st.set_yield %[[OUT]] into %{{.*}}[%[[OUTPUT_TILE]]]

// CHECK-POINT-LABEL: @broadcast_reduction
// CHECK-POINT:       %[[OUTPUT_POINT:.*]] = gml_st.point {{.*}} [%[[I:.*]], %[[J:.*]]]
// CHECK-POINT:       %[[ARG0_POINT:.*]] = gml_st.point {{.*}} [%[[I]]]
// CHECK-POINT:       %[[ARG0_MAT:.*]] = gml_st.materialize %[[ARG0:.*]][%[[ARG0_POINT]]]
// CHECK-POINT:       %[[ARG1_TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]], 0] [1, 1, %
// CHECK-POINT:       %[[ARG1_MAT:.*]] = gml_st.materialize %[[ARG1:.*]][%[[ARG1_TILE]]]
// CHECK-POINT:       %[[OUT:.*]] = linalg.generic
// CHECK-POINT-SAME:      ins(%[[ARG0_MAT]], %[[ARG1_MAT]]
// CHECK-POINT:       %[[OUT_SCALAR:.*]] = tensor.extract %[[OUT]]
// CHECK-POINT:       gml_st.set_yield %[[OUT_SCALAR]] into %{{.*}}[%[[OUTPUT_POINT]]]
