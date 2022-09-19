// RUN: mlir-hlo-opt %s -test-gml-st-bufferization -canonicalize -cse \
// RUN:   -split-input-file | FileCheck %s

func.func @set_space(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %identity = gml_st.materialize %input[%space]
    : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>

  return %identity : tensor<?x?xf32>
}
// CHECK-LABEL: func @set_space(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-NEXT:  return %[[ARG]] : memref<?x?xf32>

// -----

func.func @set_tile(%input: tensor<?x?xf32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %tile = gml_st.tile %space[0, 1][2, 4][1, 1]
    : !gml_st.tile<?x?> to !gml_st.tile<2x4>

  %slice = gml_st.materialize %input[%tile]
    : tensor<?x?xf32>[!gml_st.tile<2x4>] to tensor<2x4xf32>

  return %slice : tensor<2x4xf32>
}
// CHECK-LABEL: func @set_tile(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-NEXT:  %[[VIEW:.*]] = memref.subview %[[ARG]][0, 1] [2, 4] [1, 1]
// CHECK-NEXT:  %[[ALLOC:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK-NEXT:  memref.copy %[[VIEW]], %[[ALLOC]]
// CHECK-NEXT:  return %[[ALLOC]] : memref<2x4xf32>

// -----

func.func @set_point(%input: tensor<?x?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %pt = gml_st.point %space[0, 1] : !gml_st.tile<?x?> to !gml_st.point

  %element = gml_st.materialize %input[%pt]
    : tensor<?x?xf32>[!gml_st.point] to f32

  return %element : f32
}
// CHECK-LABEL: func @set_point(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[ELEM:.*]] = memref.load %[[ARG]][%[[C0]], %[[C1]]]
// CHECK-NEXT:  return %[[ELEM]] : f32

// -----

func.func @parallel_with_points(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                                %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>

  %result = gml_st.parallel (%i, %j) = (%c0, %c0)
      to (%dim_0, %dim_1) step (%c1, %c1) {
    %pt = gml_st.point %space [%i, %j] : !gml_st.tile<?x?> to !gml_st.point
    %lhs_elem = gml_st.materialize %lhs[%pt] : tensor<?x?xf32>[!gml_st.point] to f32
    %rhs_elem = gml_st.materialize %rhs[%pt] : tensor<?x?xf32>[!gml_st.point] to f32

    %add_elem = arith.addf %lhs_elem, %rhs_elem : f32

    gml_st.set_yield %add_elem into %init[%pt]
      : f32 into tensor<?x?xf32>[!gml_st.point]
  } : tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: func @parallel_with_points(
// CHECK-SAME:    %[[LHS:.*]]: memref<?x?xf32>, %[[RHS:.*]]: memref<?x?xf32>,
// CHECK-SAME:    %[[OUT:.*]]: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK:     %[[DIM_0:.*]] = memref.dim %[[LHS]], %[[C0]] : memref<?x?xf32>
// CHECK:     %[[DIM_1:.*]] = memref.dim %[[LHS]], %[[C1]] : memref<?x?xf32>

// CHECK:     gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:    to (%[[DIM_0]], %[[DIM_1]]) step (%[[C1]], %[[C1]]) {

// CHECK-DAG:   %[[RHS_EL:.*]] = memref.load %[[RHS]][%[[I]], %[[J]]]
// CHECK-DAG:   %[[LHS_EL:.*]] = memref.load %[[LHS]][%[[I]], %[[J]]]
// CHECK:       %[[ADD_EL:.*]] = arith.addf %[[LHS_EL]], %[[RHS_EL]] : f32
// CHECK:       memref.store %[[ADD_EL]], %[[OUT]][%[[I]], %[[J]]]
// CHECK:       gml_st.set_yield
// CHECK:     }
// CHECK:     return %[[OUT]] : memref<?x?xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @parallel_with_tiles(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                               %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim_0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>

  %result = gml_st.parallel (%i, %j) = (%c0, %c0) to (%dim_0, %dim_1) step (%c4, %c1) {
    %7 = arith.addi %i, %c4 : index
    %8 = arith.cmpi sgt, %7, %dim_0 : index
    %9 = arith.subi %dim_0, %i : index
    %size_0 = arith.select %8, %9, %c4 : index

    %tile = gml_st.tile %space [%i, %j] [%size_0, 1] [1, 1]
      : !gml_st.tile<?x?> to !gml_st.tile<?x1>
    %lhs_tile = gml_st.materialize %lhs[%tile]
      : tensor<?x?xf32>[!gml_st.tile<?x1>] to tensor<?x1xf32>
    %rhs_tile = gml_st.materialize %rhs[%tile]
      : tensor<?x?xf32>[!gml_st.tile<?x1>] to tensor<?x1xf32>
    %init_tile = gml_st.materialize %init[%tile]
      : tensor<?x?xf32>[!gml_st.tile<?x1>] to tensor<?x1xf32>
    %sum = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs_tile, %rhs_tile : tensor<?x1xf32>, tensor<?x1xf32>)
        outs(%init_tile : tensor<?x1xf32>) {
      ^bb0(%l: f32, %r: f32, %o: f32):
        %add = arith.addf %l, %r : f32
        linalg.yield %add : f32
      } -> tensor<?x1xf32>
    gml_st.set_yield %sum into %init[%tile]
      : tensor<?x1xf32> into tensor<?x?xf32>[!gml_st.tile<?x1>]
  } : tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @parallel_with_tiles(
// CHECK-SAME: %[[LHS:.*]]: memref<?x?xf32>, %[[RHS:.*]]: memref<?x?xf32>,
// CHECK-SAME: %[[OUT:.*]]: memref<?x?xf32>) -> memref<?x?xf32> {

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK:     %[[DIM_0:.*]] = memref.dim %[[LHS]], %[[C0]] : memref<?x?xf32>
// CHECK:     %[[DIM_1:.*]] = memref.dim %[[LHS]], %[[C1]] : memref<?x?xf32>

// CHECK:     gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:    to (%[[DIM_0]], %[[DIM_1]]) step (%[[C4]], %[[C1]]) {

// CHECK-DAG:   %[[LHS_SUB:.*]] = memref.subview %[[LHS]][%[[I]], %[[J]]]
// CHECK-SAME:    : memref<?x?xf32> to memref<?x1xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:   %[[RHS_SUB:.*]] = memref.subview %[[RHS]][%[[I]], %[[J]]]
// CHECK-SAME:    : memref<?x?xf32> to memref<?x1xf32, strided<[?, 1], offset: ?>>
// CHECK-DAG:   %[[OUT_SUB:.*]] = memref.subview %[[OUT]][%[[I]], %[[J]]]
// CHECK-SAME:    : memref<?x?xf32> to memref<?x1xf32, strided<[?, 1], offset: ?>>

// CHECK:       linalg.generic {
// CHECK-SAME:    indexing_maps = [#[[$MAP1]], #[[$MAP1]], #[[$MAP1]]]
// CHECK-SAME:    ins(%[[LHS_SUB]], %[[RHS_SUB]] : memref<?x1xf32, strided<[?, 1], offset: ?>>
// CHECK-SAME:    outs(%[[OUT_SUB]] : memref<?x1xf32, strided<[?, 1], offset: ?>>)
// CHECK:       gml_st.set_yield
// CHECK:     }
// CHECK: return %[[OUT]] : memref<?x?xf32>

// -----

func.func @for_with_points(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                           %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>

  %result = gml_st.for (%i, %j) = (%c0, %c0)
      to (%dim_0, %dim_1) step (%c1, %c1)
      outs(%out_ = %init : tensor<?x?xf32>) {
    %pt = gml_st.point %space [%i, %j] : !gml_st.tile<?x?> to !gml_st.point
    %lhs_elem = gml_st.materialize %lhs[%pt] : tensor<?x?xf32>[!gml_st.point] to f32
    %rhs_elem = gml_st.materialize %rhs[%pt] : tensor<?x?xf32>[!gml_st.point] to f32

    %add_elem = arith.addf %lhs_elem, %rhs_elem : f32

    gml_st.set_yield %add_elem into %out_[%pt]
      : f32 into tensor<?x?xf32>[!gml_st.point]
  } : tensor<?x?xf32>
  func.return %result: tensor<?x?xf32>
}

// CHECK-LABEL: func @for_with_points(
// CHECK-SAME:      %[[LHS:.*]]: memref<?x?xf32>, %[[RHS:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[OUT:.*]]: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM_0:.*]] = memref.dim %[[LHS]], %[[C0]] : memref<?x?xf32>
// CHECK:         %[[DIM_1:.*]] = memref.dim %[[LHS]], %[[C1]] : memref<?x?xf32>

// CHECK:         gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:        to (%[[DIM_0]], %[[DIM_1]]) step (%[[C1]], %[[C1]]) {

// CHECK-DAG:       %[[RHS_EL:.*]] = memref.load %[[RHS]][%[[I]], %[[J]]]
// CHECK-DAG:       %[[LHS_EL:.*]] = memref.load %[[LHS]][%[[I]], %[[J]]]
// CHECK:           %[[ADD_EL:.*]] = arith.addf %[[LHS_EL]], %[[RHS_EL]] : f32
// CHECK:           memref.store %[[ADD_EL]], %[[OUT]][%[[I]], %[[J]]]
// CHECK:           gml_st.set_yield
// CHECK:         return %[[OUT]] : memref<?x?xf32>

// -----

func.func @materialize_and_yield_with_constants(
    %in: tensor<8x2xf32>, %out: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %0 = gml_st.space [8, 2] : !gml_st.tile<8x2>
  %1 = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c8, %c2) step (%c1, %c1) {
    %2 = gml_st.tile %0 [%i, %j] [1, 1] [1, 1]
      : !gml_st.tile<8x2> to !gml_st.tile<1x1>
    %3 = gml_st.materialize %in[%2]
      : tensor<8x2xf32>[!gml_st.tile<1x1>] to f32
    %4 = math.absf %3: f32
    gml_st.set_yield %4 into %out[%2]
      : f32 into tensor<8x2xf32>[!gml_st.tile<1x1>]
  } : tensor<8x2xf32>
  return %1 : tensor<8x2xf32>
}
// CHECK-LABEL: func @materialize_and_yield_with_constants
// CHECK-SAME:      %[[IN:.*]]: memref<8x2xf32>, %[[OUT:.*]]: memref<8x2xf32>)

// CHECK:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-NEXT:    %[[ELEM:.*]] = memref.load %[[IN]][%[[I]], %[[J]]]
// CHECK-NEXT:    %[[ABS:.*]] = math.absf %[[ELEM]] : f32
// CHECK-NEXT:    memref.store %[[ABS]], %[[OUT]][%[[I]], %[[J]]]
// CHECK-NEXT:    gml_st.set_yield
