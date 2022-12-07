// RUN: mlir-hlo-opt %s -test-gml-st-bufferization -canonicalize -cse \
// RUN:   -split-input-file | FileCheck %s

func.func @set_tile(%input: tensor<?x?xf32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %tile = gml_st.tile [0, 1][2, 4][1, 1] : !gml_st.tile<2x4>

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

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @parallel_with_tiles(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                               %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim_0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>

  %result = gml_st.parallel (%i, %j) = (%c0, %c0) to (%dim_0, %dim_1) step (%c4, %c1) {
    %7 = arith.addi %i, %c4 : index
    %8 = arith.cmpi sgt, %7, %dim_0 : index
    %9 = arith.subi %dim_0, %i : index
    %size_0 = arith.select %8, %9, %c4 : index

    %tile = gml_st.tile [%i, %j] [%size_0, 1] [1, 1] : !gml_st.tile<?x1>
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

func.func @materialize_and_yield_with_constants(
    %in: tensor<8x2xf32>, %out: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %1 = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c8, %c2) step (%c1, %c1) {
    %2 = gml_st.tile [%i, %j] [1, 1] [1, 1] : !gml_st.tile<1x1>
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

// -----
func.func @parallel_with_vector(%in: vector<8xf32>, %init : vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %result = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4) {
    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    %in_tile = gml_st.materialize %in[%tile]
      : vector<8xf32>[!gml_st.tile<4>] to vector<4xf32>
    %neg = arith.negf %in_tile : vector<4xf32>
    gml_st.set_yield %neg into %init[%tile]
      : vector<4xf32> into vector<8xf32>[!gml_st.tile<4>]
  } : vector<8xf32>

  return %result : vector<8xf32>
}
// Bufferization should leave the parallel unchanged.
// CHECK-LABEL: func @parallel_with_vector(
// CHECK:       %[[RESULT:.*]] = gml_st.parallel
// CHECK-NOT:   memref
// CHECK-NOT:   vector.transfer_{{read|write}}
// CHECK:       return %[[RESULT]] : vector<8xf32>

// -----

func.func @nested_parallel_with_vector(%init : tensor<?x32xf32>)
    -> tensor<?x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %dim_0 = tensor.dim %init, %c0 : tensor<?x32xf32>

  %result = gml_st.parallel (%i) = (%c0) to (%dim_0) step (%c1) {
    %tile = gml_st.tile [%i, 0] [1, 32] [1, 1] : !gml_st.tile<1x32>
    %init_tile = gml_st.materialize %init[%tile]
      : tensor<?x32xf32>[!gml_st.tile<1x32>] to tensor<1x32xf32>
    %init_vec = vector.transfer_read %init_tile[%c0, %c0], %cst
      {in_bounds = [true, true]}: tensor<1x32xf32>, vector<1x32xf32>

    %result_vec = gml_st.parallel (%j) = (%c0) to (%c32) step (%c4) {
      %vtile = gml_st.tile [0, %j] [1, 4] [1, 1] : !gml_st.tile<1x4>
      %inner_tile = gml_st.materialize %init_vec[%vtile]
        : vector<1x32xf32>[!gml_st.tile<1x4>] to vector<1x4xf32>
      gml_st.set_yield %inner_tile into %init_vec[%vtile]
        : vector<1x4xf32> into vector<1x32xf32>[!gml_st.tile<1x4>]
    } : vector<1x32xf32>

    %result = vector.transfer_write %result_vec, %init_tile[%c0, %c0]
      {in_bounds = [true, true]} : vector<1x32xf32>, tensor<1x32xf32>
    gml_st.set_yield %result into %init[%tile]
      : tensor<1x32xf32> into tensor<?x32xf32>[!gml_st.tile<1x32>]
  } : tensor<?x32xf32>

  return %result : tensor<?x32xf32>
}
// The outer parallel should be bufferized, while the inner one should be left
// unchanged.
// CHECK-LABEL: func @nested_parallel_with_vector(
// CHECK-SAME:  %[[INIT:.*]]: memref<?x32xf32>) -> memref<?x32xf32> {
// CHECK:         gml_st.parallel (%[[I:.*]]) =
// CHECK-DAG:       %[[INITTILE:.*]] = memref.subview %[[INIT]][%[[I]], 0]
// CHECK-SAME:        memref<?x32xf32> to memref<1x32xf32
// CHECK-DAG:       %[[INITVEC:.*]] = vector.transfer_read %[[INITTILE]]
// CHECK-SAME:        memref<1x32xf32, {{.*}}>, vector<1x32xf32>
// CHECK:           %[[RESVEC:.*]] = gml_st.parallel
// CHECK:             gml_st.materialize %[[INITVEC]]
// CHECK:             gml_st.set_yield
// CHECK-SAME:          vector<1x4xf32> into vector<1x32xf32>[!gml_st.tile<1x4>]
// CHECK:           vector.transfer_write %[[RESVEC]], %[[INITTILE]]
// CHECK-SAME:         vector<1x32xf32>, memref<1x32xf32
// CHECK:         return %[[INIT]] : memref<?x32xf32>


// -----

func.func @scalarized_reduction(%arg: tensor<1x?xf32>) -> tensor<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1xf32>
  %fill = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<1xf32>) -> tensor<1xf32>

  %dim = tensor.dim %arg, %c1 : tensor<1x?xf32>
  %result = gml_st.for (%i) = (%c0) to (%dim) step (%c1)
      outs (%out = %fill: tensor<1xf32>) {
    %tile = gml_st.tile [0, %i] [1, 1] [1, 1] : !gml_st.tile<1x1>
    %elem = gml_st.materialize %arg[%tile]
      : tensor<1x?xf32>[!gml_st.tile<1x1>] to f32

    %extracted = tensor.extract %out[%c0] : tensor<1xf32>
    %sum = arith.addf %extracted, %elem : f32

    %tile1 = gml_st.tile [0] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %sum into %out[%tile1]
      : f32 into tensor<1xf32>[!gml_st.tile<1>]
  } : tensor<1xf32>
  return %result : tensor<1xf32>
}
// CHECK-LABEL: func.func @scalarized_reduction(
// CHECK-SAME:      %[[ARG:.*]]: memref<1x?xf32>) -> memref<1xf32> {

// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index

// CHECK:       %[[ALLOC:.*]] = memref.alloc()
// CHECK:       linalg.fill ins(%{{.*}}: f32) outs(%[[ALLOC]] : memref<1xf32>)
// CHECK:       %[[DIM:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<1x?xf32>

// CHECK-NEXT:  gml_st.for (%[[I:.*]]) = (%[[C0]]) to (%[[DIM]]) step (%[[C1]]) {
// CHECK-NEXT:    %[[ARG_ELEM:.*]] = memref.load %[[ARG]][%[[C0]], %[[I]]]
// CHECK-NEXT:    %[[ACC:.*]] = memref.load %[[ALLOC]][%[[C0]]] : memref<1xf32>
// CHECK-NEXT:    %[[SUM:.*]] = arith.addf %[[ACC]], %[[ARG_ELEM]] : f32
// CHECK-NEXT:    memref.store %[[SUM]], %[[ALLOC]][%[[C0]]] : memref<1xf32>
// CHECK-NEXT:    gml_st.set_yield
// CHECK-NEXT:  }
// CHECK:       return %[[ALLOC]] : memref<1xf32>

// -----

func.func @matmul(%lhs: tensor<128x16xf32>,
                  %rhs: tensor<16x64xf32>,
                  %out: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %matmul = gml_st.parallel (%i, %j)
      = (%c0, %c0) to (%c128, %c64) step (%c8, %c4) {
    %lhs_tile = gml_st.tile [%i, 0] [8, 16] [1, 1] : !gml_st.tile<8x16>
    %lhs_sub = gml_st.materialize %lhs[%lhs_tile]
      : tensor<128x16xf32>[!gml_st.tile<8x16>] to tensor<8x16xf32>
    %rhs_tile = gml_st.tile [0, %j] [16, 4] [1, 1] : !gml_st.tile<16x4>
    %rhs_sub = gml_st.materialize %rhs[%rhs_tile]
      : tensor<16x64xf32>[!gml_st.tile<16x4>] to tensor<16x4xf32>
    %out_tile = gml_st.tile [%i, %j] [8, 4] [1, 1] : !gml_st.tile<8x4>
    %out_sub = gml_st.materialize %out[%out_tile]
      : tensor<128x64xf32>[!gml_st.tile<8x4>] to tensor<8x4xf32>

    %mat_sub = gml_st.for (%k) = (%c0) to (%c16) step (%c2)
        outs (%out_sub_ = %out_sub: tensor<8x4xf32>) {
      %lhs_tile2 = gml_st.tile [0, %k] [8, 2] [1, 1] : !gml_st.tile<8x2>
      %lhs_sub2 = gml_st.materialize %lhs_sub[%lhs_tile2]
        : tensor<8x16xf32>[!gml_st.tile<8x2>] to tensor<8x2xf32>
      %rhs_tile2 = gml_st.tile [%k, 0] [2, 4] [1, 1] : !gml_st.tile<2x4>
      %rhs_sub2 = gml_st.materialize %rhs_sub[%rhs_tile2]
        : tensor<16x4xf32>[!gml_st.tile<2x4>] to tensor<2x4xf32>
      %out_tile2 = gml_st.tile [0, 0] [8, 4] [1, 1] : !gml_st.tile<8x4>
      %out_sub2 = gml_st.materialize %out_sub_[%out_tile2]
        : tensor<8x4xf32>[!gml_st.tile<8x4>] to tensor<8x4xf32>

      %mat_sub2 = linalg.matmul
        ins(%lhs_sub2, %rhs_sub2 : tensor<8x2xf32>, tensor<2x4xf32>)
        outs(%out_sub2 : tensor<8x4xf32>) -> tensor<8x4xf32>

      gml_st.set_yield %mat_sub2 into %out_sub_[%out_tile2]
        : tensor<8x4xf32> into tensor<8x4xf32>[!gml_st.tile<8x4>]
    } : tensor<8x4xf32>
    gml_st.set_yield %mat_sub into %out[%out_tile]
      : tensor<8x4xf32> into tensor<128x64xf32>[!gml_st.tile<8x4>]
  } : tensor<128x64xf32>
  return %matmul : tensor<128x64xf32>
}
// CHECK-LABEL: func.func @matmul
// CHECK-NOT:     alloc
// CHECK:         gml_st.parallel
// CHECK-3:         memref.subview
// CHECK-NOT:       alloc
// CHECK:           gml_st.for
// CHECK-4:           memref.subview
// CHECK-NOT:         alloc
// CHECK:             linalg.matmul
