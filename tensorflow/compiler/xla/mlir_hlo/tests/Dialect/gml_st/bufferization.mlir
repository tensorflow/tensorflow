// RUN: mlir-hlo-opt %s -empty-tensor-to-alloc-tensor \
// RUN: -hlo-one-shot-bufferize -canonicalize -cse -canonicalize \
// RUN: -split-input-file | FileCheck %s

func.func @set_tile(%input: tensor<?x?xf32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %slice = tensor.extract_slice %input[0, 1][2, 4][1, 1]
    : tensor<?x?xf32> to tensor<2x4xf32>

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
                               %out : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim_0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>

  %result = gml_st.parallel (%i, %j) = (%c0, %c0) to (%dim_0, %dim_1)
      step (%c4, %c1) outs (%out_ = %out: tensor<?x?xf32>) {
    %7 = arith.addi %i, %c4 : index
    %8 = arith.cmpi sgt, %7, %dim_0 : index
    %9 = arith.subi %dim_0, %i : index
    %size_0 = arith.select %8, %9, %c4 : index

    %lhs_tile = tensor.extract_slice %lhs[%i, %j] [%size_0, 1] [1, 1]
      : tensor<?x?xf32> to tensor<?x1xf32>
    %rhs_tile = tensor.extract_slice %rhs[%i, %j] [%size_0, 1] [1, 1]
      : tensor<?x?xf32> to tensor<?x1xf32>
    %init_tile = tensor.extract_slice %out_[%i, %j] [%size_0, 1] [1, 1]
      : tensor<?x?xf32> to tensor<?x1xf32>
    %sum = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs_tile, %rhs_tile : tensor<?x1xf32>, tensor<?x1xf32>)
        outs(%init_tile : tensor<?x1xf32>) {
      ^bb0(%l: f32, %r: f32, %o: f32):
        %add = arith.addf %l, %r : f32
        linalg.yield %add : f32
      } -> tensor<?x1xf32>
    %tile = gml_st.tile [%i, %j] [%size_0, 1] [1, 1] : !gml_st.tile<?x1>
    gml_st.set_yield %sum into %out_[%tile]
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

  %1 = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c8, %c2) step (%c1, %c1)
      outs(%out_ = %out: tensor<8x2xf32>) {
    %2 = tensor.extract_slice %in[%i, %j] [1, 1] [1, 1]
      : tensor<8x2xf32> to tensor<1x1xf32>
    %3 = tensor.extract %2[%c0, %c0] : tensor<1x1xf32>
    %4 = math.absf %3: f32
    %5 = gml_st.tile [%i, %j] [1, 1] [1, 1] : !gml_st.tile<1x1>
    gml_st.set_yield %4 into %out_[%5]
      : f32 into tensor<8x2xf32>[!gml_st.tile<1x1>]
  } : tensor<8x2xf32>
  return %1 : tensor<8x2xf32>
}
// CHECK-LABEL: func @materialize_and_yield_with_constants
// CHECK-SAME:      %[[IN:.*]]: memref<8x2xf32>, %[[OUT:.*]]: memref<8x2xf32>)

// CHECK:       gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-NEXT:    %[[SLICE:.*]] = memref.subview %[[IN]][%[[I]], %[[J]]]
// CHECK-NEXT:    %[[ELEM:.*]] = memref.load %[[SLICE]]
// CHECK-NEXT:    %[[ABS:.*]] = math.absf %[[ELEM]] : f32
// CHECK-NEXT:    memref.store %[[ABS]], %[[OUT]][%[[I]], %[[J]]]
// CHECK-NEXT:    gml_st.set_yield

// -----

func.func @parallel_with_vector(%in: vector<8xf32>, %out : vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %result = gml_st.parallel (%i) = (%c0) to (%c8) step (%c4)
      outs (%out_ = %out: vector<8xf32>) {
    %in_tile = gml_st.materialize %in[%i] [4] [1]
      : vector<8xf32> to vector<4xf32>
    %neg = arith.negf %in_tile : vector<4xf32>
    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %neg into %out_[%tile]
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

func.func @nested_parallel_with_vector(%out : tensor<?x32xf32>)
    -> tensor<?x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %dim_0 = tensor.dim %out, %c0 : tensor<?x32xf32>

  %result = gml_st.parallel (%i) = (%c0) to (%dim_0) step (%c1)
      outs (%out_ = %out: tensor<?x32xf32>) {
    %out_tile = tensor.extract_slice %out_[%i, 0] [1, 32] [1, 1]
      : tensor<?x32xf32> to tensor<1x32xf32>
    %out_vec = vector.transfer_read %out_tile[%c0, %c0], %cst
      {in_bounds = [true, true]}: tensor<1x32xf32>, vector<1x32xf32>

    %result_vec = gml_st.parallel (%j) = (%c0) to (%c32) step (%c4)
      outs (%vec_out_ = %out_vec: vector<1x32xf32>) {
      %inner_tile = gml_st.materialize %vec_out_[0, %j] [1, 4] [1, 1]
        : vector<1x32xf32> to vector<1x4xf32>
      %vtile = gml_st.tile [0, %j] [1, 4] [1, 1] : !gml_st.tile<1x4>
      gml_st.set_yield %inner_tile into %vec_out_[%vtile]
        : vector<1x4xf32> into vector<1x32xf32>[!gml_st.tile<1x4>]
    } : vector<1x32xf32>

    %result = vector.transfer_write %result_vec, %out_tile[%c0, %c0]
      {in_bounds = [true, true]} : vector<1x32xf32>, tensor<1x32xf32>
    %tile = gml_st.tile [%i, 0] [1, 32] [1, 1] : !gml_st.tile<1x32>
    gml_st.set_yield %result into %out_[%tile]
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
// CHECK-SAME:          outs (%[[VEC_OUT_:.*]] = %[[INITVEC]]:
// CHECK:             gml_st.materialize %[[VEC_OUT_]]
// CHECK:             gml_st.set_yield
// CHECK-SAME:          vector<1x4xf32> into vector<1x32xf32>[!gml_st.tile<1x4>]
// CHECK:           vector.transfer_write %[[RESVEC]], %[[INITTILE]]
// CHECK-SAME:         vector<1x32xf32>, memref<1x32xf32
// CHECK:         return %[[INIT]] : memref<?x32xf32>

// -----

func.func @same_enclosing_repetitive_region(%2: tensor<320xf32>,
                                            %3: tensor<320x10240xf32>)
  -> tensor<320xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -0.000000e+00 : f32
  %c320 = arith.constant 320 : index
  %4 = gml_st.parallel (%i) = (%c0) to (%c320) step (%c1)
      outs(%arg1 = %2: tensor<320xf32>) {
    %5 = tensor.extract_slice %3[%i, 0] [1, 10240] [1, 1]  : tensor<320x10240xf32> to tensor<1x10240xf32>
    %6 = tensor.extract_slice %arg1[%i] [1] [1] : tensor<320xf32> to tensor<1xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1xf32>) -> tensor<1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>

    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %8 into %arg1[%tile]
      : tensor<1xf32> into tensor<320xf32>[!gml_st.tile<1>]
  } : tensor<320xf32>
  return %4 : tensor<320xf32>
}
// CHECK-LABEL: @same_enclosing_repetitive_region
// CHECK-NOT: memref.alloc

// -----

// CHECK-LABEL: func @gml_st_parallel_private_var(
//  CHECK-SAME:     %[[t:.*]]: memref<10xf32
func.func @gml_st_parallel_private_var(%t: tensor<10xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index

  // A copy is inserted for the uses of %t in the loop.
  // CHECK: %[[t_copy:.*]] = memref.alloc() {{.*}} : memref<10xf32>
  // CHECK: memref.copy %[[t]], %[[t_copy]]

  // CHECK: gml_st.parallel

  // Load from the copy and store into the shared output.
  // CHECK:   %[[subview:.*]] = memref.subview %[[t]]
  // CHECK:   memref.load %[[t_copy]]
  // CHECK:   memref.store %{{.*}}, %[[subview]]
  %0 = gml_st.parallel (%tid) = (%c0) to (%c2) step (%c1)
      outs(%o = %t: tensor<10xf32>) {
    %offset = arith.muli %c5, %tid : index
    %slice = tensor.extract_slice %o[%offset] [5] [1]
        : tensor<10xf32> to tensor<5xf32>
    %r2 = tensor.extract %t[%tid] : tensor<10xf32>
    %i = tensor.insert %r2 into %slice[%c2] : tensor<5xf32>

    %tile = gml_st.tile [%offset][5][1] : !gml_st.tile<5>
    gml_st.set_yield %i into %o[%tile]
      : tensor<5xf32> into tensor<10xf32>[!gml_st.tile<5>]
  } : tensor<10xf32>

  %r = tensor.extract %0[%c2] : tensor<10xf32>
  return %r : f32
}

