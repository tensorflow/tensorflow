// RUN: mlir-hlo-opt %s --gml-st-rewrite-from-elements-ops \
// RUN: -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
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

  %result = scf.forall (%i, %j) = (%c0, %c0) to (%dim_0, %dim_1)
      step (%c4, %c1) shared_outs (%out_ = %out) -> (tensor<?x?xf32>) {
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
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sum into %out_[%i, %j] [%size_0, 1] [1, 1]
        : tensor<?x1xf32> into tensor<?x?xf32>
    }
  }
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

// CHECK:     scf.forall (%[[I:.*]], %[[J:.*]]) = (0, 0)
// CHECK-SAME:    to (%[[DIM_0]], %[[DIM_1]]) step (4, 1) {

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
// CHECK:     }
// CHECK: return %[[OUT]] : memref<?x?xf32>

// -----

func.func @materialize_and_yield_with_constants(
    %in: tensor<8x2xf32>, %out: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index

  %1 = scf.forall (%i, %j) = (%c0, %c0) to (%c8, %c2) step (%c1, %c1)
      shared_outs (%out_ = %out) -> (tensor<8x2xf32>) {
    %2 = tensor.extract_slice %in[%i, %j] [1, 1] [1, 1]
      : tensor<8x2xf32> to tensor<1x1xf32>
    %3 = tensor.extract %2[%c0, %c0] : tensor<1x1xf32>
    %4 = math.absf %3: f32
    %5 = tensor.from_elements %4 : tensor<f32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %out_[%i, %j] [1, 1] [1, 1]
        : tensor<f32> into tensor<8x2xf32>
    }
  }
  return %1 : tensor<8x2xf32>
}
// CHECK-LABEL: func @materialize_and_yield_with_constants
// CHECK-SAME:      %[[IN:.*]]: memref<8x2xf32>, %[[OUT:.*]]: memref<8x2xf32>)

// CHECK:       scf.forall (%[[I:.*]], %[[J:.*]]) in (8, 2)
// CHECK-NEXT:    %[[SLICE:.*]] = memref.subview %[[IN]][%[[I]], %[[J]]]
// CHECK-NEXT:    %[[ELEM:.*]] = memref.load %[[SLICE]]
// CHECK-NEXT:    %[[ABS:.*]] = math.absf %[[ELEM]] : f32
// CHECK-NEXT:    %[[OUT_SLICE:.*]] = memref.subview %[[OUT]]
// CHECK-SAME:      [%[[I]], %[[J]]] [1, 1] [1, 1]
// CHECK-NEXT:    memref.store %[[ABS]], %[[OUT_SLICE]][]

// -----

func.func @same_enclosing_repetitive_region(%2: tensor<320xf32>,
                                            %3: tensor<320x10240xf32>)
  -> tensor<320xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -0.000000e+00 : f32
  %c320 = arith.constant 320 : index
  %4 = scf.forall (%i) = (%c0) to (%c320) step (%c1)
      shared_outs(%arg1 = %2) -> (tensor<320xf32>) {
    %5 = tensor.extract_slice %3[%i, 0] [1, 10240] [1, 1]  : tensor<320x10240xf32> to tensor<1x10240xf32>
    %6 = tensor.extract_slice %arg1[%i] [1] [1] : tensor<320xf32> to tensor<1xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1xf32>) -> tensor<1xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg1[%i] [1] [1]
        : tensor<1xf32> into tensor<320xf32>
    }
  }
  return %4 : tensor<320xf32>
}
// CHECK-LABEL: @same_enclosing_repetitive_region
// CHECK-NOT: memref.alloc

// -----

// CHECK-LABEL: func @scf.forall_private_var(
//  CHECK-SAME:     %[[t:.*]]: memref<10xf32
func.func @scf.forall_private_var(%t: tensor<10xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index

  // A copy is inserted for the uses of %t in the loop.
  // CHECK: %[[t_copy:.*]] = memref.alloc() {{.*}} : memref<10xf32>
  // CHECK: memref.copy %[[t]], %[[t_copy]]

  // CHECK: scf.forall

  // Load from the copy and store into the shared output.
  // CHECK:   %[[subview:.*]] = memref.subview %[[t]]
  // CHECK:   memref.load %[[t_copy]]
  // CHECK:   memref.store %{{.*}}, %[[subview]]
  %0 = scf.forall (%tid) = (%c0) to (%c2) step (%c1)
      shared_outs (%o = %t) -> (tensor<10xf32>) {
    %offset = arith.muli %c5, %tid : index
    %slice = tensor.extract_slice %o[%offset] [5] [1]
        : tensor<10xf32> to tensor<5xf32>
    %r2 = tensor.extract %t[%tid] : tensor<10xf32>
    %i = tensor.insert %r2 into %slice[%c2] : tensor<5xf32>

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %i into %o[%offset][5][1]
        : tensor<5xf32> into tensor<10xf32>
    }
  }
  %r = tensor.extract %0[%c2] : tensor<10xf32>
  return %r : f32
}

// -----

func.func @gml_st_fusion(%arg0: tensor<?xf32>,
    %init: tensor<?xf32>) -> tensor<?xf32> {
  %0 = gml_st.fusion ins(%a0 = %arg0 : tensor<?xf32>)
                     inits(%in = %init : tensor<?xf32>) {
    %res = linalg.map { math.exp }
      ins(%a0 : tensor<?xf32>)
      outs(%in : tensor<?xf32>)
    gml_st.yield %res : tensor<?xf32>
  } : tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @gml_st_fusion
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>
// CHECK:         gml_st.fusion
// CHECK-SAME:        ins(%[[ARG0_:.*]] = %[[ARG0]]: memref<?xf32>)
// CHECK-SAME:        inits(%[[ARG1_:.*]] = %[[ARG1]]: memref<?xf32>)
// CHECK:           linalg.map { math.exp }
// CHECK-SAME:          ins(%[[ARG0_]] : memref<?xf32>)
// CHECK-SAME:          outs(%[[ARG1_]] : memref<?xf32>)
// CHECK:            gml_st.yield %[[ARG1_]] : memref<?xf32>
// CHECK:         return %[[ARG1]] : memref<?xf32>

// -----

func.func @gml_st_fusion_temp_tensor(
    %arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %init = tensor.empty(%dim0) : tensor<?xf32>
  %0 = gml_st.fusion ins(%arg0_ = %arg0 : tensor<?xf32>,
                         %arg1_ = %arg1 : tensor<?xf32>)
                     inits(%init_ = %init : tensor<?xf32>) {
    %c0_ = arith.constant 0 : index
    %dim0_ = tensor.dim %arg0_, %c0_ : tensor<?xf32>
    %temp = tensor.empty(%dim0_) : tensor<?xf32>
    %map0 = linalg.map { math.exp }
      ins(%arg0_ : tensor<?xf32>)
      outs(%temp : tensor<?xf32>)
    %map1 = linalg.map { arith.mulf }
      ins(%map0, %arg1_ : tensor<?xf32>, tensor<?xf32>)
      outs(%init_ : tensor<?xf32>)
    gml_st.yield %map1 : tensor<?xf32>
  } : tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL:  func @gml_st_fusion_temp_tensor
// CHECK-SAME:       (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>)
// CHECK:          %[[C0:.*]] = arith.constant 0 : index
// CHECK:          %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK:          %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
// CHECK:          gml_st.fusion
// CHECK-SAME:         ins(%[[ARG0_:.*]] = %[[ARG0]]: memref<?xf32>,
// CHECK-SAME:             %[[ARG1_:.*]] = %[[ARG1]]: memref<?xf32>)
// CHECK-SAME:         inits(%[[INIT_:.*]] = %[[ALLOC]]: memref<?xf32>)
// CHECK-DAG:        %[[C0_:.*]] = arith.constant 0 : index
// CHECK:            %[[DIM_:.*]] = memref.dim %[[ARG0_]], %[[C0_]]
// CHECK:            %[[ALLOC_:.*]] = memref.alloc(%[[DIM_]])
// CHECK:            linalg.map { math.exp }
// CHECK-SAME:         ins(%[[ARG0_]]
// CHECK-SAME:         outs(%[[ALLOC_]]
// CHECK:            linalg.map { arith.mulf }
// CHECK-SAME:         ins(%[[ALLOC_]], %[[ARG1_]]
// CHECK-SAME:         outs(%[[INIT_]]
// CHECK:            gml_st.yield %[[INIT_]] : memref<?xf32>
// CHECK:          return %[[ALLOC]] : memref<?xf32>

// -----

func.func @gml_st_fusion_scalar_scf_for(%arg0: tensor<?xi64>) -> tensor<i64> {
  %0 = tensor.empty() : tensor<i64>
  %1 = gml_st.fusion
         ins(%arg1 = %arg0: tensor<?xi64>)
         inits(%arg2 = %0: tensor<i64>) {
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xi64>
    %2 = scf.for %arg3 = %c0 to %dim step %c1
           iter_args(%arg4 = %c1_i64) -> (i64) {
      %extracted = tensor.extract %arg1[%arg3] : tensor<?xi64>
      %3 = arith.muli %arg4, %extracted : i64
      scf.yield %3 : i64
    }
    %from_elements = tensor.from_elements %2 : tensor<i64>
    gml_st.yield %from_elements : tensor<i64>
  } : tensor<i64>
  return %1 : tensor<i64>
}

// CHECK-LABEL:  func.func @gml_st_fusion_scalar_scf_for
// CHECK-SAME:       (%[[ARG0:.*]]: memref<?xi64>)
// CHECK:          %[[ALLOC:.*]] = memref.alloc()
// CHECK:          gml_st.fusion
// CHECK-SAME:         ins(%[[ARG0_:.*]] = %[[ARG0]]: memref<?xi64>)
// CHECK-SAME:         inits(%[[ALLOC_:.*]] = %[[ALLOC]]: memref<i64>)
// CHECK-DAG:        %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:            %[[DIM:.*]] = memref.dim %[[ARG0_]], %[[C0]]
// CHECK:            %[[FOR:.*]] = scf.for %[[ARG3:.*]] = %[[C0]] to %[[DIM]]
// CHECK-SAME:           step %[[C1]] iter_args(%[[ARG4:.*]] = %[[C1_I64]])
// CHECK:              %[[LOAD:.*]] = memref.load %[[ARG0_]][%[[ARG3]]]
// CHECK:              %[[MULI:.*]] = arith.muli %[[ARG4]], %[[LOAD]]
// CHECK:              scf.yield %[[MULI]] : i64
// CHECK:            %[[ALLOC_0:.*]] = memref.alloc()
// CHECK:            memref.store %[[FOR]], %[[ALLOC_0]][]
// CHECK:            memref.copy %[[ALLOC_0]], %[[ALLOC_]]
// CHECK:            gml_st.yield %[[ALLOC_]] : memref<i64>
// CHECK:          return %[[ALLOC]] : memref<i64>
