// Test vectorization of gml_st.parallel and gml_st.for loops.
// RUN: mlir-hlo-opt %s --split-input-file --vectorize-gml-st-loops \
// RUN: | FileCheck %s

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: @parallel_with_tiles(
func.func @parallel_with_tiles(
    %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)
    -> memref<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  gml_st.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c4, %c1) {
    %6 = memref.subview %arg2[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    %7 = memref.subview %arg1[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    %8 = memref.subview %arg0[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    linalg.map
            ins(%8, %7 : memref<4x1xf32, #map0>, memref<4x1xf32, #map0>)
            outs(%6 : memref<4x1xf32, #map0>)
    (%arg5: f32, %arg6: f32) {
      %9 = arith.addf %arg5, %arg6 : f32
      linalg.yield %9 : f32
    }
    gml_st.set_yield
  }
  func.return %arg2 : memref<?x?xf32>
}
// CHECK-NOT: linalg.map
// CHECK: %[[LHS:.*]] = vector.transfer_read {{%.*}}[%c0, %c0]
// CHECK: %[[RHS:.*]] = vector.transfer_read {{%.*}}[%c0, %c0]
// CHECK: %[[ADD:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<4x1xf32>
// CHECK: vector.transfer_write %[[ADD]], {{%.*}}[%c0, %c0]

// -----

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: @for_with_tiles(
func.func @for_with_tiles(
    %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)
    -> memref<?x?xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  gml_st.for (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c4, %c1) {
    %6 = memref.subview %arg2[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    %7 = memref.subview %arg1[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    %8 = memref.subview %arg0[%arg3, %arg4] [4, 1] [1, 1]
      : memref<?x?xf32> to memref<4x1xf32, #map0>
    linalg.map
            ins(%8, %7 : memref<4x1xf32, #map0>, memref<4x1xf32, #map0>)
            outs(%6 : memref<4x1xf32, #map0>)
    (%arg5: f32, %arg6: f32) {
      %9 = arith.addf %arg5, %arg6 : f32
      linalg.yield %9 : f32
    }
    gml_st.set_yield
  }
  func.return %arg2 : memref<?x?xf32>
}
// CHECK-NOT: linalg.map
// CHECK: %[[LHS:.*]] = vector.transfer_read {{%.*}}[%c0, %c0]
// CHECK: %[[RHS:.*]] = vector.transfer_read {{%.*}}[%c0, %c0]
// CHECK: %[[ADD:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<4x1xf32>
// CHECK: vector.transfer_write %[[ADD]], {{%.*}}[%c0, %c0]

// -----

// CHECK-LABEL: @parallel_on_tensor(
// CHECK: {{%.*}}: tensor<?xf32>, {{%.*}}: tensor<?xf32>, %[[ARG2:.*]]: tensor<?xf32>)
func.func @parallel_on_tensor(
    %arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>)
    -> tensor<?xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %2 = gml_st.parallel (%i) = (%c0) to (%0) step (%c4) {
    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    %6 = gml_st.materialize %arg0[%tile]
      : tensor<?xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %7 = gml_st.materialize %arg1[%tile]
      : tensor<?xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %8 = gml_st.materialize %arg2[%tile]
      : tensor<?xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %9 = linalg.map
                        ins(%6, %7 : tensor<4xf32>, tensor<4xf32>)
                        outs(%8 : tensor<4xf32>)
    (%arg5: f32, %arg6: f32) {
      %10 = arith.addf %arg5, %arg6 : f32
      linalg.yield %10 : f32
    }
    gml_st.set_yield %9 into %arg2[%tile]
      : tensor<4xf32> into tensor<?xf32>[!gml_st.tile<4>]
  } : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}
// CHECK-NOT: linalg.map
// CHECK: gml_st.parallel (%[[ITER:.*]]) = (%c0)
// CHECK: %[[LHS:.*]] = vector.transfer_read {{%.*}}[%c0]
// CHECK: %[[RHS:.*]] = vector.transfer_read {{%.*}}[%c0]
// CHECK: %[[ADD:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<4xf32>
// CHECK: vector.transfer_write %[[ADD]], %[[ARG2]][%[[ITER]]]

// -----

// CHECK-LABEL: @single_element_tensor_to_element(
// CHECK-SAME: %[[IN:.*]]: vector<1xf32>
func.func @single_element_tensor_to_element(%in : vector<1xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1xf32>
  %r = vector.transfer_write %in, %empty[%c0] {in_bounds = [true]}
    : vector<1xf32>, tensor<1xf32>
  %v = tensor.extract %r[%c0] : tensor<1xf32>
  return %v : f32
}
// CHECK: %[[RESULT:.*]] = vector.extract %[[IN]][0]
// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: @zero_dim_element_tensor_to_element(
// CHECK-SAME: %[[IN:.*]]: vector<f32>
func.func @zero_dim_element_tensor_to_element(%in : vector<f32>) -> f32 {
  %pad = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<f32>
  %r = vector.transfer_write %in, %empty[] {in_bounds = []}
    : vector<f32>, tensor<f32>
  %v = tensor.extract %r[] : tensor<f32>
  return %v : f32
}
// CHECK: %[[RESULT:.*]] = vector.extractelement %[[IN]][]
// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: @read_of_empty_float_to_constant(
func.func @read_of_empty_float_to_constant(%pad : f32) -> vector<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %c0 = arith.constant 0 : index
  %r = vector.transfer_read %empty[%c0], %pad {in_bounds = [true]}
    : tensor<32xf32>, vector<32xf32>
  return %r : vector<32xf32>
}
// CHECK: %[[RESULT:.*]] = arith.constant dense<0x7FC00000> : vector<32xf32>
// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: @read_of_empty_int_to_constant(
func.func @read_of_empty_int_to_constant(%pad : i8) -> vector<32xi8> {
  %empty = tensor.empty() : tensor<32xi8>
  %c0 = arith.constant 0 : index
  %r = vector.transfer_read %empty[%c0], %pad {in_bounds = [true]}
    : tensor<32xi8>, vector<32xi8>
  return %r : vector<32xi8>
}
// CHECK: %[[RESULT:.*]] = arith.constant dense<0> : vector<32xi8>
// CHECK: return %[[RESULT]]
// -----

// CHECK-LABEL: @materialize_scalar_from_0D_vector(
// CHECK-SAME: %[[V:.*]]: vector<f32>
func.func @materialize_scalar_from_0D_vector(%v : vector<f32>) -> f32 {
  %tile = gml_st.tile [] [] [] : !gml_st.tile<>
  %r = gml_st.materialize %v[%tile] : vector<f32>[!gml_st.tile<>] to f32
  return %r : f32
}
// CHECK: %[[R:.*]] = vector.extractelement %[[V]][]
// CHECK: return %[[R]]

// -----

// CHECK-LABEL: @materialize_scalar_from_single_element_vector(
// CHECK-SAME: %[[V:.*]]: vector<1x1xf32>
func.func @materialize_scalar_from_single_element_vector(
    %v : vector<1x1xf32>) -> f32 {
  %tile = gml_st.tile [0, 0] [1, 1] [1, 1] : !gml_st.tile<1x1>
  %r = gml_st.materialize %v[%tile] : vector<1x1xf32>[!gml_st.tile<1x1>] to f32
  return %r : f32
}
// CHECK: %[[R:.*]] = vector.extract %[[V]][0, 0]
// CHECK: return %[[R]]


// -----

// CHECK-LABEL: @set_yield_scalar_into_vector(
// CHECK-SAME: %[[F:.*]]: f32, %[[V:.*]]: vector<1x1xf32>)
func.func @set_yield_scalar_into_vector(
  %f: f32, %v: vector<1x1xf32>) {
  %tile = gml_st.tile [0, 0] [1, 1] [1, 1] : !gml_st.tile<1x1>
  gml_st.set_yield %f into %v[%tile]
    : f32 into vector<1x1xf32>[!gml_st.tile<1x1>]
}
// CHECK: %[[R:.*]] = vector.insert %[[F]], %[[V]] [0, 0]
// CHECK: gml_st.set_yield %[[R]] into %[[V]]

// -----

func.func @fold_identity_materialize(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>)
                  -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %6 = gml_st.for (%arg4) = (%c0) to (%c16) step (%c8) outs (%arg5 = %0: tensor<8x8xf32>) {
    %18 = gml_st.tile [%c0, %arg4] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %19 = gml_st.materialize %arg0[%18] : tensor<8x16xf32>[!gml_st.tile<8x8>] to tensor<8x8xf32>
    %20 = gml_st.tile [%arg4, %c0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %21 = gml_st.materialize %arg1[%20] : tensor<16x8xf32>[!gml_st.tile<8x8>] to tensor<8x8xf32>
    %22 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %23 = gml_st.materialize %arg5[%22] : tensor<8x8xf32>[!gml_st.tile<8x8>] to tensor<8x8xf32>
    %28 = linalg.fill ins(%cst_0 : f32) outs(%23 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %27 = gml_st.tile [0, 1] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %29 = gml_st.materialize %28[%27] : tensor<8x8xf32>[!gml_st.tile<8x8>] to tensor<8x8xf32>
    gml_st.set_yield %29 into %arg5[%22] : tensor<8x8xf32> into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// CHECK-LABEL: func @fold_identity_materialize(

// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty

// CHECK:         gml_st.for {{.*}} outs (%[[ARG:.*]] = %[[INIT]]
// CHECK:           %[[WRITE:.*]] = vector.transfer_write %[[CST]], %[[ARG]]
// CHECK:           %[[TILE:.*]] = gml_st.tile [0, 1] [8, 8] [1, 1]
// CHECK:           %[[MAT:.*]] = gml_st.materialize %[[WRITE]][%[[TILE]]]
// CHECK:           gml_st.set_yield %[[MAT]] into %[[ARG]]
