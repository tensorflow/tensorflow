// RUN: mlir-hlo-opt %s --vectorize-for-gpu --split-input-file |\
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_transfer_read_of_one_dim_expand_shape(
    %in: tensor<10xf32>) -> tensor<5xf32> {
  %c0 = arith.constant 0 : index
  %min_float = arith.constant dense<-3.402820e+38> : vector<5xf32>
  %zero_float = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<10xf32> into tensor<2x5xf32>
  %1 = tensor.empty() : tensor<5xf32>
  %2 = vector.transfer_read %0[%c0, %c0], %zero_float
    {in_bounds = [true, true], permutation_map = #map0}
    : tensor<2x5xf32>, vector<2x5xf32>
  %3 = vector.multi_reduction <maxf>, %2, %min_float [0]
    : vector<2x5xf32> to vector<5xf32>
  %4 = vector.transfer_write %3, %1[%c0] {in_bounds = [true]}
    : vector<5xf32>, tensor<5xf32>
  func.return %4 : tensor<5xf32>
}
// CHECK-LABEL: func @test_transfer_read_of_one_dim_expand_shape(
// CHECK-SAME: %[[IN:.*]]: tensor<10xf32>
// CHECK-DAG: %[[MIN_FLOAT:.*]] = arith.constant dense<-3.402820e+38> : vector<5xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ZERO_FLOAT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[INIT_TENSOR:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %[[IN]][%[[C0]]], %[[ZERO_FLOAT]] {in_bounds = [true]} : tensor<10xf32>, vector<10xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[TRANSFER_READ]] : vector<10xf32> to vector<2x5xf32>
// CHECK: %[[MULTI_REDUCTION:.*]] = vector.multi_reduction <maxf>, %[[SHAPE_CAST]], %[[MIN_FLOAT]] [0] : vector<2x5xf32> to vector<5xf32>
// CHECK: %[[TRANSFER_WRITE:.*]] = vector.transfer_write %[[MULTI_REDUCTION]], %[[INIT_TENSOR]][%[[C0]]] {in_bounds = [true]} : vector<5xf32>, tensor<5xf32>
// CHECK: return %[[TRANSFER_WRITE]] : tensor<5xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, 0)>
func.func @test_transfer_read_of_one_dim_expand_shape_different_shape(
    %in: tensor<1xf32>) -> tensor<18xf32> {
  %c0 = arith.constant 0 : index
  %min_float = arith.constant dense<-3.402820e+38> : vector<18xf32>
  %zero_float = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
  %1 = tensor.empty() : tensor<18xf32>
  %2 = vector.transfer_read %0[%c0, %c0], %zero_float
    {in_bounds = [true, true], permutation_map = #map0}
    : tensor<1x1xf32>, vector<1x18xf32>
  %3 = vector.multi_reduction <maxf>, %2, %min_float [0]
    : vector<1x18xf32> to vector<18xf32>
  %4 = vector.transfer_write %3, %1[%c0] {in_bounds = [true]}
    : vector<18xf32>, tensor<18xf32>
  func.return %4 : tensor<18xf32>
}
// CHECK-LABEL: func @test_transfer_read_of_one_dim_expand_shape_different_shape
// CHECK: %{{.*}} = tensor.expand_shape

// -----

func.func @do_not_vectorize_large_untiled_fill() -> tensor<2x1000xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x1000xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<2x1000xf32>) -> tensor<2x1000xf32>
  func.return %out : tensor<2x1000xf32>
}
// CHECK-LABEL: func @do_not_vectorize_large_untiled_fill
// CHECK: linalg.fill

// -----

func.func @vectorize_small_untiled_fill() -> tensor<128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<128xf32>
  %out = linalg.fill ins(%cst : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
  func.return %out : tensor<128xf32>
}
// CHECK-LABEL: func @vectorize_small_untiled_fill
// CHECK: vector.transfer_write

// -----

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
    linalg.map { arith.addf }
            ins(%8, %7 : memref<4x1xf32, #map0>, memref<4x1xf32, #map0>)
            outs(%6 : memref<4x1xf32, #map0>)
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
    linalg.map { arith.addf }
            ins(%8, %7 : memref<4x1xf32, #map0>, memref<4x1xf32, #map0>)
            outs(%6 : memref<4x1xf32, #map0>)
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
  %2 = gml_st.parallel (%i) = (%c0) to (%0) step (%c4)
      outs (%out_ = %arg2: tensor<?xf32>) {
    %6 = tensor.extract_slice %arg0[%i] [4] [1]
      : tensor<?xf32> to tensor<4xf32>
    %7 = tensor.extract_slice %arg1[%i] [4] [1]
      : tensor<?xf32> to tensor<4xf32>
    %8 = tensor.extract_slice %out_[%i] [4] [1]
      : tensor<?xf32> to tensor<4xf32>
    %9 = linalg.map { arith.addf }
           ins(%6, %7 : tensor<4xf32>, tensor<4xf32>)
           outs(%8 : tensor<4xf32>)
    %tile = gml_st.tile [%i] [4] [1] : !gml_st.tile<4>
    gml_st.set_yield %9 into %out_[%tile]
      : tensor<4xf32> into tensor<?xf32>[!gml_st.tile<4>]
  } : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}
// CHECK-NOT: linalg.map
// CHECK: gml_st.parallel (%[[ITER:.*]]) = (%[[C0:[a-z0-9]+]])
// CHECK: %[[LHS:.*]] = vector.transfer_read {{%[a-z0-9_]+}}[%[[C0]]]
// CHECK: %[[RHS:.*]] = vector.transfer_read {{%[a-z0-9_]+}}[%[[C0]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<4xf32>

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
  %r = gml_st.materialize %v[][][] : vector<f32> to f32
  return %r : f32
}
// CHECK: %[[R:.*]] = vector.extractelement %[[V]][]
// CHECK: return %[[R]]

// -----

// CHECK-LABEL: @materialize_scalar_from_single_element_vector(
// CHECK-SAME: %[[V:.*]]: vector<1x1xf32>
func.func @materialize_scalar_from_single_element_vector(
    %v : vector<1x1xf32>) -> f32 {
  %r = gml_st.materialize %v[0, 0] [1, 1] [1, 1]
    : vector<1x1xf32> to f32
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
    %19 = tensor.extract_slice %arg0[%c0, %arg4] [8, 8] [1, 1]  : tensor<8x16xf32> to tensor<8x8xf32>
    %21 = tensor.extract_slice %arg1[%arg4, %c0] [8, 8] [1, 1]  : tensor<16x8xf32> to tensor<8x8xf32>
    %23 = tensor.extract_slice %arg5[0, 0] [8, 8] [1, 1]  : tensor<8x8xf32> to tensor<8x8xf32>
    %28 = linalg.fill ins(%cst_0 : f32) outs(%23 : tensor<8x8xf32>) -> tensor<8x8xf32>
    %29 = tensor.extract_slice %28[0, 0] [8, 8] [1, 1]  : tensor<8x8xf32> to tensor<8x8xf32>
    %22 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    gml_st.set_yield %29 into %arg5[%22] : tensor<8x8xf32> into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// CHECK-LABEL: func @fold_identity_materialize(

// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty

// CHECK:         gml_st.for {{.*}} outs (%[[ARG:.*]] = %[[INIT]]
// CHECK:           %[[WRITE:.*]] = vector.transfer_write %[[CST]], %[[ARG]]
// CHECK:           %[[TILE:.*]] = gml_st.tile [0, 0] [8, 8] [1, 1]
// CHECK:           gml_st.set_yield %[[WRITE]] into %[[ARG]]
