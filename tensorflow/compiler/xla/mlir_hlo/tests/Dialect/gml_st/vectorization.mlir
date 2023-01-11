// RUN: mlir-hlo-opt %s --vectorize-gml-st-loops --split-input-file |\
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
