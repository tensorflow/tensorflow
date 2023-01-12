// RUN: tf-tfrt-opt -tf-jitrt-tile-transpose -split-input-file %s | FileCheck %s

// Make sure that transpose codegen passes only trigger on generic ops
// implementing a transpose operation.

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_2d(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%1, %0) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @transpose_2d(
// CHECK:           gml_st.loop
// CHECK:             linalg.generic
// CHECK:               linalg.yield
// CHECK:             gml_st.yield

// -----

#map0 = affine_map<(d0, d1) -> (d0, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @identity(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%1, %0) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @identity(
// CHECK-NOT:       gml_st.loop

// -----

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @transpose_add(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32>{
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%1, %0) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %add = arith.addf %arg1, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>
  func.return %3 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @transpose_add(
// CHECK-NOT:       gml_st.loop
