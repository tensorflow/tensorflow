// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 32 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=3 -vector-size=16 | FileCheck %s 

// Capture permutation maps used in vectorization.
// CHECK-DAG: #[[map_proj_d0d1_d0d1:map[0-9]+]] = (d0, d1) -> (d0, d1)

// vector<3x32xf32> -> vector<3x16xf32>
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 16)
mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  // 2x unroll (jammed by construction).
  // CHECK: for %i0 = 0 to %arg0 step 3 {
  // CHECK:   for %i1 = 0 to %arg1 step 32 {
  // CHECK:     %cst_1 = constant splat<vector<3x16xf32>, 1.000000e+00> : vector<3x16xf32>
  // CHECK:     %cst_2 = constant splat<vector<3x16xf32>, 1.000000e+00> : vector<3x16xf32>
  // CHECK:     %3 = affine_apply #map0(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_1, %0, %3#0, %3#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  // CHECK:     %4 = affine_apply #map1(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_2, %0, %4#0, %4#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  //
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  // 2x unroll (jammed by construction).
  // CHECK: for %i2 = 0 to %arg0 step 3 {
  // CHECK:   for %i3 = 0 to %arg1 step 32 {
  // CHECK:     %cst_3 = constant splat<vector<3x16xf32>, 2.000000e+00> : vector<3x16xf32>
  // CHECK:     %cst_4 = constant splat<vector<3x16xf32>, 2.000000e+00> : vector<3x16xf32>
  // CHECK:     %5 = affine_apply #map0(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_3, %1, %5#0, %5#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  // CHECK:     %6 = affine_apply #map1(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_4, %1, %6#0, %6#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  //
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  // 2x unroll (jammed by construction).
  // CHECK: for %i4 = 0 to %arg0 step 3 {
  // CHECK:   for %i5 = 0 to %arg1 step 32 {
  // CHECK:     %7 = affine_apply #map0(%i4, %i5)
  // CHECK:     %8 = vector_transfer_read %0, %7#0, %7#1 {permutation_map: #map0} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
  // CHECK:     %9 = affine_apply #map1(%i4, %i5)
  // CHECK:     %10 = vector_transfer_read %0, %9#0, %9#1 {permutation_map: #map0} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
  // CHECK:     %11 = affine_apply #map0(%i4, %i5)
  // CHECK:     %12 = vector_transfer_read %1, %11#0, %11#1 {permutation_map: #map0} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
  // CHECK:     %13 = affine_apply #map1(%i4, %i5)
  // CHECK:     %14 = vector_transfer_read %1, %13#0, %13#1 {permutation_map: #map0} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
  // CHECK:     %15 = addf %8, %12 : vector<3x16xf32>
  // CHECK:     %16 = addf %10, %14 : vector<3x16xf32>
  // CHECK:     %17 = affine_apply #map0(%i4, %i5)
  // CHECK:     vector_transfer_write %15, %2, %17#0, %17#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  // CHECK:     %18 = affine_apply #map1(%i4, %i5)
  // CHECK:     vector_transfer_write %16, %2, %18#0, %18#1 {permutation_map: #map0} : vector<3x16xf32>, memref<?x?xf32>, index, index
  //
  for %i4 = 0 to %M {
    for %i5 = 0 to %N {
      %a5 = load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = addf %a5, %b5 : f32
      store %s5, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = constant 7 : index
  %c42 = constant 42 : index
  %res = load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}
