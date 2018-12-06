// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 16 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=8 | FileCheck %s -check-prefix=CHECK

// Capture permutation maps used in vectorization.
// CHECK-DAG: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)

// vector<3x16xf32> -> vector<8xf32>
// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (d0, d1)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 8)
// CHECK-DAG: [[MAP2:#.*]] = (d0, d1) -> (d0 + 1, d1)
// CHECK-DAG: [[MAP3:#.*]] = (d0, d1) -> (d0 + 1, d1 + 8)
// CHECK-DAG: [[MAP4:#.*]] = (d0, d1) -> (d0 + 2, d1)
// CHECK-DAG: [[MAP5:#.*]] = (d0, d1) -> (d0 + 2, d1 + 8)
mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i0 = 0 to %arg0 step 3 {
  // CHECK:   for %i1 = 0 to %arg1 step 16 {
  // CHECK:     %cst_1 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %cst_2 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %cst_3 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %cst_4 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %cst_5 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %cst_6 = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     %3 = affine_apply #map0(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_1, %0, %3#0, %3#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %4 = affine_apply #map2(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_2, %0, %4#0, %4#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %5 = affine_apply #map3(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_3, %0, %5#0, %5#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %6 = affine_apply #map4(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_4, %0, %6#0, %6#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %7 = affine_apply #map5(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_5, %0, %7#0, %7#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %8 = affine_apply #map6(%i0, %i1)
  // CHECK:     vector_transfer_write %cst_6, %0, %8#0, %8#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i2 = 0 to %arg0 step 3 {
  // CHECK:   for %i3 = 0 to %arg1 step 16 {
  // CHECK:     %cst_7 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %cst_8 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %cst_9 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %cst_10 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %cst_11 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %cst_12 = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     %9 = affine_apply #map0(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_7, %1, %9#0, %9#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %10 = affine_apply #map2(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_8, %1, %10#0, %10#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %11 = affine_apply #map3(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_9, %1, %11#0, %11#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %12 = affine_apply #map4(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_10, %1, %12#0, %12#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %13 = affine_apply #map5(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_11, %1, %13#0, %13#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %14 = affine_apply #map6(%i2, %i3)
  // CHECK:     vector_transfer_write %cst_12, %1, %14#0, %14#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      // CHECK does (3x4)x unrolling.
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i4 = 0 to %arg0 step 3 {
  // CHECK:   for %i5 = 0 to %arg1 step 16 {
  // CHECK:     %15 = affine_apply #map0(%i4, %i5)
  // CHECK:     %16 = vector_transfer_read %0, %15#0, %15#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %17 = affine_apply #map2(%i4, %i5)
  // CHECK:     %18 = vector_transfer_read %0, %17#0, %17#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %19 = affine_apply #map3(%i4, %i5)
  // CHECK:     %20 = vector_transfer_read %0, %19#0, %19#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %21 = affine_apply #map4(%i4, %i5)
  // CHECK:     %22 = vector_transfer_read %0, %21#0, %21#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %23 = affine_apply #map5(%i4, %i5)
  // CHECK:     %24 = vector_transfer_read %0, %23#0, %23#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %25 = affine_apply #map6(%i4, %i5)
  // CHECK:     %26 = vector_transfer_read %0, %25#0, %25#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %27 = affine_apply #map0(%i4, %i5)
  // CHECK:     %28 = vector_transfer_read %1, %27#0, %27#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %29 = affine_apply #map2(%i4, %i5)
  // CHECK:     %30 = vector_transfer_read %1, %29#0, %29#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %31 = affine_apply #map3(%i4, %i5)
  // CHECK:     %32 = vector_transfer_read %1, %31#0, %31#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %33 = affine_apply #map4(%i4, %i5)
  // CHECK:     %34 = vector_transfer_read %1, %33#0, %33#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %35 = affine_apply #map5(%i4, %i5)
  // CHECK:     %36 = vector_transfer_read %1, %35#0, %35#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %37 = affine_apply #map6(%i4, %i5)
  // CHECK:     %38 = vector_transfer_read %1, %37#0, %37#1 {permutation_map: #map1} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %39 = addf %16, %28 : vector<8xf32>
  // CHECK:     %40 = addf %18, %30 : vector<8xf32>
  // CHECK:     %41 = addf %20, %32 : vector<8xf32>
  // CHECK:     %42 = addf %22, %34 : vector<8xf32>
  // CHECK:     %43 = addf %24, %36 : vector<8xf32>
  // CHECK:     %44 = addf %26, %38 : vector<8xf32>
  // CHECK:     %45 = affine_apply #map0(%i4, %i5)
  // CHECK:     vector_transfer_write %39, %2, %45#0, %45#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %46 = affine_apply #map2(%i4, %i5)
  // CHECK:     vector_transfer_write %40, %2, %46#0, %46#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %47 = affine_apply #map3(%i4, %i5)
  // CHECK:     vector_transfer_write %41, %2, %47#0, %47#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %48 = affine_apply #map4(%i4, %i5)
  // CHECK:     vector_transfer_write %42, %2, %48#0, %48#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %49 = affine_apply #map5(%i4, %i5)
  // CHECK:     vector_transfer_write %43, %2, %49#0, %49#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %50 = affine_apply #map6(%i4, %i5)
  // CHECK:     vector_transfer_write %44, %2, %50#0, %50#1 {permutation_map: #map1} : vector<8xf32>, memref<?x?xf32>, index, index
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
