// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 --test-fastest-varying=0 -materialize-vectors -vector-size=8 | FileCheck %s

// Capture permutation maps used in vectorization.
// CHECK-DAG: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)

// vector<32xf32> -> vector<8xf32>
// CHECK-DAG: [[MAP0:#.*]] = (d0, d1) -> (d0, d1)
// CHECK-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 8)
// CHECK-DAG: [[MAP2:#.*]] = (d0, d1) -> (d0, d1 + 16)
// CHECK-DAG: [[MAP3:#.*]] = (d0, d1) -> (d0, d1 + 24)
mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  // 4x unroll (jammed by construction).
  // CHECK: for %i0 = 0 to %arg0 {
  // CHECK:   for %i1 = 0 to %arg1 step 32 {
  // CHECK:     [[CST0:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     [[CST1:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     [[CST2:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     [[CST3:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK:     [[VAL0:%.*]] = affine_apply [[MAP0]]{{.*}}
  // CHECK:     vector_transfer_write [[CST0]], {{.*}}, [[VAL0]]#0, [[VAL0]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL1:%.*]] = affine_apply [[MAP1]]{{.*}}
  // CHECK:     vector_transfer_write [[CST1]], {{.*}}, [[VAL1]]#0, [[VAL1]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL2:%.*]] = affine_apply [[MAP2]]{{.*}}
  // CHECK:     vector_transfer_write [[CST2]], {{.*}}, [[VAL2]]#0, [[VAL2]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL3:%.*]] = affine_apply [[MAP3]]{{.*}}
  // CHECK:     vector_transfer_write [[CST3]], {{.*}}, [[VAL3]]#0, [[VAL3]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  //
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  // 4x unroll (jammed by construction).
  // CHECK: for %i2 = 0 to %arg0 {
  // CHECK:   for %i3 = 0 to %arg1 step 32 {
  // CHECK:     [[CST0:%.*]] = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     [[CST1:%.*]] = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     [[CST2:%.*]] = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     [[CST3:%.*]] = constant splat<vector<8xf32>, 2.000000e+00> : vector<8xf32>
  // CHECK:     [[VAL0:%.*]] = affine_apply [[MAP0]]{{.*}}
  // CHECK:     vector_transfer_write [[CST0]], {{.*}}, [[VAL0]]#0, [[VAL0]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL1:%.*]] = affine_apply [[MAP1]]{{.*}}
  // CHECK:     vector_transfer_write [[CST1]], {{.*}}, [[VAL1]]#0, [[VAL1]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL2:%.*]] = affine_apply [[MAP2]]{{.*}}
  // CHECK:     vector_transfer_write [[CST2]], {{.*}}, [[VAL2]]#0, [[VAL2]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  // CHECK:     [[VAL3:%.*]] = affine_apply [[MAP3]]{{.*}}
  // CHECK:     vector_transfer_write [[CST3]], {{.*}}, [[VAL3]]#0, [[VAL3]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
  //
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  // 4x unroll (jammed by construction).
  // CHECK: for %i4 = 0 to %arg0 {
  // CHECK:   for %i5 = 0 to %arg1 step 32 {
  // CHECK:     %11 = affine_apply #map0(%i4, %i5)
  // CHECK:     %12 = vector_transfer_read %0, %11#0, %11#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %13 = affine_apply #map2(%i4, %i5)
  // CHECK:     %14 = vector_transfer_read %0, %13#0, %13#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %15 = affine_apply #map3(%i4, %i5)
  // CHECK:     %16 = vector_transfer_read %0, %15#0, %15#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %17 = affine_apply #map4(%i4, %i5)
  // CHECK:     %18 = vector_transfer_read %0, %17#0, %17#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %19 = affine_apply #map0(%i4, %i5)
  // CHECK:     %20 = vector_transfer_read %1, %19#0, %19#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %21 = affine_apply #map2(%i4, %i5)
  // CHECK:     %22 = vector_transfer_read %1, %21#0, %21#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %23 = affine_apply #map3(%i4, %i5)
  // CHECK:     %24 = vector_transfer_read %1, %23#0, %23#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %25 = affine_apply #map4(%i4, %i5)
  // CHECK:     %26 = vector_transfer_read %1, %25#0, %25#1 {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<8xf32>
  // CHECK:     %27 = addf %12, %20 : vector<8xf32>
  // CHECK:     %28 = addf %14, %22 : vector<8xf32>
  // CHECK:     %29 = addf %16, %24 : vector<8xf32>
  // CHECK:     %30 = addf %18, %26 : vector<8xf32>
  // CHECK:     %31 = affine_apply #map0(%i4, %i5)
  // CHECK:     vector_transfer_write %27, %2, %31#0, %31#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %32 = affine_apply #map2(%i4, %i5)
  // CHECK:     vector_transfer_write %28, %2, %32#0, %32#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %33 = affine_apply #map3(%i4, %i5)
  // CHECK:     vector_transfer_write %29, %2, %33#0, %33#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>, memref<?x?xf32>, index, index
  // CHECK:     %34 = affine_apply #map4(%i4, %i5)
  // CHECK:     vector_transfer_write %30, %2, %34#0, %34#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>, memref<?x?xf32>, index, index
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
