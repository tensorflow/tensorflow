// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 --test-fastest-varying=0 -materialize-vectors -vector-size=8 | FileCheck %s -check-prefix=VEC1DTO1D
// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 16 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=8 | FileCheck %s -check-prefix=VEC2DTO1D
// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 32 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=3 -vector-size=16 | FileCheck %s -check-prefix=VEC2DTO2D

// Capture permutation maps used in vectorization.
// VEC1DTO1D-DAG: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)
// VEC2DTO1D-DAG: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)
// VEC2DTO2D-DAG: #[[map_proj_d0d1_d0d1:map[0-9]+]] = (d0, d1) -> (d0, d1)

// vector<32xf32> -> vector<8xf32>
// VEC1DTO1D-DAG: [[MAP0:#.*]] = (d0, d1) -> (d0, d1)
// VEC1DTO1D-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 8)
// VEC1DTO1D-DAG: [[MAP2:#.*]] = (d0, d1) -> (d0, d1 + 16)
// VEC1DTO1D-DAG: [[MAP3:#.*]] = (d0, d1) -> (d0, d1 + 24)
// vector<3x16xf32> -> vector<8xf32>
// VEC2DTO1D-DAG: [[MAP0:#.*]] = (d0, d1) -> (d0, d1)
// VEC2DTO1D-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 8)
// VEC2DTO1D-DAG: [[MAP2:#.*]] = (d0, d1) -> (d0 + 1, d1)
// VEC2DTO1D-DAG: [[MAP3:#.*]] = (d0, d1) -> (d0 + 1, d1 + 8)
// VEC2DTO1D-DAG: [[MAP4:#.*]] = (d0, d1) -> (d0 + 2, d1)
// VEC2DTO1D-DAG: [[MAP5:#.*]] = (d0, d1) -> (d0 + 2, d1 + 8)
// vector<3x32xf32> -> vector<3x16xf32>
// VEC2DTO2D-DAG: [[MAP1:#.*]] = (d0, d1) -> (d0, d1 + 16)
mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      // VEC1DTO1D does 4x unrolling.
      // VEC1DTO1D: [[CST0:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
      // VEC1DTO1D: [[CST1:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
      // VEC1DTO1D: [[CST2:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
      // VEC1DTO1D: [[CST3:%.*]] = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
      // VEC1DTO1D: [[VAL0:%.*]] = affine_apply [[MAP0]]{{.*}}
      // VEC1DTO1D: vector_transfer_write [[CST0]], {{.*}}, [[VAL0]]#0, [[VAL0]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      // VEC1DTO1D: [[VAL1:%.*]] = affine_apply [[MAP1]]{{.*}}
      // VEC1DTO1D: vector_transfer_write [[CST1]], {{.*}}, [[VAL1]]#0, [[VAL1]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      // VEC1DTO1D: [[VAL2:%.*]] = affine_apply [[MAP2]]{{.*}}
      // VEC1DTO1D:vector_transfer_write [[CST2]], {{.*}}, [[VAL2]]#0, [[VAL2]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      // VEC1DTO1D: [[VAL3:%.*]] = affine_apply [[MAP3]]{{.*}}
      // VEC1DTO1D:vector_transfer_write [[CST3]], {{.*}}, [[VAL3]]#0, [[VAL3]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      //
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      // VEC2DTO1D does (3x4)x unrolling.
      // VEC2DTO1D-COUNT-6: {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
      // VEC2DTO1D: [[VAL0:%.*]] = affine_apply [[MAP0]]{{.*}}
      // VEC2DTO1D: vector_transfer_write {{.*}}, [[VAL0]]#0, [[VAL0]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      // ... 4 other interleaved affine_apply, vector_transfer_write
      // VEC2DTO1D: [[VAL5:%.*]] = affine_apply [[MAP5]]{{.*}}
      // VEC2DTO1D: vector_transfer_write {{.*}}, [[VAL5]]#0, [[VAL5]]#1 {permutation_map: #[[map_proj_d0d1_d1]]} : vector<8xf32>
      //
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  for %i4 = 0 to %M {
    for %i5 = 0 to %N {
      // VEC2DTO2D: %7 = affine_apply #map0(%i4, %i5)
      // VEC2DTO2D: %8 = vector_transfer_read %0, %7#0, %7#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
      // VEC2DTO2D: %9 = affine_apply #map1(%i4, %i5)
      // VEC2DTO2D: %10 = vector_transfer_read %0, %9#0, %9#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
      // VEC2DTO2D: %11 = affine_apply #map0(%i4, %i5)
      // VEC2DTO2D: %12 = vector_transfer_read %1, %11#0, %11#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
      // VEC2DTO2D: %13 = affine_apply #map1(%i4, %i5)
      // VEC2DTO2D: %14 = vector_transfer_read %1, %13#0, %13#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<3x16xf32>
      // VEC2DTO2D: %15 = addf %8, %12 : vector<3x16xf32>
      // VEC2DTO2D: %16 = addf %10, %14 : vector<3x16xf32>
      // VEC2DTO2D: %17 = affine_apply #map0(%i4, %i5)
      // VEC2DTO2D: vector_transfer_write %15, %2, %17#0, %17#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : vector<3x16xf32>, memref<?x?xf32>, index, index
      // VEC2DTO2D: %18 = affine_apply #map1(%i4, %i5)
      // VEC2DTO2D: vector_transfer_write %16, %2, %18#0, %18#1 {permutation_map: #[[map_proj_d0d1_d0d1]]} : vector<3x16xf32>, memref<?x?xf32>, index, index
      //
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
