// RUN: mlir-opt %s -materialize-vectors -vector-size=4 -vector-size=4 | FileCheck %s

// CHECK-DAG: #[[map_instance_0:map[0-9]+]] = (d0, d1, d2, d3) -> (d0, d1, d2, d3)
// CHECK-DAG: #[[map_instance_1:map[0-9]+]] = (d0, d1, d2, d3) -> (d0, d1 + 1, d2, d3)
// CHECK-DAG: #[[map_instance_2:map[0-9]+]] = (d0, d1, d2, d3) -> (d0, d1 + 2, d2, d3)
// CHECK-DAG: #[[map_instance_3:map[0-9]+]] = (d0, d1, d2, d3) -> (d0, d1 + 3, d2, d3)
// CHECK-DAG: #[[map_proj_d0d1d2d3d4_d1d0:map[0-9]+]] = (d0, d1, d2, d3) -> (d1, d0)

func @materialize(%M : index, %N : index, %O : index, %P : index) {
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant splat<vector<4x4x4xf32>, 1.000000e+00> : vector<4x4x4xf32>
  // CHECK:  for %i0 = 0 to %arg0 step 4 {
  // CHECK:    for %i1 = 0 to %arg1 step 4 {
  // CHECK:      for %i2 = 0 to %arg2 {
  // CHECK:        for %i3 = 0 to %arg3 step 4 {
  // CHECK:          %1 = affine_apply #[[map_instance_0]](%i0, %i1, %i2, %i3)
  // CHECK:          vector_transfer_write {{.*}}, %0, %1#0, %1#1, %1#2, %1#3 {permutation_map: #[[map_proj_d0d1d2d3d4_d1d0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %2 = affine_apply #[[map_instance_1]](%i0, %i1, %i2, %i3)
  // CHECK:          vector_transfer_write {{.*}}, %0, %2#0, %2#1, %2#2, %2#3 {permutation_map: #[[map_proj_d0d1d2d3d4_d1d0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %3 = affine_apply #[[map_instance_2]](%i0, %i1, %i2, %i3)
  // CHECK:          vector_transfer_write {{.*}}, %0, %3#0, %3#1, %3#2, %3#3 {permutation_map: #[[map_proj_d0d1d2d3d4_d1d0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %4 = affine_apply #[[map_instance_3]](%i0, %i1, %i2, %i3)
  // CHECK:          vector_transfer_write {{.*}}, %0, %4#0, %4#1, %4#2, %4#3 {permutation_map: #[[map_proj_d0d1d2d3d4_d1d0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  for %i0 = 0 to %M step 4 {
    for %i1 = 0 to %N step 4 {
      for %i2 = 0 to %O {
        for %i3 = 0 to %P step 4 {
          "vector_transfer_write"(%f1, %A, %i0, %i1, %i2, %i3) {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d0)} : (vector<4x4x4xf32>, memref<?x?x?x?xf32, 0>, index, index, index, index) -> ()
        }
      }
    }
  }
  return
}