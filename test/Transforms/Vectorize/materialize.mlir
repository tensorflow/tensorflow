// RUN: mlir-opt %s -materialize-vectors -vector-size=4 -vector-size=4 | FileCheck %s

// CHECK-DAG: #[[ID1:map[0-9]+]] = (d0) -> (d0)
// CHECK-DAG: #[[D0D1D2D3TOD1D0:map[0-9]+]] = (d0, d1, d2, d3) -> (d1, d0)
// CHECK-DAG: #[[D0P1:map[0-9]+]] = (d0) -> (d0 + 1)
// CHECK-DAG: #[[D0P2:map[0-9]+]] = (d0) -> (d0 + 2)
// CHECK-DAG: #[[D0P3:map[0-9]+]] = (d0) -> (d0 + 3)

// CHECK-LABEL: func @materialize
func @materialize(%M : index, %N : index, %O : index, %P : index) {
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant splat<vector<4x4x4xf32>, 1.000000e+00> : vector<4x4x4xf32>
  // CHECK:  affine.for %i0 = 0 to %arg0 step 4 {
  // CHECK-NEXT:    affine.for %i1 = 0 to %arg1 step 4 {
  // CHECK-NEXT:      affine.for %i2 = 0 to %arg2 {
  // CHECK-NEXT:        affine.for %i3 = 0 to %arg3 step 4 {
  // CHECK-NEXT:          %[[a:[0-9]+]] = {{.*}}[[ID1]](%i0)
  // CHECK-NEXT:          %[[b:[0-9]+]] = {{.*}}[[ID1]](%i1)
  // CHECK-NEXT:          %[[c:[0-9]+]] = {{.*}}[[ID1]](%i2)
  // CHECK-NEXT:          %[[d:[0-9]+]] = {{.*}}[[ID1]](%i3)
  // CHECK-NEXT:          vector_transfer_write {{.*}}, %0, %[[a]], %[[b]], %[[c]], %[[d]] {permutation_map: #[[D0D1D2D3TOD1D0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %[[b1:[0-9]+]] = {{.*}}[[D0P1]](%i1)
  // CHECK:          vector_transfer_write {{.*}}, %0, {{.*}}, %[[b1]], {{.*}} {permutation_map: #[[D0D1D2D3TOD1D0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %[[b2:[0-9]+]] = {{.*}}[[D0P2]](%i1)
  // CHECK:          vector_transfer_write {{.*}}, %0, {{.*}}, %[[b2]], {{.*}} {permutation_map: #[[D0D1D2D3TOD1D0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  // CHECK:          %[[b3:[0-9]+]] = {{.*}}[[D0P3]](%i1)
  // CHECK:          vector_transfer_write {{.*}}, %0, {{.*}}, %[[b3]], {{.*}} {permutation_map: #[[D0D1D2D3TOD1D0]]} : vector<4x4xf32>, memref<?x?x?x?xf32>, index, index, index, index
  affine.for %i0 = 0 to %M step 4 {
    affine.for %i1 = 0 to %N step 4 {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 4 {
          "vector_transfer_write"(%f1, %A, %i0, %i1, %i2, %i3) {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d0)} : (vector<4x4x4xf32>, memref<?x?x?x?xf32, 0>, index, index, index, index) -> ()
        }
      }
    }
  }
  return
}