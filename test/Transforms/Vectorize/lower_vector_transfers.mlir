// RUN: mlir-opt %s -lower-vector-transfers | FileCheck %s

// CHECK: #[[ADD:map[0-9]+]] = (d0, d1) -> (d0 + d1)
mlfunc @materialize_read(%M : index, %N : index, %O : index, %P : index) {
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  // CHECK:      for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i1 = 0 to %arg1 {
  // CHECK-NEXT:     for %i2 = 0 to %arg2 {
  // CHECK-NEXT:       for %i3 = 0 to %arg3 step 5 {
  // CHECK-NEXT:         %1 = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:         %2 = "vector_type_cast"(%1) : (memref<5x4x3xf32>) -> memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:         for %i4 = 0 to 5 {
  // CHECK-NEXT:           %3 = affine_apply #[[ADD]](%i3, %i4)
  // CHECK-NEXT:           for %i5 = 0 to 4 {
  // CHECK-NEXT:             for %i6 = 0 to 3 {
  // CHECK-NEXT:               %4 = affine_apply #[[ADD]](%i0, %i6)
  // CHECK-NEXT:               %5 = load %0[%4, %i1, %i2, %3] : memref<?x?x?x?xf32>
  // CHECK-NEXT:               store %5, %1[%i4, %i5, %i6] : memref<5x4x3xf32>
  // CHECK-NEXT:             }
  // CHECK-NEXT:           }
  // CHECK-NEXT:         }
  // CHECK-NEXT:         %6 = load %2[%c0] : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:         dealloc %1 : memref<5x4x3xf32>
  for %i0 = 0 to %M step 3 {
    for %i1 = 0 to %N {
      for %i2 = 0 to %O {
        for %i3 = 0 to %P step 5 {
          %f = vector_transfer_read %A, %i0, %i1, %i2, %i3 {permutation_map: (d0, d1, d2, d3) -> (d3, 0, d0)} : (memref<?x?x?x?xf32, 0>, index, index, index, index) -> vector<5x4x3xf32>
        }
      }
    }
  }
  return
}

mlfunc @materialize_write(%M : index, %N : index, %O : index, %P : index) {
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant splat<vector<5x4x3xf32>, 1.000000e+00> : vector<5x4x3xf32>
  // CHECK:      for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i1 = 0 to %arg1 step 4 {
  // CHECK-NEXT:     for %i2 = 0 to %arg2 {
  // CHECK-NEXT:       for %i3 = 0 to %arg3 step 5 {
  // CHECK-NEXT:         %1 = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:         %2 = "vector_type_cast"(%1) : (memref<5x4x3xf32>) -> memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:         store %cst, %2[%c0] : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:         for %i4 = 0 to 5 {
  // CHECK-NEXT:           %3 = affine_apply #[[ADD]](%i3, %i4)
  // CHECK-NEXT:           for %i5 = 0 to 4 {
  // CHECK-NEXT:             %4 = affine_apply #[[ADD]](%i1, %i5)
  // CHECK-NEXT:             for %i6 = 0 to 3 {
  // CHECK-NEXT:               %5 = affine_apply #[[ADD]](%i0, %i6)
  // CHECK-NEXT:               %6 = load %1[%i4, %i5, %i6] : memref<5x4x3xf32>
  // CHECK-NEXT:               store %6, %0[%5, %4, %i2, %3] : memref<?x?x?x?xf32>
  // CHECK-NEXT:             }
  // CHECK-NEXT:           }
  // CHECK-NEXT:         }
  // CHECK-NEXT:         dealloc %1 : memref<5x4x3xf32>
  for %i0 = 0 to %M step 3 {
    for %i1 = 0 to %N step 4 {
      for %i2 = 0 to %O {
        for %i3 = 0 to %P step 5 {
          vector_transfer_write %f1, %A, %i0, %i1, %i2, %i3 {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d0)} : vector<5x4x3xf32>, memref<?x?x?x?xf32, 0>, index, index, index, index
        }
      }
    }
  }
  return
}