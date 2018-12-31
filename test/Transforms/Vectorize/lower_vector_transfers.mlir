// RUN: mlir-opt %s -lower-vector-transfers | FileCheck %s

// CHECK: #[[ADD:map[0-9]+]] = (d0, d1) -> (d0 + d1)
// CHECK: #[[SUB:map[0-9]+]] = (d0, d1) -> (d0 - d1)
// CHECK-LABEL: mlfunc @materialize_read(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
mlfunc @materialize_read(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-NEXT:  %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:    for %i1 = 0 to %arg1 {
  // CHECK-NEXT:      for %i2 = 0 to %arg2 {
  // CHECK-NEXT:        for %i3 = 0 to %arg3 step 5 {
  // CHECK-NEXT:          %c0 = constant 0 : index
  // CHECK-NEXT:          %c1 = constant 1 : index
  // CHECK:               %1 = dim %0, 0 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %2 = dim %0, 1 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %3 = dim %0, 2 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %4 = dim %0, 3 : memref<?x?x?x?xf32>
  // CHECK:               %5 = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:          %6 = vector_type_cast %5 : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          for %i4 = 0 to 3 {
  // CHECK-NEXT:            for %i5 = 0 to 4 {
  // CHECK-NEXT:              for %i6 = 0 to 5 {
  // CHECK-NEXT:                %7 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %8 = cmpi "slt", %7, %c0 : index
  // CHECK-NEXT:                %9 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %10 = cmpi "slt", %9, %1 : index
  // CHECK-NEXT:                %11 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %12 = affine_apply #[[SUB]](%1, %c1)
  // CHECK-NEXT:                %13 = select %10, %11, %12 : index
  // CHECK-NEXT:                %14 = select %8, %c0, %13 : index
  // CHECK-NEXT:                %15 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %16 = cmpi "slt", %15, %c0 : index
  // CHECK-NEXT:                %17 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %18 = cmpi "slt", %17, %4 : index
  // CHECK-NEXT:                %19 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %20 = affine_apply #[[SUB]](%4, %c1)
  // CHECK-NEXT:                %21 = select %18, %19, %20 : index
  // CHECK-NEXT:                %22 = select %16, %c0, %21 : index
  // CHECK-NEXT:                %23 = load %0[%14, %i1, %i2, %22] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                store %23, %5[%i6, %i5, %i4] : memref<5x4x3xf32>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          %24 = load %6[%c0] : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          dealloc %5 : memref<5x4x3xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
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

// CHECK-LABEL:mlfunc @materialize_write(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
mlfunc @materialize_write(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-NEXT:  %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  %cst = constant splat<vector<5x4x3xf32>, 1.000000e+00> : vector<5x4x3xf32>
  // CHECK-NEXT:  for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:    for %i1 = 0 to %arg1 step 4 {
  // CHECK-NEXT:      for %i2 = 0 to %arg2 {
  // CHECK-NEXT:        for %i3 = 0 to %arg3 step 5 {
  // CHECK-NEXT:          %c0 = constant 0 : index
  // CHECK-NEXT:          %c1 = constant 1 : index
  // CHECK:               %1 = dim %0, 0 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %2 = dim %0, 1 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %3 = dim %0, 2 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %4 = dim %0, 3 : memref<?x?x?x?xf32>
  // CHECK:               %5 = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:          %6 = vector_type_cast %5 : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          store %cst, %6[%c0] : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          for %i4 = 0 to 3 {
  // CHECK-NEXT:            for %i5 = 0 to 4 {
  // CHECK-NEXT:              for %i6 = 0 to 5 {
  // CHECK-NEXT:                %7 = load %5[%i6, %i5, %i4] : memref<5x4x3xf32>
  // CHECK-NEXT:                %8 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %9 = cmpi "slt", %8, %c0 : index
  // CHECK-NEXT:                %10 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %11 = cmpi "slt", %10, %1 : index
  // CHECK-NEXT:                %12 = affine_apply #[[ADD]](%i0, %i4)
  // CHECK-NEXT:                %13 = affine_apply #[[SUB]](%1, %c1)
  // CHECK-NEXT:                %14 = select %11, %12, %13 : index
  // CHECK-NEXT:                %15 = select %9, %c0, %14 : index
  // CHECK-NEXT:                %16 = affine_apply #[[ADD]](%i1, %i5)
  // CHECK-NEXT:                %17 = cmpi "slt", %16, %c0 : index
  // CHECK-NEXT:                %18 = affine_apply #[[ADD]](%i1, %i5)
  // CHECK-NEXT:                %19 = cmpi "slt", %18, %2 : index
  // CHECK-NEXT:                %20 = affine_apply #[[ADD]](%i1, %i5)
  // CHECK-NEXT:                %21 = affine_apply #[[SUB]](%2, %c1)
  // CHECK-NEXT:                %22 = select %19, %20, %21 : index
  // CHECK-NEXT:                %23 = select %17, %c0, %22 : index
  // CHECK-NEXT:                %24 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %25 = cmpi "slt", %24, %c0 : index
  // CHECK-NEXT:                %26 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %27 = cmpi "slt", %26, %4 : index
  // CHECK-NEXT:                %28 = affine_apply #[[ADD]](%i3, %i6)
  // CHECK-NEXT:                %29 = affine_apply #[[SUB]](%4, %c1)
  // CHECK-NEXT:                %30 = select %27, %28, %29 : index
  // CHECK-NEXT:                %31 = select %25, %c0, %30 : index
  // CHECK-NEXT:                store %7, %0[%15, %23, %i2, %31] : memref<?x?x?x?xf32>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          dealloc %5 : memref<5x4x3xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant splat<vector<5x4x3xf32>, 1.000000e+00> : vector<5x4x3xf32>
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
