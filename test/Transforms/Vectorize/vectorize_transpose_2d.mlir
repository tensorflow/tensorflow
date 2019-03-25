// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=0 --test-fastest-varying=1 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK-DAG: #[[map_proj_d0d1d2_d2d1:map[0-9]+]] = (d0, d1, d2) -> (d2, d1)

func @vec2d(%A : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?x?xf32>
   %N = dim %A, 1 : memref<?x?x?xf32>
   %P = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: for  {{.*}} = 0 to %0 {
   // CHECK:   for  {{.*}} = 0 to %1 {
   // CHECK:     for  {{.*}} = 0 to %2 {
   // For the case: --test-fastest-varying=0 --test-fastest-varying=1 no
   // vectorization happens because of loop nesting order.
  affine.for %i0 = 0 to %M {
     affine.for %i1 = 0 to %N {
       affine.for %i2 = 0 to %P {
         %a2 = load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
       }
     }
   }
   // CHECK: affine.for %i3 = 0 to %0 step 32
   // CHECK:   affine.for %i4 = 0 to %1 {
   // CHECK:     affine.for %i5 = 0 to %2 step 256
   // CHECK:       {{.*}} = vector_transfer_read %arg0, %i4, %i5, %i3 {permutation_map: #[[map_proj_d0d1d2_d2d1]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   affine.for %i3 = 0 to %M {
     affine.for %i4 = 0 to %N {
       affine.for %i5 = 0 to %P {
         %a5 = load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}

func @vec2d_imperfectly_nested(%A : memref<?x?x?xf32>) {
   %0 = dim %A, 0 : memref<?x?x?xf32>
   %1 = dim %A, 1 : memref<?x?x?xf32>
   %2 = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: affine.for %i0 = 0 to %0 step 32 {
   // CHECK:   affine.for %i1 = 0 to %1 step 256 {
   // CHECK:     affine.for %i2 = 0 to %2 {
   // CHECK:       %3 = vector_transfer_read %arg0, %i2, %i1, %i0 {permutation_map: #[[map_proj_d0d1d2_d2d1]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   // CHECK:   affine.for %i3 = 0 to %1 {
   // CHECK:     affine.for %i4 = 0 to %2 step 256 {
   // CHECK:       %4 = vector_transfer_read %arg0, %i3, %i4, %i0 {permutation_map: #[[map_proj_d0d1d2_d2d1]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   // CHECK:     affine.for %i5 = 0 to %2 step 256 {
   // CHECK:       %5 = vector_transfer_read %arg0, %i3, %i5, %i0 {permutation_map: #[[map_proj_d0d1d2_d2d1]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   affine.for %i0 = 0 to %0 {
     affine.for %i1 = 0 to %1 {
       affine.for %i2 = 0 to %2 {
         %a2 = load %A[%i2, %i1, %i0] : memref<?x?x?xf32>
       }
     }
     affine.for %i3 = 0 to %1 {
       affine.for %i4 = 0 to %2 {
         %a4 = load %A[%i3, %i4, %i0] : memref<?x?x?xf32>
       }
       affine.for %i5 = 0 to %2 {
         %a5 = load %A[%i3, %i5, %i0] : memref<?x?x?xf32>
       }
     }
   }
   return
}

