// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=2 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1d2_d0d2:map[0-9]+]] = (d0, d1, d2) -> (d0, d2)

func @vec2d(%A : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?x?xf32>
   %N = dim %A, 1 : memref<?x?x?xf32>
   %P = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: for %i0 = 0 to %0 step 32
   // CHECK:   for %i1 = 0 to %1 {
   // CHECK:     for %i2 = 0 to %2 step 256
   // CHECK:       {{.*}} = vector_transfer_read %arg0, %i0, %i1, %i2 {permutation_map: #[[map_proj_d0d1d2_d0d2]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   for %i0 = 0 to %M {
     for %i1 = 0 to %N {
       for %i2 = 0 to %P {
         %a2 = load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
       }
     }
   }
   // CHECK: for  {{.*}} = 0 to %0 {
   // CHECK:   for  {{.*}} = 0 to %1 {
   // CHECK:     for  {{.*}} = 0 to %2 {
   // For the case: --test-fastest-varying=2 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order
   for %i3 = 0 to %M {
     for %i4 = 0 to %N {
       for %i5 = 0 to %P {
         %a5 = load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}
