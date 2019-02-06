// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 64 -virtual-vector-size 256 --test-fastest-varying=2 --test-fastest-varying=1 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1d2_d0d1d2:map[0-9]+]] = (d0, d1, d2) -> (d0, d1, d2)

func @vec3d(%A : memref<?x?x?xf32>) {
   %0 = dim %A, 0 : memref<?x?x?xf32>
   %1 = dim %A, 1 : memref<?x?x?xf32>
   %2 = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: affine.for %i0 = 0 to %0 {
   // CHECK:   affine.for %i1 = 0 to %0 {
   // CHECK:     affine.for %i2 = 0 to %0 step 32 {
   // CHECK:       affine.for %i3 = 0 to %1 step 64 {
   // CHECK:         affine.for %i4 = 0 to %2 step 256 {
   // CHECK:           %3 = vector_transfer_read %arg0, %i2, %i3, %i4 {permutation_map: #[[map_proj_d0d1d2_d0d1d2]]} : (memref<?x?x?xf32>, index, index, index) -> vector<32x64x256xf32>
   affine.for %t0 = 0 to %0 {
     affine.for %t1 = 0 to %0 {
       affine.for %i0 = 0 to %0 {
         affine.for %i1 = 0 to %1 {
           affine.for %i2 = 0 to %2 {
             %a2 = load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
           }
         }
       }
     }
   }
   return
}
