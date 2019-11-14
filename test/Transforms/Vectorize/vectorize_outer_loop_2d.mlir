// RUN: mlir-opt %s -affine-vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=2 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1d2_d0d2:map[0-9]+]] = (d0, d1, d2) -> (d0, d2)

func @vec2d(%A : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?x?xf32>
   %N = dim %A, 1 : memref<?x?x?xf32>
   %P = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: affine.for %{{.*}} = 0 to %{{.*}} step 32
   // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
   // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 256
   // CHECK:       {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1d2_d0d2]]} : memref<?x?x?xf32>,  vector<32x256xf32>
   affine.for %i0 = 0 to %M {
     affine.for %i1 = 0 to %N {
       affine.for %i2 = 0 to %P {
         %a2 = affine.load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
       }
     }
   }
   // CHECK: for  {{.*}} = 0 to %{{.*}} {
   // CHECK:   for  {{.*}} = 0 to %{{.*}} {
   // CHECK:     for  {{.*}} = 0 to %{{.*}} {
   // For the case: --test-fastest-varying=2 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order
   affine.for %i3 = 0 to %M {
     affine.for %i4 = 0 to %N {
       affine.for %i5 = 0 to %P {
         %a5 = affine.load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}
