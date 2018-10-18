// RUN: mlir-opt %s -vectorize -virtual-vector-size 128 | FileCheck %s

#map0 = (d0) -> (d0)
#map1 = (d0, d1) -> (d0, d1)
#map1_t = (d0, d1) -> (d1, d0)
#map2 = (d0, d1) -> (d1 + d0, d0)
#map3 = (d0, d1) -> (d1, d0 + d1)
#map4 = (d0, d1, d2) -> (d1, d0 + d1, d0 + d2)
mlfunc @vec(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
   for %i0 = 0 to %M {
// CHECK: Vectorizable loop: for %i0
     %a0 = load %A[%cst0, %cst0] : memref<?x?xf32>
   }
   for %i1 = 0 to %M {
// CHECK: Non-vectorizable loop: for %i1
     %a1 = load %A[%i1, %i1] : memref<?x?xf32>
   }
   for %i2 = 0 to %M {
// CHECK: Non-vectorizable loop: for %i2
     %r2 = affine_apply (d0) -> (d0) (%i2)
     %a2 = load %A[%r2#0, %cst0] : memref<?x?xf32>
   }
   for %i3 = 0 to %M {
// CHECK: Vectorizable loop: for %i3
     %r3 = affine_apply (d0) -> (d0) (%i3)
     %a3 = load %A[%cst0, %r3#0] : memref<?x?xf32>
   }
   for %i4 = 0 to %M {
// CHECK: Vectorizable loop: for %i4
     for %i5 = 0 to %N {
// CHECK: Non-vectorizable loop: for %i5
       %r5 = affine_apply #map1_t (%i4, %i5)
       %a5 = load %A[%r5#0, %r5#1] : memref<?x?xf32>
     }
   }
   for %i6 = 0 to %M {
// CHECK: Non-vectorizable loop: for %i6
     for %i7 = 0 to %N {
// CHECK: Non-vectorizable loop: for %i7
       %r7 = affine_apply #map2 (%i6, %i7)
       %a7 = load %A[%r7#0, %r7#1] : memref<?x?xf32>
     }
   }
   for %i8 = 0 to %M {
// CHECK: Vectorizable loop: for %i8
     for %i9 = 0 to %N {
// CHECK: Non-vectorizable loop: for %i9
       %r9 = affine_apply #map3 (%i8, %i9)
       %a9 = load %A[%r9#0, %r9#1] : memref<?x?xf32>
     }
   }
   for %i10 = 0 to %M {
// CHECK: Non-vectorizable loop: for %i10
     for %i11 = 0 to %N {
// CHECK: Non-vectorizable loop: for %i11
       %r11 = affine_apply #map1 (%i10, %i11)
       %a11 = load %A[%r11#0, %r11#1] : memref<?x?xf32>
       %r12 = affine_apply #map1_t (%i10, %i11)
       store %a11, %A[%r12#0, %r12#1] : memref<?x?xf32>
     }
   }
   for %i12 = 0 to %M {
// CHECK: Non-vectorizable loop: for %i12
     for %i13 = 0 to %N {
// CHECK: Non-vectorizable loop: for %i13
       for %i14 = 0 to %P {
// CHECK: Vectorizable loop: for %i14
         %r14 = affine_apply #map4 (%i12, %i13, %i14)
         %a14 = load %B[%r14#0, %r14#1, %r14#2] : memref<?x?x?xf32>
       }
     }
   }
   return
}
