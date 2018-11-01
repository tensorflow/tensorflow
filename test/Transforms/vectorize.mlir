// RUN: mlir-opt %s -vectorize -virtual-vector-size 128 --test-fastest-varying=0 | FileCheck %s -check-prefix=VEC1D
// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=1 --test-fastest-varying=0 | FileCheck %s -check-prefix=VEC2D
// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=0 --test-fastest-varying=1 | FileCheck %s -check-prefix=VEC2D_T
// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=2 --test-fastest-varying=0 | FileCheck %s -check-prefix=VEC2D_O
// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=0 --test-fastest-varying=2 | FileCheck %s -check-prefix=VEC2D_OT
// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 64 -virtual-vector-size 256 --test-fastest-varying=2 --test-fastest-varying=1 --test-fastest-varying=0 | FileCheck %s -check-prefix=VEC3D

#map0 = (d0) -> (d0)
#map1 = (d0, d1) -> (d0, d1)
#map1_t = (d0, d1) -> (d1, d0)
#map2 = (d0, d1) -> (d1 + d0, d0)
#map3 = (d0, d1) -> (d1, d0 + d1)
#map4 = (d0, d1, d2) -> (d1, d0 + d1, d0 + d2)
#mapadd1 = (d0) -> (d0 + 1)
#mapadd2 = (d0) -> (d0 + 2)
#mapadd3 = (d0) -> (d0 + 3)
#set0 = (i) : (i >= 0)
// Maps introduced to vectorize fastest varying memory index.
// VEC1D: [[MAPSHIFT:#map[0-9]*]] = ()[s0] -> (s0 + 127)
mlfunc @vec1d(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// VEC1D: [[C0:%[a-z0-9_]+]] = constant 0 : index
// VEC1D: [[ARG_M:%[0-9]+]] = dim %arg0, 0 : memref<?x?xf32>
// VEC1D: [[ARG_N:%[0-9]+]] = dim %arg0, 1 : memref<?x?xf32>
// VEC1D: [[ARG_P:%[0-9]+]] = dim %arg1, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
// VEC1D: [[C1:%[a-z0-9_]+]] = constant 0 : index
   %cst0 = constant 0 : index
// CHECK:for [[IV0:%[a-zA-Z0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// CHECK:   [[ALLOC0:%[a-zA-Z0-9]+]] = alloc() : memref<1xvector<128xf32>>
// CHECK:   [[UNALIGNED0: %[0-9]*]] = "n_d_unaligned_load"(%arg0, [[C1]], [[C1]], [[ALLOC0]], [[C0]]) : {{.*}}
// CHECK:   %{{.*}} = load [[ALLOC0]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
// For this simple loop, the current transformation generates:
//   for %i0 = 0 to #map0()[%0] step 128 {
//     %3 = alloc() : memref<1xvector<128xf32>>
//     %4 = "n_d_unaligned_load"(%arg0, %c0_0, %c0_0, %3, %c0) : (memref<?x?xf32>, index, index, memref<1xvector<128xf32>>, index) -> (memref<?x?xf32>, index, index, memref<1xvector<128xf32>>, index)
//     %5 = load %3[%c0] : memref<1xvector<128xf32>>
//   }
   for %i0 = 0 to %M { // vectorized due to scalar -> vector 
     %a0 = load %A[%cst0, %cst0] : memref<?x?xf32>
   }
// VEC1D:for {{.*}} [[ARG_M]] {
   for %i1 = 0 to %M { // not vectorized 
     %a1 = load %A[%i1, %i1] : memref<?x?xf32>
   }
// VEC1D:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i2 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1 
     %r2 = affine_apply (d0) -> (d0) (%i2)
     %a2 = load %A[%r2#0, %cst0] : memref<?x?xf32>
   }
// VEC1D:for [[IV3:%[a-zA-Z0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// VEC1D:   [[APP3:%[a-zA-Z0-9]+]] = affine_apply {{.*}}[[IV3]]
// VEC1D:   [[ALLOC3:%[a-zA-Z0-9]+]] = alloc() : memref<1xvector<128xf32>>
// VEC1D:   [[UNALIGNED3: %[0-9]*]] = "n_d_unaligned_load"(%arg0, [[C1]], [[APP3]], [[ALLOC3]], [[C0]]) : {{.*}}
// VEC1D:   %{{.*}} = load [[ALLOC3]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i3 = 0 to %M { // vectorized
     %r3 = affine_apply (d0) -> (d0) (%i3)
     %a3 = load %A[%cst0, %r3#0] : memref<?x?xf32>
   }
// VEC1D:for [[IV4:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// VEC1D:   for [[IV5:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:   [[APP5:%[0-9]+]] = affine_apply {{.*}}([[IV4]], [[IV5]])
// VEC1D:   [[ALLOC5:%[0-9]+]] = alloc() : memref<1xvector<128xf32>>
// VEC1D:   [[UNALIGNED5:%[0-9]*]] = "n_d_unaligned_load"(%arg0, [[APP5]]#0, [[APP5]]#1, [[ALLOC5]], [[C0]]) : {{.*}}
// VEC1D:   %{{.*}} = load [[ALLOC5]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i4 = 0 to %M { // vectorized 
     for %i5 = 0 to %N { // not vectorized, would vectorize with --test-fastest-varying=1
       %r5 = affine_apply #map1_t (%i4, %i5)
       %a5 = load %A[%r5#0, %r5#1] : memref<?x?xf32>
     }
   }
// VEC1D: for [[IV6:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:   for [[IV7:%[i0-9]*]] = 0 to %{{[0-9]*}} {
   for %i6 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     for %i7 = 0 to %N { // not vectorized, can never vectorize
       %r7 = affine_apply #map2 (%i6, %i7)
       %a7 = load %A[%r7#0, %r7#1] : memref<?x?xf32>
     }
   }
// VEC1D:for [[IV8:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// VEC1D:   for [[IV9:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:   [[APP9:%[0-9]+]] = affine_apply {{.*}}([[IV8]], [[IV9]])
// VEC1D:   [[ALLOC9:%[0-9]+]] = alloc() : memref<1xvector<128xf32>>
// VEC1D:   [[UNALIGNED9:%[0-9]*]] = "n_d_unaligned_load"(%arg0, [[APP9]]#0, [[APP9]]#1, [[ALLOC9]], [[C0]]) : {{.*}}
// VEC1D:   %{{.*}} = load [[ALLOC9]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i8 = 0 to %M { // vectorized
     for %i9 = 0 to %N {
       %r9 = affine_apply #map3 (%i8, %i9)
       %a9 = load %A[%r9#0, %r9#1] : memref<?x?xf32>
     }
   }
// VEC1D: for [[IV10:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:   for [[IV11:%[i0-9]*]] = 0 to %{{[0-9]*}} {
   for %i10 = 0 to %M { // not vectorized, need per load transposes 
     for %i11 = 0 to %N { // not vectorized, need per load transposes 
       %r11 = affine_apply #map1 (%i10, %i11)
       %a11 = load %A[%r11#0, %r11#1] : memref<?x?xf32>
       %r12 = affine_apply #map1_t (%i10, %i11)
       store %a11, %A[%r12#0, %r12#1] : memref<?x?xf32>
     }
   }
// VEC1D: for [[IV12:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:   for [[IV13:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// VEC1D:     for [[IV14:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_P]]{{.}} step 128
   for %i12 = 0 to %M { // not vectorized, can never vectorize
     for %i13 = 0 to %N { // not vectorized, can never vectorize
       for %i14 = 0 to %P { // vectorized
         %r14 = affine_apply #map4 (%i12, %i13, %i14)
         %a14 = load %B[%r14#0, %r14#1, %r14#2] : memref<?x?x?xf32>
       }
     }
   }
// VEC1D:  for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i15 = 0 to %M { // not vectorized due to condition below
     if #set0(%i15) {
       %a15 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
// VEC1D:  for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i16 = 0 to %M { // not vectorized, can't vectorize a vector load
     %a16 = alloc(%M) : memref<?xvector<2xf32>>
     %l16 = load %a16[%i16] : memref<?xvector<2xf32>>
   }
// VEC1D: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
// VEC1D:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// VEC1D:     [[ALLOC18:%[a-zA-Z0-9]+]] = alloc() : memref<1xvector<128xf32>>
// VEC1D:     [[UNALIGNED18: %[0-9]*]] = "n_d_unaligned_load"(%arg0, [[C1]], [[C1]], [[ALLOC18]], [[C0]]) : {{.*}}
// VEC1D:     %{{.*}} = load [[ALLOC18]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %i18 in DFS post-order prevents vectorizing %i17
     for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}

// VEC2D: [[MAPSHIFT0:#map[0-9]*]] = ()[s0] -> (s0 + 31)
// VEC2D: [[MAPSHIFT1:#map[0-9]*]] = ()[s0] -> (s0 + 255)
// VEC2D_T: [[MAPSHIFT0:#map[0-9]*]] = ()[s0] -> (s0 + 31)
// VEC2D_T: [[MAPSHIFT1:#map[0-9]*]] = ()[s0] -> (s0 + 255)
// VEC2D_O: [[MAPSHIFT0:#map[0-9]*]] = ()[s0] -> (s0 + 31)
// VEC2D_O: [[MAPSHIFT1:#map[0-9]*]] = ()[s0] -> (s0 + 255)
// VEC2D_OT: [[MAPSHIFT0:#map[0-9]*]] = ()[s0] -> (s0 + 31)
// VEC2D_OT: [[MAPSHIFT1:#map[0-9]*]] = ()[s0] -> (s0 + 255)
mlfunc @vec2d(%A : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?x?xf32>
   %N = dim %A, 1 : memref<?x?x?xf32>
   %P = dim %A, 2 : memref<?x?x?xf32>
   // VEC2D: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D:   for {{.*}} = 0 to [[MAPSHIFT0]](){{.*}} step 32
   // VEC2D:     for {{.*}} = 0 to [[MAPSHIFT1]](){{.*}} step 256
   // For the case: --test-fastest-varying=1 --test-fastest-varying=0:
   // for %i0 = 0 to %0 {
   //   for %i1 = 0 to #map6()[%1] step 32 {
   //     for %i2 = 0 to #map7()[%2] step 256 {
   //       %3 = alloc() : memref<1xvector<32x256xf32>>
   //       %4 = "n_d_unaligned_load"(%arg0, %i0, %i1, %i2, %3, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   //       %5 = load %3[%c0] : memref<1xvector<32x256xf32>>
   //
   // VEC2D_T: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_T:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_T:     for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // For the case: --test-fastest-varying=0 --test-fastest-varying=1 no
   // vectorization happens because of loop nesting order (i.e. only one of
   // VEC2D and VEC2D_T may ever vectorize).
   //
   // VEC2D_O: for {{.*}} = 0 to [[MAPSHIFT0]](){{.*}} step 32
   // VEC2D_O:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_O:     for {{.*}} = 0 to [[MAPSHIFT1]](){{.*}} step 256
   // For the case: --test-fastest-varying=2 --test-fastest-varying=0:
   // for %i0 = 0 to #map6()[%0] step 32 {
   //   for %i1 = 0 to %1 {
   //     for %i2 = 0 to #map7()[%2] step 256 {
   //       %3 = alloc() : memref<1xvector<32x256xf32>>
   //       %4 = "n_d_unaligned_load"(%arg0, %i0, %i1, %i2, %3, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   //       %5 = load %3[%c0] : memref<1xvector<32x256xf32>>
   //
   // VEC2D_OT: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_OT:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_OT:     for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // For the case: --test-fastest-varying=0 --test-fastest-varying=2 no
   // vectorization happens because of loop nesting order(i.e. only one of
   // VEC2D_O and VEC2D_OT may ever vectorize).
   for %i0 = 0 to %M {
     for %i1 = 0 to %N {
       for %i2 = 0 to %P {
         %a2 = load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
       }
     }
   }
   // VEC2D: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D:     for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // For the case: --test-fastest-varying=1 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order (i.e. only one of
   // VEC2D and VEC2D_T may ever vectorize).
   //
   // VEC2D_T: for {{.*}} = 0 to [[MAPSHIFT0]](){{.*}} step 32
   // VEC2D_T:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_T:     for {{.*}} = 0 to [[MAPSHIFT1]](){{.*}} step 256
   // For the case: --test-fastest-varying=0 --test-fastest-varying=1:
   // for %i3 = 0 to #map1()[%0] step 32 {
   //   for %i4 = 0 to %1 {
   //     for %i5 = 0 to #map2()[%2] step 256 {
   //       %4 = alloc() : memref<1xvector<32x256xf32>>
   //       %5 = "n_d_unaligned_load"(%arg0, %i4, %i5, %i3, %4, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   //       %6 = load %4[%c0] : memref<1xvector<32x256xf32>>
   //
   // VEC2D_O: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_O:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // VEC2D_O:     for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // For the case: --test-fastest-varying=2 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order(i.e. only one of
   // VEC2D_O and VEC2D_OT may ever vectorize).
   //
   // VEC2D_OT: for {{.*}} = 0 to [[MAPSHIFT0]](){{.*}} step 32
   // VEC2D_OT:   for {{.*}} = 0 to [[MAPSHIFT1]](){{.*}} step 256
   // VEC2D_OT:     for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   // For the case: --test-fastest-varying=0 --test-fastest-varying=2:
   // for %i3 = 0 to #map6()[%0] step 32 {
   //   for %i4 = 0 to #map7()[%1] step 256 {
   //     for %i5 = 0 to %2 {
   //       %4 = alloc() : memref<1xvector<32x256xf32>>
   //       %5 = "n_d_unaligned_load"(%arg0, %i4, %i5, %i3, %4, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   //       %6 = load %4[%c0] : memref<1xvector<32x256xf32>>
   for %i3 = 0 to %M {
     for %i4 = 0 to %N {
       for %i5 = 0 to %P {
         %a5 = load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}

mlfunc @vec2d_imperfectly_nested(%A : memref<?x?x?xf32>) {
   %0 = dim %A, 0 : memref<?x?x?xf32>
   %1 = dim %A, 1 : memref<?x?x?xf32>
   %2 = dim %A, 2 : memref<?x?x?xf32>
   // VEC2D_T: for %i0 = 0 to #map{{.}}()[%0] step 32 {
   // VEC2D_T:   for %i1 = 0 to #map{{.}}()[%1] step 256 {
   // VEC2D_T:     for %i2 = 0 to %2 {
   // VEC2D_T:       %3 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_T:       %4 = "n_d_unaligned_load"(%arg0, %i2, %i1, %i0, %3, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_T:       %5 = load %3[%c0] : memref<1xvector<32x256xf32>>
   // VEC2D_T:   for %i3 = 0 to %1 {
   // VEC2D_T:     for %i4 = 0 to #map{{.}}()[%2] step 256 {
   // VEC2D_T:       %6 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_T:       %7 = "n_d_unaligned_load"(%arg0, %i3, %i4, %i0, %6, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_T:       %8 = load %6[%c0] : memref<1xvector<32x256xf32>>
   // VEC2D_T:     for %i5 = 0 to #map{{.}}()[%2] step 256 {
   // VEC2D_T:       %9 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_T:       %10 = "n_d_unaligned_load"(%arg0, %i3, %i5, %i0, %9, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_T:       %11 = load %9[%c0] : memref<1xvector<32x256xf32>>
   //
   // VEC2D_OT: for %i0 = 0 to #map{{.}}()[%0] step 32 {
   // VEC2D_OT:   for %i1 = 0 to %1 {
   // VEC2D_OT:     for %i2 = 0 to #map{{.}}()[%2] step 256 {
   // VEC2D_OT:       %3 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_OT:       %4 = "n_d_unaligned_load"(%arg0, %i2, %i1, %i0, %3, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_OT:       %5 = load %3[%c0] : memref<1xvector<32x256xf32>>
   // VEC2D_OT:   for %i3 = 0 to #map{{.}}()[%1] step 256 {
   // VEC2D_OT:     for %i4 = 0 to %2 {
   // VEC2D_OT:       %6 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_OT:       %7 = "n_d_unaligned_load"(%arg0, %i3, %i4, %i0, %6, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_OT:       %8 = load %6[%c0] : memref<1xvector<32x256xf32>>
   // VEC2D_OT:     for %i5 = 0 to %2 {
   // VEC2D_OT:       %9 = alloc() : memref<1xvector<32x256xf32>>
   // VEC2D_OT:       %10 = "n_d_unaligned_load"(%arg0, %i3, %i5, %i0, %9, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x256xf32>>, index)
   // VEC2D_OT:       %11 = load %9[%c0] : memref<1xvector<32x256xf32>>
   for %i0 = 0 to %0 {
     for %i1 = 0 to %1 {
       for %i2 = 0 to %2 {
         %a2 = load %A[%i2, %i1, %i0] : memref<?x?x?xf32>
       }
     }
     for %i3 = 0 to %1 {
       for %i4 = 0 to %2 {
         %a4 = load %A[%i3, %i4, %i0] : memref<?x?x?xf32>
       }
       for %i5 = 0 to %2 {
         %a5 = load %A[%i3, %i5, %i0] : memref<?x?x?xf32>
       }
     }
   }
   return
}

mlfunc @vec3d(%A : memref<?x?x?xf32>) {
   %0 = dim %A, 0 : memref<?x?x?xf32>
   %1 = dim %A, 1 : memref<?x?x?xf32>
   %2 = dim %A, 2 : memref<?x?x?xf32>
   // VEC3D: for %i0 = 0 to %0 {
   // VEC3D:   for %i1 = 0 to %0 {
   // VEC3D:     for %i2 = 0 to #map{{.}}()[%0] step 32 {
   // VEC3D:       for %i3 = 0 to #map{{.}}()[%1] step 64 {
   // VEC3D:         for %i4 = 0 to #map{{.}}()[%2] step 256 {
   // VEC3D:           %3 = alloc() : memref<1xvector<32x64x256xf32>>
   // VEC3D:           %4 = "n_d_unaligned_load"(%arg0, %i2, %i3, %i4, %3, %c0) : (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x64x256xf32>>, index) -> (memref<?x?x?xf32>, index, index, index, memref<1xvector<32x64x256xf32>>, index)
   // VEC3D:           %5 = load %3[%c0] : memref<1xvector<32x64x256xf32>>
   for %t0 = 0 to %0 {
     for %t1 = 0 to %0 {
       for %i0 = 0 to %0 {
         for %i1 = 0 to %1 {
           for %i2 = 0 to %2 {
             %a2 = load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
           }
         }
       }
     }
   }
   return
}