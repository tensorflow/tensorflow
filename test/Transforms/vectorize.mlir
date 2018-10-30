// RUN: mlir-opt %s -vectorize -virtual-vector-size 128 --test-fastest-varying=0 | FileCheck %s

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
// CHECK: [[MAPSHIFT:#map[0-9]*]] = ()[s0] -> (s0 + 127)
mlfunc @vec1d(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK: [[ARG_M:%[0-9]+]] = dim %arg0, 0 : memref<?x?xf32>
// CHECK: [[ARG_N:%[0-9]+]] = dim %arg0, 1 : memref<?x?xf32>
// CHECK: [[ARG_P:%[0-9]+]] = dim %arg1, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
// CHECK: [[C1:%[a-z0-9_]+]] = constant 0 : index
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
// CHECK:for {{.*}} [[ARG_M]] {
   for %i1 = 0 to %M { // not vectorized 
     %a1 = load %A[%i1, %i1] : memref<?x?xf32>
   }
// CHECK:   for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i2 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1 
     %r2 = affine_apply (d0) -> (d0) (%i2)
     %a2 = load %A[%r2#0, %cst0] : memref<?x?xf32>
   }
// CHECK:for [[IV3:%[a-zA-Z0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// CHECK:   [[APP3:%[a-zA-Z0-9]+]] = affine_apply {{.*}}[[IV3]]
// CHECK:   [[ALLOC3:%[a-zA-Z0-9]+]] = alloc() : memref<1xvector<128xf32>>
// CHECK:   [[UNALIGNED3: %[0-9]*]] = "n_d_unaligned_load"(%arg0, [[C1]], [[APP3]], [[ALLOC3]], [[C0]]) : {{.*}}
// CHECK:   %{{.*}} = load [[ALLOC3]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i3 = 0 to %M { // vectorized
     %r3 = affine_apply (d0) -> (d0) (%i3)
     %a3 = load %A[%cst0, %r3#0] : memref<?x?xf32>
   }
// CHECK:for [[IV4:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// CHECK:   for [[IV5:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   [[APP5:%[0-9]+]] = affine_apply {{.*}}([[IV4]], [[IV5]])
// CHECK:   [[ALLOC5:%[0-9]+]] = alloc() : memref<1xvector<128xf32>>
// CHECK:   [[UNALIGNED5:%[0-9]*]] = "n_d_unaligned_load"(%arg0, [[APP5]]#0, [[APP5]]#1, [[ALLOC5]], [[C0]]) : {{.*}}
// CHECK:   %{{.*}} = load [[ALLOC5]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i4 = 0 to %M { // vectorized 
     for %i5 = 0 to %N { // not vectorized, would vectorize with --test-fastest-varying=1
       %r5 = affine_apply #map1_t (%i4, %i5)
       %a5 = load %A[%r5#0, %r5#1] : memref<?x?xf32>
     }
   }
// CHECK: for [[IV6:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV7:%[i0-9]*]] = 0 to %{{[0-9]*}} {
   for %i6 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     for %i7 = 0 to %N { // not vectorized, can never vectorize
       %r7 = affine_apply #map2 (%i6, %i7)
       %a7 = load %A[%r7#0, %r7#1] : memref<?x?xf32>
     }
   }
// CHECK:for [[IV8:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// CHECK:   for [[IV9:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   [[APP9:%[0-9]+]] = affine_apply {{.*}}([[IV8]], [[IV9]])
// CHECK:   [[ALLOC9:%[0-9]+]] = alloc() : memref<1xvector<128xf32>>
// CHECK:   [[UNALIGNED9:%[0-9]*]] = "n_d_unaligned_load"(%arg0, [[APP9]]#0, [[APP9]]#1, [[ALLOC9]], [[C0]]) : {{.*}}
// CHECK:   %{{.*}} = load [[ALLOC9]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i8 = 0 to %M { // vectorized
     for %i9 = 0 to %N {
       %r9 = affine_apply #map3 (%i8, %i9)
       %a9 = load %A[%r9#0, %r9#1] : memref<?x?xf32>
     }
   }
// CHECK: for [[IV10:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV11:%[i0-9]*]] = 0 to %{{[0-9]*}} {
   for %i10 = 0 to %M { // not vectorized, need per load transposes 
     for %i11 = 0 to %N { // not vectorized, need per load transposes 
       %r11 = affine_apply #map1 (%i10, %i11)
       %a11 = load %A[%r11#0, %r11#1] : memref<?x?xf32>
       %r12 = affine_apply #map1_t (%i10, %i11)
       store %a11, %A[%r12#0, %r12#1] : memref<?x?xf32>
     }
   }
// CHECK: for [[IV12:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV13:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:     for [[IV14:%[i0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_P]]{{.}} step 128
   for %i12 = 0 to %M { // not vectorized, can never vectorize
     for %i13 = 0 to %N { // not vectorized, can never vectorize
       for %i14 = 0 to %P { // vectorized
         %r14 = affine_apply #map4 (%i12, %i13, %i14)
         %a14 = load %B[%r14#0, %r14#1, %r14#2] : memref<?x?x?xf32>
       }
     }
   }
// CHECK:  for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i15 = 0 to %M { // not vectorized due to condition below
     if #set0(%i15) {
       %a15 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
// CHECK:  for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   for %i16 = 0 to %M { // not vectorized, can't vectorize a vector load
     %a16 = alloc(%M) : memref<?xvector<2xf32>>
     %l16 = load %a16[%i16] : memref<?xvector<2xf32>>
   }
// CHECK: for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[MAPSHIFT]](){{.}}[[ARG_M]]{{.}} step 128
// CHECK:     [[ALLOC18:%[a-zA-Z0-9]+]] = alloc() : memref<1xvector<128xf32>>
// CHECK:     [[UNALIGNED18: %[0-9]*]] = "n_d_unaligned_load"(%arg0, [[C1]], [[C1]], [[ALLOC18]], [[C0]]) : {{.*}}
// CHECK:     %{{.*}} = load [[ALLOC18]]{{.}}[[C0]]{{.}} : memref<1xvector<128xf32>>
   for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %i18 in DFS post-order prevents vectorizing %i17
     for %i18 = 0 to %M { // vectorized due to scalar -> vector 
       %a18 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}
