// RUN: mlir-opt %s -vectorize -virtual-vector-size 128 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1_0:map[0-9]+]] = (d0, d1) -> (0)
// CHECK: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)

#map0 = (d0) -> (d0)
#mapadd1 = (d0) -> (d0 + 1)
#mapadd2 = (d0) -> (d0 + 2)
#mapadd3 = (d0) -> (d0 + 3)
#set0 = (i) : (i >= 0)

// Maps introduced to vectorize fastest varying memory index.
func @vec1d(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %arg0, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %arg0, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %arg1, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: for {{.*}} step 128
// CHECK-NEXT: {{.*}} = vector_transfer_read %arg0, [[C0]], [[C0]] {permutation_map: #[[map_proj_d0d1_0]]} : (memref<?x?xf32>, index, index) -> vector<128xf32>
   affine.for %i0 = 0 to %M { // vectorized due to scalar -> vector
     %a0 = load %A[%cst0, %cst0] : memref<?x?xf32>
   }
//
// CHECK:for {{.*}} [[ARG_M]] {
   affine.for %i1 = 0 to %M { // not vectorized
     %a1 = load %A[%i1, %i1] : memref<?x?xf32>
   }
//
// CHECK:   affine.for %i{{[0-9]*}} = 0 to [[ARG_M]] {
   affine.for %i2 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     %r2 = affine.apply (d0) -> (d0) (%i2)
     %a2 = load %A[%r2#0, %cst0] : memref<?x?xf32>
   }
//
// CHECK:for [[IV3:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   [[APP3:%[a-zA-Z0-9]+]] = affine.apply {{.*}}[[IV3]]
// CHECK-NEXT:   {{.*}} = vector_transfer_read %arg0, [[C0]], [[APP3]] {permutation_map: #[[map_proj_d0d1_d1]]} : {{.*}} -> vector<128xf32>
   affine.for %i3 = 0 to %M { // vectorized
     %r3 = affine.apply (d0) -> (d0) (%i3)
     %a3 = load %A[%cst0, %r3#0] : memref<?x?xf32>
   }
//
// CHECK:for [[IV4:%[i0-9]+]] = 0 to [[ARG_M]] step 128 {
// CHECK-NEXT:   for [[IV5:%[i0-9]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   [[APP50:%[0-9]+]] = affine.apply {{.*}}([[IV4]], [[IV5]])
// CHECK-NEXT:   [[APP51:%[0-9]+]] = affine.apply {{.*}}([[IV4]], [[IV5]])
// CHECK-NEXT:   {{.*}} = vector_transfer_read %arg0, [[APP50]], [[APP51]] {permutation_map: #[[map_proj_d0d1_d1]]} : {{.*}} -> vector<128xf32>
   affine.for %i4 = 0 to %M { // vectorized
     affine.for %i5 = 0 to %N { // not vectorized, would vectorize with --test-fastest-varying=1
       %r50 = affine.apply (d0, d1) -> (d1) (%i4, %i5)
       %r51 = affine.apply (d0, d1) -> (d0) (%i4, %i5)
       %a5 = load %A[%r50, %r51] : memref<?x?xf32>
     }
   }
//
// CHECK: for [[IV6:%[i0-9]*]] = 0 to [[ARG_M]] {
// CHECK-NEXT:   for [[IV7:%[i0-9]*]] = 0 to [[ARG_N]] {
   affine.for %i6 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     affine.for %i7 = 0 to %N { // not vectorized, can never vectorize
       %r70 = affine.apply (d0, d1) -> (d1 + d0) (%i6, %i7)
       %r71 = affine.apply (d0, d1) -> (d0) (%i6, %i7)
       %a7 = load %A[%r70, %r71] : memref<?x?xf32>
     }
   }
//
// CHECK:for [[IV8:%[i0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   for [[IV9:%[i0-9]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   [[APP9_0:%[0-9]+]] = affine.apply {{.*}}([[IV8]], [[IV9]])
// CHECK-NEXT:   [[APP9_1:%[0-9]+]] = affine.apply {{.*}}([[IV8]], [[IV9]])
// CHECK-NEXT:   {{.*}} = vector_transfer_read %arg0, [[APP9_0]], [[APP9_1]] {permutation_map: #[[map_proj_d0d1_d1]]} : {{.*}} -> vector<128xf32>
   affine.for %i8 = 0 to %M { // vectorized
     affine.for %i9 = 0 to %N {
       %r90 = affine.apply (d0, d1) -> (d1) (%i8, %i9)
       %r91 = affine.apply (d0, d1) -> (d0 + d1) (%i8, %i9)
       %a9 = load %A[%r90, %r91] : memref<?x?xf32>
     }
   }
//
// CHECK: for [[IV10:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV11:%[i0-9]*]] = 0 to %{{[0-9]*}} {
   affine.for %i10 = 0 to %M { // not vectorized, need per load transposes
     affine.for %i11 = 0 to %N { // not vectorized, need per load transposes
       %r11_0 = affine.apply (d0, d1) -> (d0) (%i10, %i11)
       %r11_1 = affine.apply (d0, d1) -> (d1) (%i10, %i11)
       %a11 = load %A[%r11_0, %r11_1] : memref<?x?xf32>
       %r12_0 = affine.apply (d0, d1) -> (d1) (%i10, %i11)
       %r12_1 = affine.apply (d0, d1) -> (d0) (%i10, %i11)
       store %a11, %A[%r12_0, %r12_1] : memref<?x?xf32>
     }
   }
//
// CHECK: for [[IV12:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV13:%[i0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:     for [[IV14:%[i0-9]+]] = 0 to [[ARG_P]] step 128
   affine.for %i12 = 0 to %M { // not vectorized, can never vectorize
     affine.for %i13 = 0 to %N { // not vectorized, can never vectorize
       affine.for %i14 = 0 to %P { // vectorized
         %r14_0 = affine.apply (d0, d1, d2) -> (d1) (%i12, %i13, %i14)
         %r14_1 = affine.apply (d0, d1, d2) -> (d0 + d1) (%i12, %i13, %i14)
         %r14_2 = affine.apply (d0, d1, d2) -> (d0 + d2) (%i12, %i13, %i14)
         %a14 = load %B[%r14_0, %r14_1, %r14_2] : memref<?x?x?xf32>
       }
     }
   }
//
// CHECK:  affine.for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   affine.for %i15 = 0 to %M { // not vectorized due to condition below
     if #set0(%i15) {
       %a15 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
//
// CHECK:  affine.for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
   affine.for %i16 = 0 to %M { // not vectorized, can't vectorize a vector load
     %a16 = alloc(%M) : memref<?xvector<2xf32>>
     %l16 = load %a16[%i16] : memref<?xvector<2xf32>>
   }
//
// CHECK: affine.for %i{{[0-9]*}} = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK:     {{.*}} = vector_transfer_read %arg0, [[C0]], [[C0]] {permutation_map: #[[map_proj_d0d1_0]]} : {{.*}} -> vector<128xf32>
   affine.for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %i18 in DFS post-order prevents vectorizing %i17
     affine.for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}

func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      // CHECK: [[C1:%.*]] = constant splat<vector<128xf32>, 1.000000e+00> : vector<128xf32>
      // CHECK: vector_transfer_write [[C1]], {{.*}} {permutation_map: #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>, index, index
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      // CHECK: [[C3:%.*]] = constant splat<vector<128xf32>, 2.000000e+00> : vector<128xf32>
      // CHECK: vector_transfer_write [[C3]], {{.*}} {permutation_map: #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>, index, index
      // non-scoped %f2
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      // CHECK: [[A5:%.*]] = vector_transfer_read %0, {{.*}} {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<128xf32>
      // CHECK: [[B5:%.*]] = vector_transfer_read %1, {{.*}} {permutation_map: #[[map_proj_d0d1_d1]]} : (memref<?x?xf32>, index, index) -> vector<128xf32>
      // CHECK: [[S5:%.*]] = addf [[A5]], [[B5]] : vector<128xf32>
      // CHECK: [[SPLAT1:%.*]] = constant splat<vector<128xf32>, 1.000000e+00> : vector<128xf32>
      // CHECK: [[S6:%.*]] = addf [[S5]], [[SPLAT1]] : vector<128xf32>
      // CHECK: [[SPLAT2:%.*]] = constant splat<vector<128xf32>, 2.000000e+00> : vector<128xf32>
      // CHECK: [[S7:%.*]] = addf [[S5]], [[SPLAT2]] : vector<128xf32>
      // CHECK: [[S8:%.*]] = addf [[S7]], [[S6]] : vector<128xf32>
      // CHECK: vector_transfer_write [[S8]], {{.*}} {permutation_map: #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>, index, index
      %a5 = load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = addf %a5, %b5 : f32
      // non-scoped %f1
      %s6 = addf %s5, %f1 : f32
      // non-scoped %f2
      %s7 = addf %s5, %f2 : f32
      // diamond dependency.
      %s8 = addf %s7, %s6 : f32
      store %s8, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = constant 7 : index
  %c42 = constant 42 : index
  %res = load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}

// This should not vectorize and should not crash.
// CHECK-LABEL: @vec_rejected
func @vec_rejected(%A : memref<?x?xf32>, %C : memref<?x?xf32>) {
  %N = dim %A, 0 : memref<?x?xf32>
  affine.for %i = 0 to %N {
// CHECK-NOT: vector
    %a = load %A[%i, %i] : memref<?x?xf32> // not vectorized
    affine.for %j = 0 to %N {
      %b = load %A[%i, %j] : memref<?x?xf32> // may be vectorized
// CHECK-NOT: vector
      %c = addf %a, %b : f32 // not vectorized because %a wasn't
// CHECK-NOT: vector
      store %c, %C[%i, %j] : memref<?x?xf32> // not vectorized because %c wasn't
    }
  }
  return
}
