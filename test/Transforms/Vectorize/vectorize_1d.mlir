// RUN: mlir-opt %s -affine-vectorize -virtual-vector-size 128 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1_0:map[0-9]+]] = (d0, d1) -> (0)
// CHECK: #[[map_proj_d0d1_d1:map[0-9]+]] = (d0, d1) -> (d1)

#map0 = (d0) -> (d0)
#mapadd1 = (d0) -> (d0 + 1)
#mapadd2 = (d0) -> (d0 + 2)
#mapadd3 = (d0) -> (d0 + 3)
#set0 = (i) : (i >= 0)

// Maps introduced to vectorize fastest varying memory index.
// CHECK-LABEL: func @vec1d_1
func @vec1d_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: %[[C0:[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: for {{.*}} step 128
// CHECK-NEXT: %{{.*}} = affine.apply #map0(%[[C0]])
// CHECK-NEXT: %{{.*}} = affine.apply #map0(%[[C0]])
// CHECK-NEXT: {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i0 = 0 to %M { // vectorized due to scalar -> vector
     %a0 = affine.load %A[%cst0, %cst0] : memref<?x?xf32>
   }
   return
}

// CHECK-LABEL: func @vec1d_2
func @vec1d_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: %[[C0:[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:for [[IV3:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i3 = 0 to %M { // vectorized
     %a3 = affine.load %A[%cst0, %i3] : memref<?x?xf32>
   }
   return
}

// CHECK-LABEL: func @vec1d_3
func @vec1d_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: %[[C0:[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %arg0, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %arg0, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %arg1, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:for [[IV8:%[arg0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   for [[IV9:%[arg0-9]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   %[[APP9_0:[0-9]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   %[[APP9_1:[0-9]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   {{.*}} = vector.transfer_read %{{.*}}[%[[APP9_0]], %[[APP9_1]]], %{{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i8 = 0 to %M { // vectorized
     affine.for %i9 = 0 to %N {
       %a9 = affine.load %A[%i9, %i8 + %i9] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vector_add_2d
func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      // CHECK: [[C1:%.*]] = constant dense<1.000000e+00> : vector<128xf32>
      // CHECK: vector.transfer_write [[C1]], {{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>
      // non-scoped %f1
      affine.store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      // CHECK: [[C3:%.*]] = constant dense<2.000000e+00> : vector<128xf32>
      // CHECK: vector.transfer_write [[C3]], {{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>
      // non-scoped %f2
      affine.store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      // CHECK: [[A5:%.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : memref<?x?xf32>, vector<128xf32>
      // CHECK: [[B5:%.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : memref<?x?xf32>, vector<128xf32>
      // CHECK: [[S5:%.*]] = addf [[A5]], [[B5]] : vector<128xf32>
      // CHECK: [[SPLAT1:%.*]] = constant dense<1.000000e+00> : vector<128xf32>
      // CHECK: [[S6:%.*]] = addf [[S5]], [[SPLAT1]] : vector<128xf32>
      // CHECK: [[SPLAT2:%.*]] = constant dense<2.000000e+00> : vector<128xf32>
      // CHECK: [[S7:%.*]] = addf [[S5]], [[SPLAT2]] : vector<128xf32>
      // CHECK: [[S8:%.*]] = addf [[S7]], [[S6]] : vector<128xf32>
      // CHECK: vector.transfer_write [[S8]], {{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : vector<128xf32>, memref<?x?xf32>
      %a5 = affine.load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = affine.load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = addf %a5, %b5 : f32
      // non-scoped %f1
      %s6 = addf %s5, %f1 : f32
      // non-scoped %f2
      %s7 = addf %s5, %f2 : f32
      // diamond dependency.
      %s8 = addf %s7, %s6 : f32
      affine.store %s8, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = constant 7 : index
  %c42 = constant 42 : index
  %res = affine.load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}

// CHECK-LABEL: func @vec_rejected_1
func @vec_rejected_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:for {{.*}} [[ARG_M]] {
   affine.for %i1 = 0 to %M { // not vectorized
     %a1 = affine.load %A[%i1, %i1] : memref<?x?xf32>
   }
   return
}

// CHECK-LABEL: func @vec_rejected_2
func @vec_rejected_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:   affine.for %{{.*}}{{[0-9]*}} = 0 to [[ARG_M]] {
   affine.for %i2 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     %a2 = affine.load %A[%i2, %cst0] : memref<?x?xf32>
   }
   return
}

// CHECK-LABEL: func @vec_rejected_3
func @vec_rejected_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:for [[IV4:%[arg0-9]+]] = 0 to [[ARG_M]] step 128 {
// CHECK-NEXT:   for [[IV5:%[arg0-9]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_d1]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i4 = 0 to %M { // vectorized
     affine.for %i5 = 0 to %N { // not vectorized, would vectorize with --test-fastest-varying=1
       %a5 = affine.load %A[%i5, %i4] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_4
func @vec_rejected_4(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: for [[IV6:%[arg0-9]*]] = 0 to [[ARG_M]] {
// CHECK-NEXT:   for [[IV7:%[arg0-9]*]] = 0 to [[ARG_N]] {
   affine.for %i6 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     affine.for %i7 = 0 to %N { // not vectorized, can never vectorize
       %a7 = affine.load %A[%i6 + %i7, %i6] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_5
func @vec_rejected_5(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: for [[IV10:%[arg0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV11:%[arg0-9]*]] = 0 to %{{[0-9]*}} {
   affine.for %i10 = 0 to %M { // not vectorized, need per load transposes
     affine.for %i11 = 0 to %N { // not vectorized, need per load transposes
       %a11 = affine.load %A[%i10, %i11] : memref<?x?xf32>
       affine.store %a11, %A[%i11, %i10] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_6
func @vec_rejected_6(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: for [[IV12:%[arg0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV13:%[arg0-9]*]] = 0 to %{{[0-9]*}} {
// CHECK:     for [[IV14:%[arg0-9]+]] = 0 to [[ARG_P]] step 128
   affine.for %i12 = 0 to %M { // not vectorized, can never vectorize
     affine.for %i13 = 0 to %N { // not vectorized, can never vectorize
       affine.for %i14 = 0 to %P { // vectorized
         %a14 = affine.load %B[%i13, %i12 + %i13, %i12 + %i14] : memref<?x?x?xf32>
       }
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_7
func @vec_rejected_7(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:  affine.for %{{.*}}{{[0-9]*}} = 0 to %{{[0-9]*}} {
   affine.for %i16 = 0 to %M { // not vectorized, can't vectorize a vector load
     %a16 = alloc(%M) : memref<?xvector<2xf32>>
     %l16 = affine.load %a16[%i16] : memref<?xvector<2xf32>>
   }
   return
}

// CHECK-LABEL: func @vec_rejected_8
func @vec_rejected_8(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: %[[C0:[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: affine.for %{{.*}}{{[0-9]*}} = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK:     %{{.*}} = affine.apply #map0(%{{.*}})
// CHECK:     %{{.*}} = affine.apply #map0(%{{.*}})
// CHECK:     {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %{{.*}} in DFS post-order prevents vectorizing %{{.*}}
     affine.for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = affine.load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_9
func @vec_rejected_9(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %{{.*}} = constant 0.0{{.*}}: f32
// CHECK-DAG: %[[C0:[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK: affine.for %{{.*}}{{[0-9]*}} = 0 to %{{[0-9]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK:      %{{.*}} = affine.apply #map0(%{{.*}})
// CHECK-NEXT: %{{.*}} = affine.apply #map0(%{{.*}})
// CHECK-NEXT: {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %i18 in DFS post-order prevents vectorizing %{{.*}}
     affine.for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = affine.load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}

// CHECK-LABEL: func @vec_rejected_10
func @vec_rejected_10(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: [[C0:%[a-z0-9_]+]] = constant 0 : index
// CHECK-DAG: [[ARG_M:%[0-9]+]] = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9]+]] = dim %{{.*}}, 1 : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9]+]] = dim %{{.*}}, 2 : memref<?x?x?xf32>
   %M = dim %A, 0 : memref<?x?xf32>
   %N = dim %A, 1 : memref<?x?xf32>
   %P = dim %B, 2 : memref<?x?x?xf32>
   %cst0 = constant 0 : index
//
// CHECK:  affine.for %{{.*}}{{[0-9]*}} = 0 to %{{[0-9]*}} {
   affine.for %i15 = 0 to %M { // not vectorized due to condition below
     affine.if #set0(%i15) {
       %a15 = affine.load %A[%cst0, %cst0] : memref<?x?xf32>
     }
   }
   return
}

// This should not vectorize and should not crash.
// CHECK-LABEL: @vec_rejected_11
func @vec_rejected_11(%A : memref<?x?xf32>, %C : memref<?x?xf32>) {
  %N = dim %A, 0 : memref<?x?xf32>
  affine.for %i = 0 to %N {
// CHECK-NOT: vector
    %a = affine.load %A[%i, %i] : memref<?x?xf32> // not vectorized
    affine.for %j = 0 to %N {
      %b = affine.load %A[%i, %j] : memref<?x?xf32> // may be vectorized
// CHECK-NOT: vector
      %c = addf %a, %b : f32 // not vectorized because %a wasn't
// CHECK-NOT: vector
      affine.store %c, %C[%i, %j] : memref<?x?xf32> // not vectorized because %c wasn't
    }
  }
  return
}

// This should not vectorize due to the sequential dependence in the loop.
// CHECK-LABEL: @vec_rejected_sequential
func @vec_rejected_sequential(%A : memref<?xf32>) {
  %N = dim %A, 0 : memref<?xf32>
  affine.for %i = 0 to %N {
    // CHECK-NOT: vector
    %a = affine.load %A[%i] : memref<?xf32>
    // CHECK-NOT: vector
    affine.store %a, %A[%i + 1] : memref<?xf32>
  }
  return
}
