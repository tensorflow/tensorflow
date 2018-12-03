// RUN: mlir-opt %s -vectorize -virtual-vector-size 32 -virtual-vector-size 256 --test-fastest-varying=1 --test-fastest-varying=0 | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1_d0d1:map[0-9]+]] = (d0, d1) -> (d0, d1)

mlfunc @vec2d(%A : memref<?x?x?xf32>) {
   %M = dim %A, 0 : memref<?x?x?xf32>
   %N = dim %A, 1 : memref<?x?x?xf32>
   %P = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: for  {{.*}} = 0 to %0 {
   // CHECK:   for {{.*}} = 0 to %1 step 32
   // CHECK:     for {{.*}} = 0 to %2 step 256
   // Example:
   // for %i0 = 0 to %0 {
   //   for %i1 = 0 to %1 step 32 {
   //     for %i2 = 0 to %2 step 256 {
   //       %3 = "vector_transfer_read"(%arg0, %i0, %i1, %i2) : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
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
   // For the case: --test-fastest-varying=1 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order .
   for %i3 = 0 to %M {
     for %i4 = 0 to %N {
       for %i5 = 0 to %P {
         %a5 = load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}

mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // CHECK: [[C1:%.*]] = constant splat<vector<32x256xf32>, 1.000000e+00> : vector<32x256xf32>
      // CHECK: vector_transfer_write [[C1]], {{.*}} {permutation_map: #[[map_proj_d0d1_d0d1]]} : vector<32x256xf32>, memref<?x?xf32>, index, index
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // CHECK: [[C3:%.*]] = constant splat<vector<32x256xf32>, 2.000000e+00> : vector<32x256xf32>
      // CHECK: vector_transfer_write [[C3]], {{.*}} {permutation_map: #[[map_proj_d0d1_d0d1]]}  : vector<32x256xf32>, memref<?x?xf32>, index, index
      // non-scoped %f2
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  for %i4 = 0 to %M {
    for %i5 = 0 to %N {
      // CHECK: [[A5:%.*]] = vector_transfer_read %0, {{.*}} {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<32x256xf32>
      // CHECK: [[B5:%.*]] = vector_transfer_read %1, {{.*}} {permutation_map: #[[map_proj_d0d1_d0d1]]} : (memref<?x?xf32>, index, index) -> vector<32x256xf32>
      // CHECK: [[S5:%.*]] = addf [[A5]], [[B5]] : vector<32x256xf32>
      // CHECK: [[SPLAT1:%.*]] = constant splat<vector<32x256xf32>, 1.000000e+00> : vector<32x256xf32>
      // CHECK: [[S6:%.*]] = addf [[S5]], [[SPLAT1]] : vector<32x256xf32>
      // CHECK: [[SPLAT2:%.*]] = constant splat<vector<32x256xf32>, 2.000000e+00> : vector<32x256xf32>
      // CHECK: [[S7:%.*]] = addf [[S5]], [[SPLAT2]] : vector<32x256xf32>
      // CHECK: [[S8:%.*]] = addf [[S7]], [[S6]] : vector<32x256xf32>
      // CHECK: vector_transfer_write [[S8]], {{.*}} {permutation_map: #[[map_proj_d0d1_d0d1]]} : vector<32x256xf32>, memref<?x?xf32>, index, index
      //
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

