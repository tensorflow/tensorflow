// RUN: mlir-opt %s -lower-vector-transfers | FileCheck %s

// CHECK: #[[ADD:map[0-9]+]] = (d0, d1) -> (d0 + d1)
// CHECK: #[[SUB:map[0-9]+]] = ()[s0] -> (s0 - 1)

// CHECK-LABEL: func @materialize_read_1d() {
func @materialize_read_1d() {
  %A = alloc () : memref<7x42xf32>
  affine.for %i0 = 0 to 7 step 4 {
    affine.for %i1 = 0 to 42 step 4 {
      %f1 = vector_transfer_read %A, %i0, %i1 {permutation_map: (d0, d1) -> (d0)} : (memref<7x42xf32>, index, index) -> vector<4xf32>
      %ip1 = affine.apply (d0) -> (d0 + 1) (%i1)
      %f2 = vector_transfer_read %A, %i0, %ip1 {permutation_map: (d0, d1) -> (d0)} : (memref<7x42xf32>, index, index) -> vector<4xf32>
      %ip2 = affine.apply (d0) -> (d0 + 2) (%i1)
      %f3 = vector_transfer_read %A, %i0, %ip2 {permutation_map: (d0, d1) -> (d0)} : (memref<7x42xf32>, index, index) -> vector<4xf32>
      %ip3 = affine.apply (d0) -> (d0 + 3) (%i1)
      %f4 = vector_transfer_read %A, %i0, %ip3 {permutation_map: (d0, d1) -> (d0)} : (memref<7x42xf32>, index, index) -> vector<4xf32>
      // Both accesses in the load must be clipped otherwise %i1 + 2 and %i1 + 3 will go out of bounds.
      // CHECK: {{.*}} = select
      // CHECK: %[[FILTERED1:.*]] = select
      // CHECK: {{.*}} = select
      // CHECK: %[[FILTERED2:.*]] = select
      // CHECK-NEXT: %{{.*}} = load {{.*}}[%[[FILTERED1]], %[[FILTERED2]]] : memref<7x42xf32>
    }
  }
  return
}

// CHECK-LABEL: func @materialize_read_1d_partially_specialized
func @materialize_read_1d_partially_specialized(%dyn1 : index, %dyn2 : index, %dyn4 : index) {
  %A = alloc (%dyn1, %dyn2, %dyn4) : memref<7x?x?x42x?xf32>
  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to %dyn1 {
      affine.for %i2 = 0 to %dyn2 {
        affine.for %i3 = 0 to 42 step 2 {
          affine.for %i4 = 0 to %dyn4 {
            %f1 = vector_transfer_read %A, %i0, %i1, %i2, %i3, %i4 {permutation_map: (d0, d1, d2, d3, d4) -> (d3)} : ( memref<7x?x?x42x?xf32>, index, index, index, index, index) -> vector<4xf32>
            %i3p1 = affine.apply (d0) -> (d0 + 1) (%i3)
            %f2 = vector_transfer_read %A, %i0, %i1, %i2, %i3p1, %i4 {permutation_map: (d0, d1, d2, d3, d4) -> (d3)} : ( memref<7x?x?x42x?xf32>, index, index, index, index, index) -> vector<4xf32>
          }
        }
      }
    }
  }
  // CHECK: %[[tensor:[0-9]+]] = alloc
  // CHECK-NOT: {{.*}} dim %[[tensor]], 0
  // CHECK: {{.*}} dim %[[tensor]], 1
  // CHECK: {{.*}} dim %[[tensor]], 2
  // CHECK-NOT: {{.*}} dim %[[tensor]], 3
  // CHECK: {{.*}} dim %[[tensor]], 4
  return
}

// CHECK-LABEL: func @materialize_read(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
func @materialize_read(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-NEXT:  %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %arg0 step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %arg1 {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %arg2 {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %arg3 step 5 {
  //      CHECK:          %[[D0:.*]] = dim %0, 0 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D1:.*]] = dim %0, 1 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D2:.*]] = dim %0, 2 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D3:.*]] = dim %0, 3 : memref<?x?x?x?xf32>
  //      CHECK:          %[[ALLOC:.*]] = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:          %[[VECTOR_VIEW:.*]] = vector_type_cast %[[ALLOC]] : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          affine.for %[[I4:.*]] = 0 to 3 {
  // CHECK-NEXT:            affine.for %[[I5:.*]] = 0 to 4 {
  // CHECK-NEXT:              affine.for %[[I6:.*]] = 0 to 5 {
  // CHECK-NEXT:                %[[C0:.*]] = constant 0 : index
  // CHECK-NEXT:                %[[C1:.*]] = constant 1 : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D0]]]
  // CHECK-NEXT:                {{.*}} = select
  // CHECK-NEXT:                %[[L0:.*]] = select
  //
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D1]]]
  // CHECK-NEXT:                {{.*}} = select
  // CHECK-NEXT:                %[[L1:.*]] = select
  //
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D2]]]
  // CHECK-NEXT:                {{.*}} = select
  // CHECK-NEXT:                %[[L2:.*]] = select
  //
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D3]]]
  // CHECK-NEXT:                {{.*}} = select
  // CHECK-NEXT:                %[[L3:.*]] = select
  //
  // CHECK-NEXT:                {{.*}} = load %0[%[[L0]], %[[L1]], %[[L2]], %[[L3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                store {{.*}}, %[[ALLOC]][%[[I6]], %[[I5]], %[[I4]]] : memref<5x4x3xf32>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  //      CHECK:          {{.*}} = load %[[VECTOR_VIEW]][{{.*}}] : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          dealloc %[[ALLOC]] : memref<5x4x3xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}

  // Check that I0 + I4 (of size 3) read from first index load(L0, ...) and write into last index store(..., I4)
  // Check that I3 + I6 (of size 5) read from last index load(..., L3) and write into first index store(I6, ...)
  // Other dimensions are just accessed with I1, I2 resp.
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          %f = vector_transfer_read %A, %i0, %i1, %i2, %i3 {permutation_map: (d0, d1, d2, d3) -> (d3, 0, d0)} : (memref<?x?x?x?xf32, 0>, index, index, index, index) -> vector<5x4x3xf32>
        }
      }
    }
  }
  return
}

// CHECK-LABEL:func @materialize_write(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
func @materialize_write(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-NEXT:  %0 = alloc(%arg0, %arg1, %arg2, %arg3) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  %cst = constant splat<vector<5x4x3xf32>, 1.000000e+00> : vector<5x4x3xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %arg0 step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %arg1 step 4 {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %arg2 {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %arg3 step 5 {
  //      CHECK:          %[[D0:.*]] = dim %0, 0 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D1:.*]] = dim %0, 1 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D2:.*]] = dim %0, 2 : memref<?x?x?x?xf32>
  // CHECK-NEXT:          %[[D3:.*]] = dim %0, 3 : memref<?x?x?x?xf32>
  // CHECK:               %[[ALLOC:.*]] = alloc() : memref<5x4x3xf32>
  // CHECK-NEXT:          %[[VECTOR_VIEW:.*]] = vector_type_cast {{.*}} : memref<5x4x3xf32>, memref<1xvector<5x4x3xf32>>
  //      CHECK:          store %cst, {{.*}} : memref<1xvector<5x4x3xf32>>
  // CHECK-NEXT:          affine.for %[[I4:.*]] = 0 to 3 {
  // CHECK-NEXT:            affine.for %[[I5:.*]] = 0 to 4 {
  // CHECK-NEXT:              affine.for %[[I6:.*]] = 0 to 5 {
  // CHECK-NEXT:                %[[C0:.*]] = constant 0 : index
  // CHECK-NEXT:                %[[C1:.*]] = constant 1 : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I0]], %[[I4]])
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D0]]]
  // CHECK-NEXT:                {{.*}} = select {{.*}}, {{.*}}, {{.*}} : index
  // CHECK-NEXT:                %[[S0:.*]] = select {{.*}}, %[[C0]], {{.*}} : index
  //
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I1]], %[[I5]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I1]], %[[I5]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I1]], %[[I5]])
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D1]]]
  // CHECK-NEXT:                {{.*}} = select {{.*}}, {{.*}}, {{.*}} : index
  // CHECK-NEXT:                %[[S1:.*]] = select {{.*}}, %[[C0]], {{.*}} : index
  //
  // CHECK-NEXT:                {{.*}} = cmpi "slt", %[[I2]], %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = cmpi "slt", %[[I2]], %3 : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D2]]]
  // CHECK-NEXT:                {{.*}} = select {{.*}}, %[[I2]], {{.*}} : index
  // CHECK-NEXT:                %[[S2:.*]] = select {{.*}}, %[[C0]], {{.*}} : index
  //
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, %[[C0]] : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = cmpi "slt", {{.*}}, {{.*}} : index
  // CHECK-NEXT:                {{.*}} = affine.apply #[[ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                {{.*}} = affine.apply #[[SUB]]()[%[[D3]]]
  // CHECK-NEXT:                {{.*}} = select {{.*}}, {{.*}}, {{.*}} : index
  // CHECK-NEXT:                %[[S3:.*]] = select {{.*}}, %[[C0]], {{.*}} : index
  //
  // CHECK-NEXT:                {{.*}} = load {{.*}}[%[[I6]], %[[I5]], %[[I4]]] : memref<5x4x3xf32>
  //      CHECK:                store {{.*}}, {{.*}}[%[[S0]], %[[S1]], %[[S2]], %[[S3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          dealloc {{.*}} : memref<5x4x3xf32>
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}
  //
  // Check that I0 + I4 (of size 3) read from last index load(..., I4) and write into first index store(S0, ...)
  // Check that I1 + I5 (of size 4) read from second index load(..., I5, ...) and write into second index store(..., S1, ...)
  // Check that I3 + I6 (of size 5) read from first index load(I6, ...) and write into last index store(..., S3)
  // Other dimension is just accessed with I2.
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant splat<vector<5x4x3xf32>, 1.000000e+00> : vector<5x4x3xf32>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N step 4 {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          vector_transfer_write %f1, %A, %i0, %i1, %i2, %i3 {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d0)} : vector<5x4x3xf32>, memref<?x?x?x?xf32, 0>, index, index, index, index
        }
      }
    }
  }
  return
}
