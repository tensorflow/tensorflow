// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 16 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=8 | FileCheck %s

// vector<3x16xf32> -> vector<8xf32>
// CHECK-DAG: [[ID1:#.*]] = (d0) -> (d0)
// CHECK-DAG: [[D0D1TOD1:#.*]] = (d0, d1) -> (d1)
// CHECK-DAG: [[D0P8:#.*]] = (d0) -> (d0 + 8)
// CHECK-DAG: [[D0P1:#.*]] = (d0) -> (d0 + 1)
// CHECK-DAG: [[D0P2:#.*]] = (d0) -> (d0 + 2)

// CHECK-LABEL: func @vector_add_2d
func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i1 = 0 to %arg1 step 16 {
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<8xf32>, 1.000000e+00> : vector<8xf32>
  // CHECK-NEXT:     [[VAL00:%.*]] = affine.apply [[ID1]](%i0)
  // CHECK-NEXT:     [[VAL01:%.*]] = affine.apply [[ID1]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL00]], [[VAL01]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  // CHECK-NEXT:     [[VAL10:%.*]] = affine.apply [[ID1]](%i0)
  // CHECK-NEXT:     [[VAL11:%.*]] = affine.apply [[D0P8]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL10]], [[VAL11]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  // CHECK-NEXT:     [[VAL20:%.*]] = affine.apply [[D0P1]](%i0)
  // CHECK-NEXT:     [[VAL21:%.*]] = affine.apply [[ID1]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL20]], [[VAL21]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  // CHECK-NEXT:     [[VAL30:%.*]] = affine.apply [[D0P1]](%i0)
  // CHECK-NEXT:     [[VAL31:%.*]] = affine.apply [[D0P8]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL30]], [[VAL31]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  // CHECK-NEXT:     [[VAL40:%.*]] = affine.apply [[D0P2]](%i0)
  // CHECK-NEXT:     [[VAL41:%.*]] = affine.apply [[ID1]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL40]], [[VAL41]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  // CHECK-NEXT:     [[VAL50:%.*]] = affine.apply [[D0P2]](%i0)
  // CHECK-NEXT:     [[VAL51:%.*]] = affine.apply [[D0P8]](%i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL50]], [[VAL51]] {permutation_map: [[D0D1TOD1]]} : vector<8xf32>
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i2 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i3 = 0 to %arg1 step 16 {
  // .....
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      // CHECK does (3x4)x unrolling.
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  // (3x2)x unroll (jammed by construction).
  // CHECK: for %i4 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i5 = 0 to %arg1 step 16 {
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<8xf32>
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     {{.*}} = affine.apply
  // CHECK-NEXT:     vector_transfer_write
  //
  for %i4 = 0 to %M {
    for %i5 = 0 to %N {
      %a5 = load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = addf %a5, %b5 : f32
      store %s5, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = constant 7 : index
  %c42 = constant 42 : index
  %res = load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}
