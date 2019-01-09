// RUN: mlir-opt %s -vectorize -virtual-vector-size 3 -virtual-vector-size 32 --test-fastest-varying=1 --test-fastest-varying=0 -materialize-vectors -vector-size=3 -vector-size=16 | FileCheck %s 

// vector<3x32xf32> -> vector<3x16xf32>
// CHECK-DAG: [[D0D1TOD0:#.*]] = (d0, d1) -> (d0)
// CHECK-DAG: [[D0D1TOD1:#.*]] = (d0, d1) -> (d1)
// CHECK-DAG: [[D0D1TOD0D1:#.*]] = (d0, d1) -> (d0, d1)
// CHECK-DAG: [[D0D1TOD1P16:#.*]] = (d0, d1) -> (d1 + 16)

// CHECK-LABEL: func @vector_add_2d
func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = alloc (%M, %N) : memref<?x?xf32, 0>
  %B = alloc (%M, %N) : memref<?x?xf32, 0>
  %C = alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  // 2x unroll (jammed by construction).
  // CHECK: for %i0 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i1 = 0 to %arg1 step 32 {
  // CHECK-NEXT:     {{.*}} = constant splat<vector<3x16xf32>, 1.000000e+00> : vector<3x16xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<3x16xf32>, 1.000000e+00> : vector<3x16xf32>
  // CHECK-NEXT:     [[VAL00:%.*]] = affine_apply [[D0D1TOD0]](%i0, %i1)
  // CHECK-NEXT:     [[VAL01:%.*]] = affine_apply [[D0D1TOD1]](%i0, %i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL00]], [[VAL01]] {permutation_map: [[D0D1TOD0D1]]} : vector<3x16xf32>
  // CHECK-NEXT:     [[VAL10:%.*]] = affine_apply [[D0D1TOD0]](%i0, %i1)
  // CHECK-NEXT:     [[VAL11:%.*]] = affine_apply [[D0D1TOD1P16]](%i0, %i1)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL10]], [[VAL11]] {permutation_map: [[D0D1TOD0D1]]} : vector<3x16xf32>
  //
  for %i0 = 0 to %M {
    for %i1 = 0 to %N {
      // non-scoped %f1
      store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  // 2x unroll (jammed by construction).
  // CHECK: for %i2 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i3 = 0 to %arg1 step 32 {
  // CHECK-NEXT:     {{.*}} = constant splat<vector<3x16xf32>, 2.000000e+00> : vector<3x16xf32>
  // CHECK-NEXT:     {{.*}} = constant splat<vector<3x16xf32>, 2.000000e+00> : vector<3x16xf32>
  // CHECK-NEXT:     [[VAL00:%.*]] = affine_apply [[D0D1TOD0]](%i2, %i3)
  // CHECK-NEXT:     [[VAL01:%.*]] = affine_apply [[D0D1TOD1]](%i2, %i3)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL00]], [[VAL01]] {permutation_map: [[D0D1TOD0D1]]} : vector<3x16xf32>
  // CHECK-NEXT:     [[VAL10:%.*]] = affine_apply [[D0D1TOD0]](%i2, %i3)
  // CHECK-NEXT:     [[VAL11:%.*]] = affine_apply [[D0D1TOD1P16]](%i2, %i3)
  // CHECK-NEXT:     vector_transfer_write {{.*}}, {{.*}}, [[VAL10]], [[VAL11]] {permutation_map: [[D0D1TOD0D1]]} : vector<3x16xf32>
  //
  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      // non-scoped %f2
      store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  // 2x unroll (jammed by construction).
  // CHECK: for %i4 = 0 to %arg0 step 3 {
  // CHECK-NEXT:   for %i5 = 0 to %arg1 step 32 {
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = vector_transfer_read
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<3x16xf32>
  // CHECK-NEXT:     {{.*}} = addf {{.*}} : vector<3x16xf32>
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     vector_transfer_write
  // CHECK-NEXT:     {{.*}} = affine_apply
  // CHECK-NEXT:     {{.*}} = affine_apply
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
