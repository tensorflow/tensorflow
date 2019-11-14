// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @vector_transfer_ops(
func @vector_transfer_ops(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  %f0 = constant 0.0 : f32
  //
  // CHECK: %0 = vector.transfer_read
  %0 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = (d0, d1)->(d0)} : memref<?x?xf32>, vector<128xf32>
  // CHECK: %1 = vector.transfer_read
  %1 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = (d0, d1)->(d1, d0)} : memref<?x?xf32>, vector<3x7xf32>
  // CHECK: vector.transfer_read
  %2 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0)} : memref<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read
  %3 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d1)} : memref<?x?xf32>,  vector<128xf32>
  //
  // CHECK: vector.transfer_write
  vector.transfer_write %0, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0)} : vector<128xf32>, memref<?x?xf32>
  // CHECK: vector.transfer_write
  vector.transfer_write %1, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d1, d0)} : vector<3x7xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: extractelement
func @extractelement(%arg0: vector<4x8x16xf32>) -> (vector<8x16xf32>, vector<16xf32>, f32) {
  //      CHECK: vector.extractelement {{.*}}[3 : i32] : vector<4x8x16xf32>
  %1 = vector.extractelement %arg0[3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32] : vector<4x8x16xf32>
  %2 = vector.extractelement %arg0[3 : i32, 3 : i32] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extractelement {{.*}}[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  %3 = vector.extractelement %arg0[3 : i32, 3 : i32, 3 : i32] : vector<4x8x16xf32>
  return %1, %2, %3 : vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: outerproduct
func @outerproduct(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  //     CHECK: vector.outerproduct {{.*}} : vector<4xf32>, vector<8xf32>
  %0 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  //     CHECK: vector.outerproduct {{.*}}, {{.*}}, {{.*}} : vector<4xf32>, vector<8xf32>
  %1 = vector.outerproduct %arg0, %arg1, %arg2 : vector<4xf32>, vector<8xf32>
  return %1 : vector<4x8xf32>
}
