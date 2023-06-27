// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @write_4_at_3_inbounds() -> memref<?xi32> {
  %c8 = arith.constant 8 : index
  %a = memref.alloc(%c8) : memref<?xi32>
  %base = arith.constant 3 : index
  %f = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  vector.transfer_write %f, %a[%base]
    {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]}
    : vector<4xi32>, memref<?xi32>
  return %a : memref<?xi32>
}

// CHECK-LABEL: @write_4_at_3_inbounds
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 0, 0, 1, 2, 3, 4, 0]

func.func @write_6_at_5() -> memref<8xi32> {
  %a = memref.alloc() : memref<8xi32>
  %base = arith.constant 5 : index
  %f = arith.constant dense<[1, 2, 3, 4, 5, 6]> : vector<6xi32>
  vector.transfer_write %f, %a[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<6xi32>, memref<8xi32>
  return %a : memref<8xi32>
}

// CHECK-LABEL: @write_6_at_5
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 0, 0, 0, 0, 1, 2, 3]

func.func @write_to_tensor() -> (tensor<3xi32>, tensor<3xi32>) {
  %a = arith.constant dense<[1,2,3]> : tensor<3xi32>
  %base = arith.constant 1 : index
  %f = arith.constant dense<[4, 5]> : vector<2xi32>
  %b = vector.transfer_write %f, %a[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<2xi32>, tensor<3xi32>
  return %a, %b : tensor<3xi32>, tensor<3xi32>
}

// CHECK-LABEL: @write_to_tensor
// CHECK-NEXT: Results
// CHECK-NEXT: [1, 2, 3]
// CHECK-NEXT: [1, 4, 5]

func.func @write_masked() -> memref<4xi32> {
  %a = arith.constant dense<[10, 11, 12, 13]> : memref<4xi32>
  %base = arith.constant 1 : index
  %f = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %mask = arith.constant dense<[true, false, true]> : vector<3xi1>
  vector.transfer_write %f, %a[%base], %mask
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<3xi32>, memref<4xi32>
  return %a : memref<4xi32>
}

// CHECK-LABEL: @write_masked
// CHECK-NEXT: Results
// CHECK-NEXT: [10, 1, 12, 3]

func.func @write_vector_mask() -> memref<4xi32> {
  %a = arith.constant dense<[10, 11, 12, 13]> : memref<4xi32>
  %base = arith.constant 1 : index
  %f = arith.constant dense<[1, 2, 3]> : vector<3xi32>
  %mask = arith.constant dense<[true, false, true]> : vector<3xi1>
  vector.mask %mask {
    vector.transfer_write %f, %a[%base]
      {permutation_map = affine_map<(d0) -> (d0)>}
      : vector<3xi32>, memref<4xi32>
  } : vector<3xi1>
  return %a : memref<4xi32>
}

// CHECK-LABEL: @write_vector_mask
// CHECK-NEXT: Results
// CHECK-NEXT: [10, 1, 12, 3]

func.func @write_1d_to_2d() -> memref<2x4xi32> {
  %a = memref.alloc() : memref<2x4xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %f = arith.constant dense<[1, 2]> : vector<2xi32>
  vector.transfer_write %f, %a[%c1, %c2]
    : vector<2xi32>, memref<2x4xi32>
  return %a : memref<2x4xi32>
}

// CHECK-LABEL: @write_1d_to_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[0, 0, 0, 0], [0, 0, 1, 2]]