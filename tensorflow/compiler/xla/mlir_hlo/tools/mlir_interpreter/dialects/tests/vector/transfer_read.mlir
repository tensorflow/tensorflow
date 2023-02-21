// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

// Adapted from mlir/test/Integration/Dialect/Vector/CPU/test-transfer-read.mlir.

func.func @transfer_read_1d() -> vector<13xi32> {
  %a = arith.constant dense<[0, 1, 2, 3, 4]> : memref<5xi32>
  %c2 = arith.constant 2 : index
  %c-42 = arith.constant -42 : i32
  %f = vector.transfer_read %a[%c2], %c-42
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<5xi32>, vector<13xi32>
  return %f : vector<13xi32>
}

// CHECK-LABEL: @transfer_read_1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<13xi32>: [2, 3, 4, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42]

func.func @transfer_read_mask_1d() -> vector<13xi32> {
  %a = arith.constant dense<[0, 1, 2, 3, 4]> : memref<5xi32>
  %c2 = arith.constant 2 : index
  %c-42 = arith.constant -42: i32
  %m = arith.constant dense<[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]> : vector<13xi1>
  %f = vector.transfer_read %a[%c2], %c-42, %m : memref<5xi32>, vector<13xi32>
  return %f : vector<13xi32>
}

// CHECK-LABEL: @transfer_read_mask_1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<13xi32>: [-42, -42, 4, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42]

func.func @transfer_read_vector_mask() -> vector<6xi32> {
  %a = arith.constant dense<[0, 1, 2, 3, 4]> : memref<5xi32>
  %c2 = arith.constant 2 : index
  %c-42 = arith.constant -42: i32
  %m = arith.constant dense<[0, 0, 1, 1, 1, 1]> : vector<6xi1>
  %passthrough = arith.constant dense<[-1, -2, -3, -4, -5, -6]> : vector<6xi32>
  %f = vector.mask %m, %passthrough {
    vector.transfer_read %a[%c2], %c-42 : memref<5xi32>, vector<6xi32>
  } : vector<6xi1> -> vector<6xi32>
  return %f : vector<6xi32>
}

// CHECK-LABEL: @transfer_read_vector_mask
// CHECK-NEXT: Results
// CHECK-NEXT: vector<6xi32>: [-1, -2, 4, -42, -42, -42]

func.func @transfer_read_inbounds_4() -> vector<4xi32> {
  %a = arith.constant dense<[0, 1, 2, 0, 0]> : memref<5xi32>
  %c-42 = arith.constant -42: i32
  %c1 = arith.constant 1 : index
  %f = vector.transfer_read %a[%c1], %c-42
      {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]} :
    memref<5xi32>, vector<4xi32>
  return %f : vector<4xi32>
}

// CHECK-LABEL: @transfer_read_inbounds_4
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi32>: [1, 2, 0, 0]

func.func @transfer_read_mask_inbounds_4() -> vector<4xi32> {
  %a = arith.constant dense<[0, 1, 2, 0, 0]> : memref<5xi32>
  %c-42 = arith.constant -42: i32
  %c1 = arith.constant 1 : index
  %m = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  %f = vector.transfer_read %a[%c1], %c-42, %m {in_bounds = [true]}
      : memref<5xi32>, vector<4xi32>
  return %f : vector<4xi32>
}

// CHECK-LABEL: @transfer_read_mask_inbounds_4
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi32>: [-42, 2, -42, 0]

func.func @transfer_read_2d()-> vector<2x2xi32> {
  %a = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]>
    : memref<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c-42 = arith.constant -42: i32
  %f = vector.transfer_read %a[%c1, %c0], %c-42
      : memref<3x4xi32>, vector<2x2xi32>
  return %f : vector<2x2xi32>
}

// CHECK-LABEL: @transfer_read_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[4, 5], [8, 9]]

func.func @transfer_read_2d_1d()-> vector<2xi32> {
  %a = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : memref<2x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c-42 = arith.constant -42: i32
  %f = vector.transfer_read %a[%c1, %c0], %c-42
      : memref<2x4xi32>, vector<2xi32>
  return %f : vector<2xi32>
}

// CHECK-LABEL: @transfer_read_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2xi32>: [4, 5]
