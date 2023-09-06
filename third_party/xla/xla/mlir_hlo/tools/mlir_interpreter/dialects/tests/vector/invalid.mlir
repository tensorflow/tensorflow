// RUN: (! mlir-interpreter-runner %s -run-all 2>&1) | FileCheck %s

func.func @write_4_at_3_inbounds() {
  %a = memref.alloc() : memref<5xi32>
  %base = arith.constant 3 : index
  %f = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  vector.transfer_write %f, %a[%base]
    {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]}
    : vector<4xi32>, memref<5xi32>
  return
}

// CHECK-LABEL: @write_4_at_3_inbounds
// CHECK-NEXT: index out of bounds

func.func @transfer_read_2d_1d_oob()-> vector<2xi32> {
  %a = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : memref<2x4xi32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c-42 = arith.constant -42: i32
  %f = vector.transfer_read %a[%c2, %c0], %c-42
      : memref<2x4xi32>, vector<2xi32>
  return %f : vector<2xi32>
}

// CHECK-LABEL: @transfer_read_2d_1d_oob
// CHECK-NEXT: index out of bounds
