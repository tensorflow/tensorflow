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
// CHECK-NEXT: bounds check failed
