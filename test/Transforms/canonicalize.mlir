// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @test_subi_zero
mlfunc @test_subi_zero(%x: i32) -> i32 {
  // CHECK: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = subi %x, %x : i32
  return %y: i32
}

