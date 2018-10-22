// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: mlfunc @test_subi_zero
mlfunc @test_subi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = subi %arg0, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: cfgfunc @test_subi_zero_cfg
cfgfunc @test_subi_zero_cfg(i32) -> i32 {
// CHECK-NEXT: bb0(%arg0: i32):
bb0(%arg0: i32):
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0
  %y = subi %arg0, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: mlfunc @dim
mlfunc @dim(%arg0 : tensor<8x4xf32>) -> index {

  // CHECK: %c4 = constant 4 : index
  %0 = dim %arg0, 1 : tensor<8x4xf32>

  // CHECK-NEXT: return %c4
  return %0 : index
}

// CHECK-LABEL: mlfunc @test_associative
mlfunc @test_associative(%arg0: i32) -> i32 {
  // CHECK: %c42_i32 = constant 42 : i32
  // CHECK-NEXT: %0 = addi %arg0, %c42_i32 : i32
  // CHECK-NEXT: return %0

  %c42_i32 = constant 42 : i32
  %y = addi %c42_i32, %arg0 : i32
  return %y: i32
}

// CHECK-LABEL: mlfunc @trivial_dce
mlfunc @trivial_dce(%arg0: tensor<8x4xf32>) {
  %0 = dim %arg0, 1 : tensor<8x4xf32>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: mlfunc @addi_zero
mlfunc @addi_zero(%arg0: i32) -> i32 {
  // CHECK-NEXT: return %arg0
  %c0_i32 = constant 0 : i32
  %y = addi %c0_i32, %arg0 : i32
  return %y: i32
}
