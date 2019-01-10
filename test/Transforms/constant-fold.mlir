// RUN: mlir-opt %s -constant-fold | FileCheck %s

// CHECK: [[MAP0:#map[0-9]+]] = ()[s0] -> (0, s0)
// CHECK: [[MAP1:#map[0-9]+]] = ()[s0] -> (100, s0)

// CHECK-LABEL: @test(%arg0: memref<f32>) {
func @test(%p : memref<f32>) {
  for %i0 = 0 to 128 {
    for %i1 = 0 to 8 { // CHECK: for %i1 = 0 to 8 {
      %0 = constant 4.5 : f32
      %1 = constant 1.5 : f32

      // CHECK-NEXT: %cst = constant 6.000000e+00 : f32
      %2 = addf %0, %1 : f32

      // CHECK-NEXT: store %cst, %arg0[]
      store %2, %p[] : memref<f32>
    }
  }
  return
}

// CHECK-LABEL: func @simple_addf
func @simple_addf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 6.000000e+00 : f32
  %2 = addf %0, %1 : f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: func @simple_addi
func @simple_addi() -> i32 {
  %0 = constant 1 : i32
  %1 = constant 5 : i32

  // CHECK-NEXT: %c6_i32 = constant 6 : i32
  %2 = addi %0, %1 : i32

  // CHECK-NEXT: return %c6_i32
  return %2 : i32
}

// CHECK-LABEL: func @simple_subf
func @simple_subf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 3.000000e+00 : f32
  %2 = subf %0, %1 : f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: func @simple_subi
func @simple_subi() -> i32 {
  %0 = constant 4 : i32
  %1 = constant 1 : i32

  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  %2 = subi %0, %1 : i32

  // CHECK-NEXT: return %c3_i32
  return %2 : i32
}

// CHECK-LABEL: func @affine_apply
func @affine_apply(%variable : index) -> (index, index, index) {
  %c177 = constant 177 : index
  %c211 = constant 211 : index
  %N = constant 1075 : index

  // CHECK: %c1159 = constant 1159 : index
  // CHECK: %c1152 = constant 1152 : index
  %x = affine_apply (d0, d1)[S0] -> ( (d0 + 128 * S0) floordiv 128 + d1 mod 128, 128 * (S0 ceildiv 128) )
           (%c177, %c211)[%N]

  // CHECK: %c42 = constant 42 : index
  %y = affine_apply (d0) -> (42) (%variable)

  // CHECK: return %c1159, %c1152, %c42
  return %x#0, %x#1, %y : index, index, index
}

// CHECK-LABEL:  func @constant_fold_bounds(%arg0: index) {
func @constant_fold_bounds(%N : index) {
  // CHECK:      %c3 = constant 3 : index
  // CHECK-NEXT: %0 = "foo"() : () -> index
  %c9 = constant 9 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = affine_apply (d0, d1) -> (d0 + d1) (%c1, %c2)
  %l = "foo"() : () -> index

  // CHECK:  for %i0 = 5 to 7 {
  for %i = max (d0, d1) -> (0, d0 + d1)(%c2, %c3) to min (d0, d1) -> (d0 - 2, 32*d1) (%c9, %c1) {
    "foo"(%i, %c3) : (index, index) -> ()
  }

  // Bound takes a non-constant argument but can still be folded.
  // CHECK:  for %i1 = 1 to 7 {
  for %j = max (d0) -> (0, 1)(%N) to min (d0, d1) -> (7, 9)(%N, %l) {
    "foo"(%j, %c3) : (index, index) -> ()
  }

  // None of the bounds can be folded.
  // CHECK: for %i2 = max [[MAP0]]()[%0] to min [[MAP1]]()[%arg0] {
  for %k = max ()[s0] -> (0, s0) ()[%l] to min ()[s0] -> (100, s0)()[%N] {
    "foo"(%k, %c3) : (index, index) -> ()
  }
  return
}


// CHECK-LABEL: func @simple_mulf
func @simple_mulf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 6.750000e+00 : f32
  %2 = mulf %0, %1 : f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: func @simple_divis
func @simple_divis() -> (i32, i32) {
  %0 = constant 6 : i32
  %1 = constant 2 : i32

  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  %2 = divis %0, %1 : i32

  %3 = constant -2 : i32

  // CHECK-NEXT: %c-3_i32 = constant -3 : i32
  %4 = divis %0, %3 : i32

  // CHECK-NEXT: return %c3_i32, %c-3_i32 : i32, i32
  return %2, %4 : i32, i32
}

// CHECK-LABEL: func @simple_diviu
func @simple_diviu() -> (i32, i32) {
  %0 = constant 6 : i32
  %1 = constant 2 : i32

  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  %2 = diviu %0, %1 : i32

  %3 = constant -2 : i32

  // Unsigned division interprets -2 as 2^32-2, so the result is 0.
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %4 = diviu %0, %3 : i32

  // CHECK-NEXT: return %c3_i32, %c0_i32 : i32, i32
  return %2, %4 : i32, i32
}

// CHECK-LABEL: func @simple_remis
func @simple_remis(%a : i32) -> (i32, i32, i32) {
  %0 = constant 5 : i32
  %1 = constant 2 : i32
  %2 = constant 1 : i32
  %3 = constant -2 : i32

  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  %4 = remis %0, %1 : i32
  // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
  %5 = remis %0, %3 : i32
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %6 = remis %a, %2 : i32

  // CHECK-NEXT: return %c1_i32, %c1_i32_0, %c0_i32 : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// CHECK-LABEL: func @simple_remiu
func @simple_remiu(%a : i32) -> (i32, i32, i32) {
  %0 = constant 5 : i32
  %1 = constant 2 : i32
  %2 = constant 1 : i32
  %3 = constant -2 : i32

  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  %4 = remiu %0, %1 : i32
  // CHECK-NEXT: %c5_i32 = constant 5 : i32
  %5 = remiu %0, %3 : i32
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %6 = remiu %a, %2 : i32

  // CHECK-NEXT: return %c1_i32, %c5_i32, %c0_i32 : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// CHECK-LABEL: func @muli(%arg0: i32)
func @muli(i32) -> (i32, i32) {
^bb0(%a : i32):
  %0 = constant 4 : i32
  %1 = constant 2 : i32
  %zero = constant 0 : i32

  // CHECK-NEXT: %c8_i32 = constant 8 : i32
  %2 = muli %0, %1 : i32

  // x*0 = 0
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  %3 = muli %a, %zero : i32

  // CHECK-NEXT: return %c8_i32, %c0_i32
  return %2, %3 : i32, i32
}

// CHECK-LABEL: func @dim
func @dim(%x : tensor<8x4xf32>) -> index {

  // CHECK: %c4 = constant 4 : index
  %0 = dim %x, 1 : tensor<8x4xf32>

  // CHECK-NEXT: return %c4
  return %0 : index
}

// CHECK-LABEL: func @cmpi
func @cmpi() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = constant 42 : i32
  %cm1 = constant -1 : i32
// CHECK-NEXT: %false = constant 0 : i1
  %0 = cmpi "eq", %c42, %cm1 : i32
// CHECK-NEXT: %true = constant 1 : i1
  %1 = cmpi "ne", %c42, %cm1 : i32
// CHECK-NEXT: %false_0 = constant 0 : i1
  %2 = cmpi "slt", %c42, %cm1 : i32
// CHECK-NEXT: %false_1 = constant 0 : i1
  %3 = cmpi "sle", %c42, %cm1 : i32
// CHECK-NEXT: %true_2 = constant 1 : i1
  %4 = cmpi "sgt", %c42, %cm1 : i32
// CHECK-NEXT: %true_3 = constant 1 : i1
  %5 = cmpi "sge", %c42, %cm1 : i32
// CHECK-NEXT: %true_4 = constant 1 : i1
  %6 = cmpi "ult", %c42, %cm1 : i32
// CHECK-NEXT: %true_5 = constant 1 : i1
  %7 = cmpi "ule", %c42, %cm1 : i32
// CHECK-NEXT: %false_6 = constant 0 : i1
  %8 = cmpi "ugt", %c42, %cm1 : i32
// CHECK-NEXT: %false_7 = constant 0 : i1
  %9 = cmpi "uge", %c42, %cm1 : i32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine_apply (i) -> (i mod 42) to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
// --------------------------------------------------------------------------//
// CHECK-LABEL: @lowered_affine_mod
func @lowered_affine_mod() -> (index, index) {
// CHECK-NEXT: {{.*}} = constant 41 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %0 = remis %c-43, %c42 : index
  %c0 = constant 0 : index
  %1 = cmpi "slt", %0, %c0 : index
  %2 = addi %0, %c42 : index
  %3 = select %1, %2, %0 : index
// CHEKC-NEXT: {{.*}} = constant 1 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %4 = remis %c43, %c42_0 : index
  %c0_1 = constant 0 : index
  %5 = cmpi "slt", %4, %c0_1 : index
  %6 = addi %4, %c42_0 : index
  %7 = select %5, %6, %4 : index
  return %3, %7 : index, index
}

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine_apply (i) -> (i mod 42) to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @lowered_affine_floordiv
func @lowered_affine_floordiv() -> (index, index) {
// CHECK-NEXT: %c-2 = constant -2 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %c0 = constant 0 : index
  %c-1 = constant -1 : index
  %0 = cmpi "slt", %c-43, %c0 : index
  %1 = subi %c-1, %c-43 : index
  %2 = select %0, %1, %c-43 : index
  %3 = divis %2, %c42 : index
  %4 = subi %c-1, %3 : index
  %5 = select %0, %4, %3 : index
// CHECK-NEXT: %c1 = constant 1 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %c0_1 = constant 0 : index
  %c-1_2 = constant -1 : index
  %6 = cmpi "slt", %c43, %c0_1 : index
  %7 = subi %c-1_2, %c43 : index
  %8 = select %6, %7, %c43 : index
  %9 = divis %8, %c42_0 : index
  %10 = subi %c-1_2, %9 : index
  %11 = select %6, %10, %9 : index
  return %5, %11 : index, index
}

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: the operations in this test are exactly those produced by
// lowering affine_apply (i) -> (i mod 42) to standard operations.  Please only
// change these operations together with the affine lowering pass tests.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @lowered_affine_ceildiv
func @lowered_affine_ceildiv() -> (index, index) {
// CHECK-NEXT:  %c-1 = constant -1 : index
  %c-43 = constant -43 : index
  %c42 = constant 42 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = cmpi "sle", %c-43, %c0 : index
  %1 = subi %c0, %c-43 : index
  %2 = subi %c-43, %c1 : index
  %3 = select %0, %1, %2 : index
  %4 = divis %3, %c42 : index
  %5 = subi %c0, %4 : index
  %6 = addi %4, %c1 : index
  %7 = select %0, %5, %6 : index
// CHECK-NEXT:  %c2 = constant 2 : index
  %c43 = constant 43 : index
  %c42_0 = constant 42 : index
  %c0_1 = constant 0 : index
  %c1_2 = constant 1 : index
  %8 = cmpi "sle", %c43, %c0_1 : index
  %9 = subi %c0_1, %c43 : index
  %10 = subi %c43, %c1_2 : index
  %11 = select %8, %9, %10 : index
  %12 = divis %11, %c42_0 : index
  %13 = subi %c0_1, %12 : index
  %14 = addi %12, %c1_2 : index
  %15 = select %8, %13, %14 : index
  return %7, %15 : index, index
}


