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

