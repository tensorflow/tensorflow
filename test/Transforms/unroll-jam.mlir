// RUN: mlir-opt %s -loop-unroll-jam -unroll-jam-factor=2 | FileCheck %s

// CHECK: [[MAP_PLUS_1:#map[0-9]+]] = (d0) -> (d0 + 1)
// This should be matched to M1, but M1 is defined later.
// CHECK: {{#map[0-9]+}} = ()[s0] -> (s0 + 8)

// CHECK-LABEL: func @unroll_jam_imperfect_nest() {
func @unroll_jam_imperfect_nest() {
  // CHECK: %c100 = constant 100 : index
  // CHECK-NEXT: affine.for %i0 = 0 to 99 step 2 {
  affine.for %i = 0 to 101 {
    // CHECK: %0 = "addi32"(%i0, %i0) : (index, index) -> i32
    // CHECK-NEXT: %1 = affine.apply [[MAP_PLUS_1]](%i0)
    // CHECK-NEXT: %2 = "addi32"(%1, %1) : (index, index) -> i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    affine.for %j = 0 to 17 {
      // CHECK: %3 = "addi32"(%i0, %i0) : (index, index) -> i32
      // CHECK-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
      // CHECK-NEXT: %5 = affine.apply [[MAP_PLUS_1]](%i0)
      // CHECK-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> i32
      // CHECK-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
      %y = "addi32"(%i, %i) : (index, index) -> i32
      %z = "addi32"(%y, %y) : (i32, i32) -> i32
    }
    // CHECK: %8 = "addi32"(%i0, %i0) : (index, index) -> i32
    // CHECK-NEXT: %9 = affine.apply [[MAP_PLUS_1]](%i0)
    // CHECK-NEXT: %10 = "addi32"(%9, %9) : (index, index) -> i32
    %w = "addi32"(%i, %i) : (index, index) -> i32
  } // CHECK }
  // cleanup loop (single iteration)
  // CHECK: %11 = "addi32"(%c100, %c100) : (index, index) -> i32
  // CHECK-NEXT: affine.for %i2 = 0 to 17 {
    // CHECK-NEXT: %12 = "addi32"(%c100, %c100) : (index, index) -> i32
    // CHECK-NEXT: %13 = "addi32"(%12, %12) : (i32, i32) -> i32
  // CHECK-NEXT: }
  // CHECK-NEXT: %14 = "addi32"(%c100, %c100) : (index, index) -> i32
  return
}

// UNROLL-BY-4-LABEL: func @loop_nest_unknown_count_1(%arg0: index) {
func @loop_nest_unknown_count_1(%N : index) {
  // UNROLL-BY-4-NEXT: affine.for %i0 = 1 to  #map{{[0-9]+}}()[%arg0] step 4 {
    // UNROLL-BY-4-NEXT: affine.for %i1 = 1 to 100 {
      // UNROLL-BY-4-NEXT: %0 = "foo"() : () -> i32
      // UNROLL-BY-4-NEXT: %1 = "foo"() : () -> i32
      // UNROLL-BY-4-NEXT: %2 = "foo"() : () -> i32
      // UNROLL-BY-4-NEXT: %3 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: }
  // UNROLL-BY-4-NEXT: }
  // A cleanup loop should be generated here.
  // UNROLL-BY-4-NEXT: affine.for %i2 = #map{{[0-9]+}}()[%arg0] to %arg0 {
    // UNROLL-BY-4-NEXT: affine.for %i3 = 1 to 100 {
      // UNROLL-BY-4-NEXT: %4 = "foo"() : () -> i32
    // UNROLL-BY-4_NEXT: }
  // UNROLL-BY-4_NEXT: }
  // Specify the lower bound in a form so that both lb and ub operands match.
  affine.for %i = ()[s0] -> (1)()[%N] to %N {
    affine.for %j = 1 to 100 {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// UNROLL-BY-4-LABEL: func @loop_nest_unknown_count_2(%arg0: index) {
func @loop_nest_unknown_count_2(%arg : index) {
  // UNROLL-BY-4-NEXT: affine.for %i0 = %arg0 to  #map{{[0-9]+}}()[%arg0] step 4 {
    // UNROLL-BY-4-NEXT: affine.for %i1 = 1 to 100 {
      // UNROLL-BY-4-NEXT: %0 = "foo"(%i0) : (index) -> i32
      // UNROLL-BY-4-NEXT: %1 = affine.apply #map{{[0-9]+}}(%i0)
      // UNROLL-BY-4-NEXT: %2 = "foo"(%1) : (index) -> i32
      // UNROLL-BY-4-NEXT: %3 = affine.apply #map{{[0-9]+}}(%i0)
      // UNROLL-BY-4-NEXT: %4 = "foo"(%3) : (index) -> i32
      // UNROLL-BY-4-NEXT: %5 = affine.apply #map{{[0-9]+}}(%i0)
      // UNROLL-BY-4-NEXT: %6 = "foo"(%5) : (index) -> i32
    // UNROLL-BY-4-NEXT: }
  // UNROLL-BY-4-NEXT: }
  // The cleanup loop is a single iteration one and is promoted.
  // UNROLL-BY-4-NEXT: %7 = affine.apply [[M1:#map{{[0-9]+}}]]()[%arg0]
  // UNROLL-BY-4-NEXT: affine.for %i3 = 1 to 100 {
    // UNROLL-BY-4-NEXT: %8 = "foo"() : () -> i32
  // UNROLL-BY-4_NEXT: }
  // Specify the lower bound in a form so that both lb and ub operands match.
  affine.for %i = ()[s0] -> (s0) ()[%arg] to ()[s0] -> (s0+8) ()[%arg] {
    affine.for %j = 1 to 100 {
      %x = "foo"(%i) : (index) -> i32
    }
  }
  return
}
