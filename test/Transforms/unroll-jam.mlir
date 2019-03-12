// RUN: mlir-opt %s -loop-unroll-jam -unroll-jam-factor=2 | FileCheck %s

// CHECK-DAG: [[MAP_PLUS_1:#map[0-9]+]] = (d0) -> (d0 + 1)
// CHECK-DAG: [[M1:#map[0-9]+]] = ()[s0] -> (s0 + 8)
// CHECK-DAG: [[MAP_DIV_OFFSET:#map[0-9]+]] = ()[s0] -> (((s0 - 1) floordiv 2) * 2 + 1)
// CHECK-DAG: [[MAP_MULTI_RES:#map[0-9]+]] = ()[s0, s1] -> ((s0 floordiv 2) * 2, (s1 floordiv 2) * 2, 1024)

// CHECK-LABEL: func @unroll_jam_imperfect_nest() {
func @unroll_jam_imperfect_nest() {
  // CHECK: %c100 = constant 100 : index
  // CHECK-NEXT: for %i0 = 0 to 100 step 2 {
  for %i = 0 to 101 {
    // CHECK: %0 = "addi32"(%i0, %i0) : (index, index) -> i32
    // CHECK-NEXT: %1 = affine.apply [[MAP_PLUS_1]](%i0)
    // CHECK-NEXT: %2 = "addi32"(%1, %1) : (index, index) -> i32
    %x = "addi32"(%i, %i) : (index, index) -> i32
    for %j = 0 to 17 {
      // CHECK:      %3 = "addi32"(%i0, %i0) : (index, index) -> i32
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
  // CHECK-NEXT: for %i2 = 0 to 17 {
  // CHECK-NEXT:   %12 = "addi32"(%c100, %c100) : (index, index) -> i32
  // CHECK-NEXT:   %13 = "addi32"(%12, %12) : (i32, i32) -> i32
  // CHECK-NEXT: }
  // CHECK-NEXT: %14 = "addi32"(%c100, %c100) : (index, index) -> i32
  return
}

// CHECK-LABEL: func @loop_nest_unknown_count_1(%arg0: index) {
func @loop_nest_unknown_count_1(%N : index) {
  // CHECK-NEXT: for %i0 = 1 to [[MAP_DIV_OFFSET]]()[%arg0] step 2 {
  // CHECK-NEXT:   for %i1 = 1 to 100 {
  // CHECK-NEXT:     %0 = "foo"() : () -> i32
  // CHECK-NEXT:     %1 = "foo"() : () -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // A cleanup loop should be generated here.
  // CHECK-NEXT: for %i2 = [[MAP_DIV_OFFSET]]()[%arg0] to %arg0 {
  // CHECK-NEXT:   for %i3 = 1 to 100 {
  // CHECK-NEXT:     %2 = "foo"() : () -> i32
  // CHECK_NEXT:   }
  // CHECK_NEXT: }
  for %i = 1 to %N {
    for %j = 1 to 100 {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// CHECK-LABEL: func @loop_nest_unknown_count_2(%arg0: index) {
func @loop_nest_unknown_count_2(%arg : index) {
  // CHECK-NEXT: for %i0 = %arg0 to  [[M1]]()[%arg0] step 2 {
  // CHECK-NEXT:   for %i1 = 1 to 100 {
  // CHECK-NEXT:     %0 = "foo"(%i0) : (index) -> i32
  // CHECK-NEXT:     %1 = affine.apply #map{{[0-9]+}}(%i0)
  // CHECK-NEXT:     %2 = "foo"(%1) : (index) -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // The cleanup loop is a single iteration one and is promoted.
  // CHECK-NEXT: %3 = affine.apply [[M1]]()[%arg0]
  // CHECK-NEXT: for %i2 = 1 to 100 {
  // CHECK-NEXT:   %4 = "foo"(%3) : (index) -> i32
  // CHECK_NEXT: }
  for %i = %arg to ()[s0] -> (s0+9) ()[%arg] {
    for %j = 1 to 100 {
      %x = "foo"(%i) : (index) -> i32
    }
  }
  return
}

// CHECK-LABEL: func @loop_nest_symbolic_and_min_upper_bound
func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  for %i = 0 to min ()[s0, s1] -> (s0, s1, 1024)()[%M, %N] {
    for %j = 0 to %K {
      "foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
// CHECK-NEXT:  for %i0 = 0 to min [[MAP_MULTI_RES]]()[%arg0, %arg1] step 2 {
// CHECK-NEXT:    for %i1 = 0 to %arg2 {
// CHECK-NEXT:      "foo"(%i0, %i1) : (index, index) -> ()
// CHECK-NEXT:      %0 = affine.apply #map2(%i0)
// CHECK-NEXT:      "foo"(%0, %i1) : (index, index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  for %i2 = max [[MAP_MULTI_RES]]()[%arg0, %arg1] to min #map9()[%arg0, %arg1] {
// CHECK-NEXT:    for %i3 = 0 to %arg2 {
// CHECK-NEXT:      "foo"(%i2, %i3) : (index, index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
