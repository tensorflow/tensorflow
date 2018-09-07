// RUN: mlir-opt %s -o - -loop-unroll-jam -unroll-jam-factor=2 | FileCheck %s

// CHECK: #map0 = (d0) -> (d0 + 1)

// CHECK-LABEL: mlfunc @unroll_jam_imperfect_nest() {
mlfunc @unroll_jam_imperfect_nest() {
  // CHECK: %c100 = constant 100 : affineint
  // CHECK-NEXT: for %i0 = 0 to 99 step 2 {
  for %i = 0 to 100 {
    // CHECK: %0 = "addi32"(%i0, %i0) : (affineint, affineint) -> i32
    // CHECK-NEXT: %1 = affine_apply #map0(%i0)
    // CHECK-NEXT: %2 = "addi32"(%1, %1) : (affineint, affineint) -> i32
    %x = "addi32"(%i, %i) : (affineint, affineint) -> i32
    for %j = 0 to 17 {
      // CHECK: %3 = "addi32"(%i0, %i0) : (affineint, affineint) -> i32
      // CHECK-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
      // CHECK-NEXT: %5 = affine_apply #map0(%i0)
      // CHECK-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
      // CHECK-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
      %y = "addi32"(%i, %i) : (affineint, affineint) -> i32
      %z = "addi32"(%y, %y) : (i32, i32) -> i32
    }
    // CHECK: %8 = "addi32"(%i0, %i0) : (affineint, affineint) -> i32
    // CHECK-NEXT: %9 = affine_apply #map0(%i0)
    // CHECK-NEXT: %10 = "addi32"(%9, %9) : (affineint, affineint) -> i32
    %w = "addi32"(%i, %i) : (affineint, affineint) -> i32
  } // CHECK }
  // cleanup loop (single iteration)
  // CHECK: %11 = "addi32"(%c100, %c100) : (affineint, affineint) -> i32
  // CHECK-NEXT: for %i2 = 0 to 17 {
    // CHECK-NEXT: %12 = "addi32"(%c100, %c100) : (affineint, affineint) -> i32
    // CHECK-NEXT: %13 = "addi32"(%12, %12) : (i32, i32) -> i32
  // CHECK-NEXT: }
  // CHECK-NEXT: %14 = "addi32"(%c100, %c100) : (affineint, affineint) -> i32
  return
}
