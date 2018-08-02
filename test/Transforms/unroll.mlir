// RUN: %S/../../mlir-opt %s -o - -unroll-innermost-loops | FileCheck %s

// CHECK-LABEL: mlfunc @loops1() {
mlfunc @loops1() {
  // CHECK: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: %c1_i32_0 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_1 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_2 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_3 = constant 1 : i32
    for %j = 1 to 4 {
      %x = constant 1 : i32
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// CHECK-LABEL: mlfunc @loops2() {
mlfunc @loops2() {
  // CHECK: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c0_i32_0 = constant 0 : i32
  // CHECK-NEXT: %c1_i32_1 = constant 1 : i32
  // CHECK-NEXT: %c2_i32_2 = constant 2 : i32
  // CHECK-NEXT: %c3_i32_3 = constant 3 : i32
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
     // CHECK: %0 = affine_apply (d0) -> (d0 + 1)(%c0_i32_0)
    // CHECK-NEXT: %1 = affine_apply (d0) -> (d0 + 1)(%c1_i32_1)
    // CHECK-NEXT: %2 = affine_apply (d0) -> (d0 + 1)(%c2_i32_2)
    // CHECK-NEXT: %3 = affine_apply (d0) -> (d0 + 1)(%c3_i32_3)
    for %j = 1 to 4 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
    }
  }    // CHECK:  }

  // CHECK: %c99 = constant 99 : affineint
  %k = "constant"(){value: 99} : () -> affineint
  // CHECK: for %i1 = 1 to 100 step 2 {
  for %m = 1 to 100 step 2 {
    // CHECK: %4 = affine_apply (d0) -> (d0 + 1)(%c0_i32)
    // CHECK-NEXT: %5 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c0_i32)[%c99]
    // CHECK-NEXT: %6 = affine_apply (d0) -> (d0 + 1)(%c1_i32)
    // CHECK-NEXT: %7 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c1_i32)[%c99]
    // CHECK-NEXT: %8 = affine_apply (d0) -> (d0 + 1)(%c2_i32)
    // CHECK-NEXT: %9 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c2_i32)[%c99]
    // CHECK-NEXT: %10 = affine_apply (d0) -> (d0 + 1)(%c3_i32)
    // CHECK-NEXT: %11 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c3_i32)[%c99]
    for %n = 1 to 4 {
      %y = "affine_apply" (%n) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %z = "affine_apply" (%n, %k) { map: (d0) [s0] -> (d0 + s0 + 1) } :
        (affineint, affineint) -> (affineint)
    }     // CHECK }
  }       // CHECK }
  return  // CHECK:  return
}         // CHECK }
