// RUN: %S/../../mlir-opt %s -o - -unroll-innermost-loops | FileCheck %s

// CHECK-LABEL: mlfunc @loop_nest_simplest() {
mlfunc @loop_nest_simplest() {
  // CHECK: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c4_i32 = constant 4 : i32
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

// CHECK-LABEL: mlfunc @loop_nest_simple_iv_use() {
mlfunc @loop_nest_simple_iv_use() {
  // CHECK: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c4_i32 = constant 4 : i32
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK:       %0 = "addi32"(%c1_i32, %c1_i32) : (i32, i32) -> i32
    // CHECK-NEXT:  %1 = "addi32"(%c2_i32, %c2_i32) : (i32, i32) -> i32
    // CHECK-NEXT:  %2 = "addi32"(%c3_i32, %c3_i32) : (i32, i32) -> i32
    // CHECK-NEXT:  %3 = "addi32"(%c4_i32, %c4_i32) : (i32, i32) -> i32
    for %j = 1 to 4 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// CHECK-LABEL: mlfunc @loop_nest_strided() {
mlfunc @loop_nest_strided() {
  // CHECK: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c5_i32 = constant 5 : i32
  // CHECK-NEXT: %c7_i32 = constant 7 : i32
  // CHECK-NEXT: %c3_i32_0 = constant 3 : i32
  // CHECK-NEXT: %c5_i32_1 = constant 5 : i32
  // CHECK-NEXT: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // CHECK:      %0 = affine_apply (d0) -> (d0 + 1)(%c3_i32_0)
    // CHECK-NEXT: %1 = "addi32"(%0, %0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %2 = affine_apply (d0) -> (d0 + 1)(%c5_i32_1)
    // CHECK-NEXT: %3 = "addi32"(%2, %2) : (affineint, affineint) -> affineint
    for %j = 3 to 6 step 2 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %y = "addi32"(%x, %x) : (affineint, affineint) -> affineint
    }
    // CHECK:      %4 = affine_apply (d0) -> (d0 + 1)(%c3_i32)
    // CHECK-NEXT: %5 = "addi32"(%4, %4) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %6 = affine_apply (d0) -> (d0 + 1)(%c5_i32)
    // CHECK-NEXT: %7 = "addi32"(%6, %6) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %8 = affine_apply (d0) -> (d0 + 1)(%c7_i32)
    // CHECK-NEXT: %9 = "addi32"(%8, %8) : (affineint, affineint) -> affineint
    for %k = 3 to 7 step 2 {
      %z = "affine_apply" (%k) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %w = "addi32"(%z, %z) : (affineint, affineint) -> affineint
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// Operations in the loop body have results that are used therein.
// CHECK-LABEL: mlfunc @loop_nest_body_def_use() {
mlfunc @loop_nest_body_def_use() {
  // CHECK: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: %c0 = constant 0 : affineint
    %c0 = constant 0 : affineint
    // CHECK:      %0 = affine_apply (d0) -> (d0 + 1)(%c0_i32)
    // CHECK-NEXT: %1 = "addi32"(%0, %c0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %2 = affine_apply (d0) -> (d0 + 1)(%c1_i32)
    // CHECK-NEXT: %3 = "addi32"(%2, %c0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %4 = affine_apply (d0) -> (d0 + 1)(%c2_i32)
    // CHECK-NEXT: %5 = "addi32"(%4, %c0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %6 = affine_apply (d0) -> (d0 + 1)(%c3_i32)
    // CHECK-NEXT: %7 = "addi32"(%6, %c0) : (affineint, affineint) -> affineint
    for %j = 0 to 3 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %y = "addi32"(%x, %c0) : (affineint, affineint) -> affineint
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }


// Imperfect loop nest. Unrolling innermost here yields a perfect nest.
// CHECK-LABEL: mlfunc @loop_nest_seq_imperfect(%arg0 : memref<128x128xf32>) {
mlfunc @loop_nest_seq_imperfect(%a : memref<128x128xf32>) {
  // CHECK: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c4_i32 = constant 4 : i32
  // CHECK-NEXT: %c128 = constant 128 : affineint
  %c128 = constant 128 : affineint
  // CHECK: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // CHECK: %0 = "vld"(%i0) : (affineint) -> i32
    %ld = "vld"(%i) : (affineint) -> i32
    // CHECK: %1 = affine_apply (d0) -> (d0 + 1)(%c1_i32)
    // CHECK-NEXT: %2 = "vmulf"(%c1_i32, %1) : (i32, affineint) -> affineint
    // CHECK-NEXT: %3 = "vaddf"(%2, %2) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %4 = affine_apply (d0) -> (d0 + 1)(%c2_i32)
    // CHECK-NEXT: %5 = "vmulf"(%c2_i32, %4) : (i32, affineint) -> affineint
    // CHECK-NEXT: %6 = "vaddf"(%5, %5) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %7 = affine_apply (d0) -> (d0 + 1)(%c3_i32)
    // CHECK-NEXT: %8 = "vmulf"(%c3_i32, %7) : (i32, affineint) -> affineint
    // CHECK-NEXT: %9 = "vaddf"(%8, %8) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %10 = affine_apply (d0) -> (d0 + 1)(%c4_i32)
    // CHECK-NEXT: %11 = "vmulf"(%c4_i32, %10) : (i32, affineint) -> affineint
    // CHECK-NEXT: %12 = "vaddf"(%11, %11) : (affineint, affineint) -> affineint
    for %j = 1 to 4 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
       %y = "vmulf"(%j, %x) : (affineint, affineint) -> affineint
       %z = "vaddf"(%y, %y) : (affineint, affineint) -> affineint
    }
    // CHECK: %13 = "scale"(%c128, %i0) : (affineint, affineint) -> affineint
    %addr = "scale"(%c128, %i) : (affineint, affineint) -> affineint
    // CHECK: "vst"(%13, %i0) : (affineint, affineint) -> ()
    "vst"(%addr, %i) : (affineint, affineint) -> ()
  }       // CHECK }
  return  // CHECK:  return
}

// CHECK-LABEL: mlfunc @loop_nest_seq_multiple() {
mlfunc @loop_nest_seq_multiple() {
  // CHECK: %c1_i32 = constant 1 : i32
  // CHECK-NEXT: %c2_i32 = constant 2 : i32
  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  // CHECK-NEXT: %c4_i32 = constant 4 : i32
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
  // CHECK-NEXT: %c2_i32_1 = constant 2 : i32
  // CHECK-NEXT: %c3_i32_2 = constant 3 : i32
  // CHECK-NEXT: %0 = affine_apply (d0) -> (d0 + 1)(%c0_i32)
  // CHECK-NEXT: "mul"(%0, %0) : (affineint, affineint) -> ()
  // CHECK-NEXT: %1 = affine_apply (d0) -> (d0 + 1)(%c1_i32_0)
  // CHECK-NEXT: "mul"(%1, %1) : (affineint, affineint) -> ()
  // CHECK-NEXT: %2 = affine_apply (d0) -> (d0 + 1)(%c2_i32_1)
  // CHECK-NEXT: "mul"(%2, %2) : (affineint, affineint) -> ()
  // CHECK-NEXT: %3 = affine_apply (d0) -> (d0 + 1)(%c3_i32_2)
  // CHECK-NEXT: "mul"(%3, %3) : (affineint, affineint) -> ()
  for %j = 0 to 3 {
    %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
      (affineint) -> (affineint)
    "mul"(%x, %x) : (affineint, affineint) -> ()
  }

  // CHECK: %c99 = constant 99 : affineint
  %k = "constant"(){value: 99} : () -> affineint
  // CHECK: for %i0 = 1 to 100 step 2 {
  for %m = 1 to 100 step 2 {
    // CHECK: %4 = affine_apply (d0) -> (d0 + 1)(%c1_i32)
    // CHECK-NEXT: %5 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c1_i32)[%c99]
    // CHECK-NEXT: %6 = affine_apply (d0) -> (d0 + 1)(%c2_i32)
    // CHECK-NEXT: %7 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c2_i32)[%c99]
    // CHECK-NEXT: %8 = affine_apply (d0) -> (d0 + 1)(%c3_i32)
    // CHECK-NEXT: %9 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c3_i32)[%c99]
    // CHECK-NEXT: %10 = affine_apply (d0) -> (d0 + 1)(%c4_i32)
    // CHECK-NEXT: %11 = affine_apply (d0)[s0] -> (d0 + s0 + 1)(%c4_i32)[%c99]
    for %n = 1 to 4 {
      %y = "affine_apply" (%n) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %z = "affine_apply" (%n, %k) { map: (d0) [s0] -> (d0 + s0 + 1) } :
        (affineint, affineint) -> (affineint)
    }     // CHECK }
  }       // CHECK }
  return  // CHECK:  return
}         // CHECK }
