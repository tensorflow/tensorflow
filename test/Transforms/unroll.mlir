// RUN: mlir-opt %s -loop-unroll -unroll-full | FileCheck %s
// RUN: mlir-opt %s -loop-unroll -unroll-full -unroll-full-threshold=2 | FileCheck %s --check-prefix SHORT
// RUN: mlir-opt %s -loop-unroll -unroll-factor=4 | FileCheck %s --check-prefix UNROLL-BY-4

// CHECK: #map0 = (d0) -> (d0 + 1)
// CHECK: #map1 = (d0) -> (d0 + 2)
// CHECK: #map2 = (d0) -> (d0 + 3)
// CHECK: #map3 = (d0) -> (d0 + 4)
// CHECK: #map4 = (d0, d1) -> (d0 + 1, d1 + 2)
// CHECK: #map5 = (d0, d1) -> (d0 + 3, d1 + 4)
// CHECK: #map6 = (d0)[s0] -> (d0 + s0 + 1)
// CHECK: #map7 = (d0) -> (d0 + 5)
// CHECK: #map8 = (d0) -> (d0 + 6)
// CHECK: #map9 = (d0) -> (d0 + 7)
// CHECK: #map10 = (d0, d1) -> (d0 * 16 + d1)
// CHECK: #map11 = (d0) -> (d0 + 8)
// CHECK: #map12 = (d0) -> (d0 + 9)
// CHECK: #map13 = (d0) -> (d0 + 10)
// CHECK: #map14 = (d0) -> (d0 + 15)
// CHECK: #map15 = (d0) -> (d0 + 20)
// CHECK: #map16 = (d0) -> (d0 + 25)
// CHECK: #map17 = (d0) -> (d0 + 30)
// CHECK: #map18 = (d0) -> (d0 + 35)

// SHORT: #map0 = (d0) -> (d0 + 1)
// SHORT: #map1 = (d0) -> (d0 + 2)
// SHORT: #map2 = (d0, d1) -> (d0 + 1, d1 + 2)
// SHORT: #map3 = (d0, d1) -> (d0 + 3, d1 + 4)
// SHORT: #map4 = (d0)[s0] -> (d0 + s0 + 1)
// SHORT: #map5 = (d0, d1) -> (d0 * 16 + d1)

// UNROLL-BY-4: #map0 = (d0) -> (d0 + 1)
// UNROLL-BY-4: #map1 = (d0) -> (d0 + 2)
// UNROLL-BY-4: #map2 = (d0) -> (d0 + 3)
// UNROLL-BY-4: #map3 = (d0, d1) -> (d0 + 1, d1 + 2)
// UNROLL-BY-4: #map4 = (d0, d1) -> (d0 + 3, d1 + 4)
// UNROLL-BY-4: #map5 = (d0)[s0] -> (d0 + s0 + 1)
// UNROLL-BY-4: #map6 = (d0, d1) -> (d0 * 16 + d1)
// UNROLL-BY-4: #map7 = (d0) -> (d0 + 5)
// UNROLL-BY-4: #map8 = (d0) -> (d0 + 10)
// UNROLL-BY-4: #map9 = (d0) -> (d0 + 15)

// CHECK-LABEL: mlfunc @loop_nest_simplest() {
mlfunc @loop_nest_simplest() {
  // CHECK: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: %c1_i32 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_0 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_1 = constant 1 : i32
    // CHECK-NEXT: %c1_i32_2 = constant 1 : i32
    for %j = 1 to 4 {
      %x = constant 1 : i32
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// CHECK-LABEL: mlfunc @loop_nest_simple_iv_use() {
mlfunc @loop_nest_simple_iv_use() {
  // CHECK: %c1 = constant 1 : affineint
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: %0 = "addi32"(%c1, %c1) : (affineint, affineint) -> i32
    // CHECK: %1 = affine_apply #map0(%c1)
    // CHECK-NEXT:  %2 = "addi32"(%1, %1) : (affineint, affineint) -> i32
    // CHECK: %3 = affine_apply #map1(%c1)
    // CHECK-NEXT:  %4 = "addi32"(%3, %3) : (affineint, affineint) -> i32
    // CHECK: %5 = affine_apply #map2(%c1)
    // CHECK-NEXT:  %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
    for %j = 1 to 4 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// Operations in the loop body have results that are used therein.
// CHECK-LABEL: mlfunc @loop_nest_body_def_use() {
mlfunc @loop_nest_body_def_use() {
  // CHECK: %c0 = constant 0 : affineint
  // CHECK-NEXT: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: %c0_0 = constant 0 : affineint
    %c0 = constant 0 : affineint
    // CHECK:      %0 = affine_apply #map0(%c0)
    // CHECK-NEXT: %1 = "addi32"(%0, %c0_0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %2 = affine_apply #map0(%c0)
    // CHECK-NEXT: %3 = affine_apply #map0(%2)
    // CHECK-NEXT: %4 = "addi32"(%3, %c0_0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %5 = affine_apply #map1(%c0)
    // CHECK-NEXT: %6 = affine_apply #map0(%5)
    // CHECK-NEXT: %7 = "addi32"(%6, %c0_0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %8 = affine_apply #map2(%c0)
    // CHECK-NEXT: %9 = affine_apply #map0(%8)
    // CHECK-NEXT: %10 = "addi32"(%9, %c0_0) : (affineint, affineint) -> affineint
    for %j = 0 to 3 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %y = "addi32"(%x, %c0) : (affineint, affineint) -> affineint
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// CHECK-LABEL: mlfunc @loop_nest_strided() {
mlfunc @loop_nest_strided() {
  // CHECK: %c3 = constant 3 : affineint
  // CHECK-NEXT: %c3_0 = constant 3 : affineint
  // CHECK-NEXT: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // CHECK:      %0 = affine_apply #map0(%c3_0)
    // CHECK-NEXT: %1 = "addi32"(%0, %0) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %2 = affine_apply #map1(%c3_0)
    // CHECK-NEXT: %3 = affine_apply #map0(%2)
    // CHECK-NEXT: %4 = "addi32"(%3, %3) : (affineint, affineint) -> affineint
    for %j = 3 to 6 step 2 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %y = "addi32"(%x, %x) : (affineint, affineint) -> affineint
    }
    // CHECK:      %5 = affine_apply #map0(%c3)
    // CHECK-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %7 = affine_apply #map1(%c3)
    // CHECK-NEXT: %8 = affine_apply #map0(%7)
    // CHECK-NEXT: %9 = "addi32"(%8, %8) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %10 = affine_apply #map3(%c3)
    // CHECK-NEXT: %11 = affine_apply #map0(%10)
    // CHECK-NEXT: %12 = "addi32"(%11, %11) : (affineint, affineint) -> affineint
    for %k = 3 to 7 step 2 {
      %z = "affine_apply" (%k) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %w = "addi32"(%z, %z) : (affineint, affineint) -> affineint
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }

// CHECK-LABEL: mlfunc @loop_nest_multiple_results() {
mlfunc @loop_nest_multiple_results() {
  // CHECK: %c0 = constant 0 : affineint
  // CHECK-NEXT: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // CHECK: %0 = affine_apply #map4(%i0, %c0)
    // CHECK-NEXT: %1 = "addi32"(%0#0, %0#1) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %2 = affine_apply #map5(%i0, %c0)
    // CHECK-NEXT: %3 = "fma"(%2#0, %2#1, %0#0) : (affineint, affineint, affineint) -> (affineint, affineint)
    // CHECK-NEXT: %4 = affine_apply #map0(%c0)
    // CHECK-NEXT: %5 = affine_apply #map4(%i0, %4)
    // CHECK-NEXT: %6 = "addi32"(%5#0, %5#1) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %7 = affine_apply #map5(%i0, %4)
    // CHECK-NEXT: %8 = "fma"(%7#0, %7#1, %5#0) : (affineint, affineint, affineint) -> (affineint, affineint)
    for %j = 0 to 1 step 1 {
      %x = "affine_apply" (%i, %j) { map: (d0, d1) -> (d0 + 1, d1 + 2) } :
        (affineint, affineint) -> (affineint, affineint)
      %y = "addi32"(%x#0, %x#1) : (affineint, affineint) -> affineint
      %z = "affine_apply" (%i, %j) { map: (d0, d1) -> (d0 + 3, d1 + 4) } :
        (affineint, affineint) -> (affineint, affineint)
      %w = "fma"(%z#0, %z#1, %x#0) : (affineint, affineint, affineint) -> (affineint, affineint)
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }


// Imperfect loop nest. Unrolling innermost here yields a perfect nest.
// CHECK-LABEL: mlfunc @loop_nest_seq_imperfect(%arg0 : memref<128x128xf32>) {
mlfunc @loop_nest_seq_imperfect(%a : memref<128x128xf32>) {
  // CHECK: %c1 = constant 1 : affineint
  // CHECK-NEXT: %c128 = constant 128 : affineint
  %c128 = constant 128 : affineint
  // CHECK: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // CHECK: %0 = "vld"(%i0) : (affineint) -> i32
    %ld = "vld"(%i) : (affineint) -> i32
    // CHECK: %1 = affine_apply #map0(%c1)
    // CHECK-NEXT: %2 = "vmulf"(%c1, %1) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %3 = "vaddf"(%2, %2) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %4 = affine_apply #map0(%c1)
    // CHECK-NEXT: %5 = affine_apply #map0(%4)
    // CHECK-NEXT: %6 = "vmulf"(%4, %5) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %7 = "vaddf"(%6, %6) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %8 = affine_apply #map1(%c1)
    // CHECK-NEXT: %9 = affine_apply #map0(%8)
    // CHECK-NEXT: %10 = "vmulf"(%8, %9) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %11 = "vaddf"(%10, %10) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %12 = affine_apply #map2(%c1)
    // CHECK-NEXT: %13 = affine_apply #map0(%12)
    // CHECK-NEXT: %14 = "vmulf"(%12, %13) : (affineint, affineint) -> affineint
    // CHECK-NEXT: %15 = "vaddf"(%14, %14) : (affineint, affineint) -> affineint
    for %j = 1 to 4 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
       %y = "vmulf"(%j, %x) : (affineint, affineint) -> affineint
       %z = "vaddf"(%y, %y) : (affineint, affineint) -> affineint
    }
    // CHECK: %16 = "scale"(%c128, %i0) : (affineint, affineint) -> affineint
    %addr = "scale"(%c128, %i) : (affineint, affineint) -> affineint
    // CHECK: "vst"(%16, %i0) : (affineint, affineint) -> ()
    "vst"(%addr, %i) : (affineint, affineint) -> ()
  }       // CHECK }
  return  // CHECK:  return
}

// CHECK-LABEL: mlfunc @loop_nest_seq_multiple() {
mlfunc @loop_nest_seq_multiple() {
  // CHECK: %c1 = constant 1 : affineint
  // CHECK-NEXT: %c0 = constant 0 : affineint
  // CHECK-NEXT: %0 = affine_apply #map0(%c0)
  // CHECK-NEXT: "mul"(%0, %0) : (affineint, affineint) -> ()
  // CHECK-NEXT: %1 = affine_apply #map0(%c0)
  // CHECK-NEXT: %2 = affine_apply #map0(%1)
  // CHECK-NEXT: "mul"(%2, %2) : (affineint, affineint) -> ()
  // CHECK-NEXT: %3 = affine_apply #map1(%c0)
  // CHECK-NEXT: %4 = affine_apply #map0(%3)
  // CHECK-NEXT: "mul"(%4, %4) : (affineint, affineint) -> ()
  // CHECK-NEXT: %5 = affine_apply #map2(%c0)
  // CHECK-NEXT: %6 = affine_apply #map0(%5)
  // CHECK-NEXT: "mul"(%6, %6) : (affineint, affineint) -> ()
  for %j = 0 to 3 {
    %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
      (affineint) -> (affineint)
    "mul"(%x, %x) : (affineint, affineint) -> ()
  }

  // CHECK: %c99 = constant 99 : affineint
  %k = "constant"(){value: 99} : () -> affineint
  // CHECK: for %i0 = 1 to 100 step 2 {
  for %m = 1 to 100 step 2 {
    // CHECK: %7 = affine_apply #map0(%c1)
    // CHECK-NEXT: %8 = affine_apply #map6(%c1)[%c99]
    // CHECK-NEXT: %9 = affine_apply #map0(%c1)
    // CHECK-NEXT: %10 = affine_apply #map0(%9)
    // CHECK-NEXT: %11 = affine_apply #map6(%9)[%c99]
    // CHECK-NEXT: %12 = affine_apply #map1(%c1)
    // CHECK-NEXT: %13 = affine_apply #map0(%12)
    // CHECK-NEXT: %14 = affine_apply #map6(%12)[%c99]
    // CHECK-NEXT: %15 = affine_apply #map2(%c1)
    // CHECK-NEXT: %16 = affine_apply #map0(%15)
    // CHECK-NEXT: %17 = affine_apply #map6(%15)[%c99]
    for %n = 1 to 4 {
      %y = "affine_apply" (%n) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %z = "affine_apply" (%n, %k) { map: (d0) [s0] -> (d0 + s0 + 1) } :
        (affineint, affineint) -> (affineint)
    }     // CHECK }
  }       // CHECK }
  return  // CHECK:  return
}         // CHECK }

// SHORT-LABEL: mlfunc @loop_nest_outer_unroll() {
mlfunc @loop_nest_outer_unroll() {
  // SHORT:      for %i0 = 1 to 4 {
  // SHORT-NEXT:   %0 = affine_apply #map0(%i0)
  // SHORT-NEXT:   %1 = "addi32"(%0, %0) : (affineint, affineint) -> affineint
  // SHORT-NEXT: }
  // SHORT-NEXT: for %i1 = 1 to 4 {
  // SHORT-NEXT:   %2 = affine_apply #map0(%i1)
  // SHORT-NEXT:   %3 = "addi32"(%2, %2) : (affineint, affineint) -> affineint
  // SHORT-NEXT: }
  for %i = 1 to 2 {
    for %j = 1 to 4 {
      %x = "affine_apply" (%j) { map: (d0) -> (d0 + 1) } :
        (affineint) -> (affineint)
      %y = "addi32"(%x, %x) : (affineint, affineint) -> affineint
    }
  }
  return  // SHORT:  return
}         // SHORT }

// We aren't doing any file check here. We just need this test case to
// successfully run. Both %i0 and i1 will get unrolled here with the min trip
// count threshold set to 2.
// SHORT-LABEL: mlfunc @loop_nest_seq_long() -> i32 {
mlfunc @loop_nest_seq_long() -> i32 {
  %A = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
  %B = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
  %C = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>

  %zero = constant 0 : i32
  %one = constant 1 : i32
  %two = constant 2 : i32

  %zero_idx = constant 0 : affineint

  for %n0 = 0 to 512 {
    for %n1 = 0 to 7 {
      store %one,  %A[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
      store %two,  %B[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
      store %zero, %C[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
    }
  }

  for %i0 = 0 to 1 {
    for %i1 = 0 to 1 {
      for %i2 = 0 to 7 {
        %b2 = "affine_apply" (%i1, %i2) {map: (d0, d1) -> (16*d0 + d1)} : (affineint, affineint) -> affineint
        %x = load %B[%i0, %b2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
        "op1"(%x) : (i32) -> ()
      }
      for %j1 = 0 to 7 {
        for %j2 = 0 to 7 {
          %a2 = "affine_apply" (%i1, %j2) {map: (d0, d1) -> (16*d0 + d1)} : (affineint, affineint) -> affineint
          %v203 = load %A[%j1, %a2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
          "op2"(%v203) : (i32) -> ()
        }
        for %k2 = 0 to 7 {
          %s0 = "op3"() : () -> i32
          %c2 = "affine_apply" (%i0, %k2) {map: (d0, d1) -> (16*d0 + d1)} : (affineint, affineint) -> affineint
          %s1 =  load %C[%j1, %c2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
          %s2 = "addi32"(%s0, %s1) : (i32, i32) -> i32
          store %s2, %C[%j1, %c2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
        }
      }
      "op4"() : () -> ()
    }
  }
  %ret = load %C[%zero_idx, %zero_idx] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
  return %ret : i32
}

// UNROLL-BY-4-LABEL: mlfunc @unroll_unit_stride_no_cleanup() {
mlfunc @unroll_unit_stride_no_cleanup() {
  // UNROLL-BY-4: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 1 to 8 step 4 {
    // UNROLL-BY-4-NEXT: %0 = "addi32"([[L1]], [[L1]]) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %2 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %3 = "addi32"(%2, %2) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %8 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %9 = "addi32"(%8, %8) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 1 to 8 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
    // empty loop
    // UNROLL-BY-4: for %i2 = 1 to 8 {
    for %k = 1 to 8 {
    }
  }
  return
}

// UNROLL-BY-4-LABEL: mlfunc @unroll_unit_stride_cleanup() {
mlfunc @unroll_unit_stride_cleanup() {
  // UNROLL-BY-4: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 1 to 8 step 4 {
    // UNROLL-BY-4-NEXT: %0 = "addi32"([[L1]], [[L1]]) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %2 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %3 = "addi32"(%2, %2) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %8 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %9 = "addi32"(%8, %8) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: for [[L2:%i[0-9]+]] = 9 to 10 {
    // UNROLL-BY-4-NEXT: %11 = "addi32"([[L2]], [[L2]]) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %12 = "addi32"(%11, %11) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 1 to 10 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}

// UNROLL-BY-4-LABEL: mlfunc @unroll_non_unit_stride_cleanup() {
mlfunc @unroll_non_unit_stride_cleanup() {
  // UNROLL-BY-4: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 2 to 37 step 20 {
    // UNROLL-BY-4-NEXT: %0 = "addi32"([[L1]], [[L1]]) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %2 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %3 = "addi32"(%2, %2) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %8 = affine_apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %9 = "addi32"(%8, %8) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: for [[L2:%i[0-9]+]] = 42 to 48 step 5 {
    // UNROLL-BY-4-NEXT: %11 = "addi32"([[L2]], [[L2]]) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %12 = "addi32"(%11, %11) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 2 to 48 step 5 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}

// Both the unrolled loop and the cleanup loop are single iteration loops.
mlfunc @loop_nest_single_iteration_after_unroll(%N: affineint) {
  // UNROLL-BY-4: %c0 = constant 0 : affineint
  // UNROLL-BY-4: %c4 = constant 4 : affineint
  // UNROLL-BY-4: for %i0 = 1 to %arg0 {
  for %i = 1 to %N {
    // UNROLL-BY-4: %0 = "addi32"(%c0, %c0) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %1 = affine_apply #map0(%c0)
    // UNROLL-BY-4-NEXT: %2 = "addi32"(%1, %1) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %3 = affine_apply #map1(%c0)
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine_apply #map2(%c0)
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%c4, %c4) : (affineint, affineint) -> i32
    // UNROLL-BY-4-NOT: for
    for %j = 0 to 4 {
      %x = "addi32"(%j, %j) : (affineint, affineint) -> i32
    } // UNROLL-BY-4-NOT: }
  } // UNROLL-BY-4:  }
  return
}

// Test cases with loop bound operands.

// No cleanup will be generated here.
// UNROLL-BY-4-LABEL: mlfunc @loop_nest_operand1() {
mlfunc @loop_nest_operand1() {
  // UNROLL-BY-4: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // UNROLL-BY-4: %0 = "foo"() : () -> i32
    // UNROLL-BY-4: %1 = "foo"() : () -> i32
    // UNROLL-BY-4: %2 = "foo"() : () -> i32
    // UNROLL-BY-4: %3 = "foo"() : () -> i32
    for %j = (d0) -> (0) (%i) to (d0) -> (d0 - d0 mod 4 - 1) (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// No cleanup will be generated here.
// UNROLL-BY-4-LABEL: mlfunc @loop_nest_operand2() {
mlfunc @loop_nest_operand2() {
  // UNROLL-BY-4: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // UNROLL-BY-4: %0 = "foo"() : () -> i32
    // UNROLL-BY-4: %1 = "foo"() : () -> i32
    // UNROLL-BY-4: %2 = "foo"() : () -> i32
    // UNROLL-BY-4: %3 = "foo"() : () -> i32
    for %j = (d0) -> (d0) (%i) to (d0) -> (5*d0 + 3) (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// Difference between loop bounds is constant, but not a multiple of unroll
// factor. The cleanup loop happens to be a single iteration one and is promoted.
// UNROLL-BY-4-LABEL: mlfunc @loop_nest_operand3() {
mlfunc @loop_nest_operand3() {
  // UNROLL-BY-4: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // UNROLL-BY-4: for %i1 = (d0) -> (d0)(%i0) to #map{{[0-9]+}}(%i0) step 4 {
    // UNROLL-BY-4-NEXT: %0 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %1 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %2 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %3 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: %4 = "foo"() : () -> i32
    for %j = (d0) -> (d0) (%i) to (d0) -> (d0 + 8) (%i) {
      %x = "foo"() : () -> i32
    }
  } // UNROLL-BY-4: }
  return
}

// UNROLL-BY-4-LABEL: mlfunc @loop_nest_operand4(%arg0 : affineint) {
mlfunc @loop_nest_operand4(%N : affineint) {
  // UNROLL-BY-4: for %i0 = 1 to 100 {
  for %i = 1 to 100 {
    // UNROLL-BY-4: for %i1 = ()[s0] -> (1)()[%arg0] to #map{{[0-9]+}}()[%arg0] step 4 {
    // UNROLL-BY-4: %0 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %1 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %2 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %3 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: }
    // A cleanup loop will be be generated here.
    // UNROLL-BY-4-NEXT: for %i2 = #map{{[0-9]+}}()[%arg0] to %arg0 {
    // UNROLL-BY-4-NEXT: %4 = "foo"() : () -> i32
    // UNROLL-BY-4_NEXT: }
    // Specify the lower bound so that both lb and ub operands match.
    for %j = ()[s0] -> (1)()[%N] to %N {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// CHECK-LABEL: mlfunc @loop_nest_unroll_full() {
mlfunc @loop_nest_unroll_full() {
  // CHECK-NEXT: %0 = "foo"() : () -> i32
  // CHECK-NEXT: %1 = "bar"() : () -> i32
  // CHECK-NEXT:  return
  for %i = 0 to 0 {
    %x = "foo"() : () -> i32
    %y = "bar"() : () -> i32
  }
  return
} // CHECK }
