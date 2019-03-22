// RUN: mlir-opt %s -loop-unroll -unroll-full | FileCheck %s --check-prefix UNROLL-FULL
// RUN: mlir-opt %s -loop-unroll -unroll-full -unroll-full-threshold=2 | FileCheck %s --check-prefix SHORT
// RUN: mlir-opt %s -loop-unroll -unroll-factor=4 | FileCheck %s --check-prefix UNROLL-BY-4
// RUN: mlir-opt %s -loop-unroll -unroll-factor=1 | FileCheck %s --check-prefix UNROLL-BY-1

// UNROLL-FULL-DAG: [[MAP0:#map[0-9]+]] = (d0) -> (d0 + 1)
// UNROLL-FULL-DAG: [[MAP1:#map[0-9]+]] = (d0) -> (d0 + 2)
// UNROLL-FULL-DAG: [[MAP2:#map[0-9]+]] = (d0) -> (d0 + 3)
// UNROLL-FULL-DAG: [[MAP3:#map[0-9]+]] = (d0) -> (d0 + 4)
// UNROLL-FULL-DAG: [[MAP4:#map[0-9]+]] = (d0, d1) -> (d0 + 1)
// UNROLL-FULL-DAG: [[MAP5:#map[0-9]+]] = (d0, d1) -> (d0 + 3)
// UNROLL-FULL-DAG: [[MAP6:#map[0-9]+]] = (d0)[s0] -> (d0 + s0 + 1)

// SHORT-DAG: [[MAP0:#map[0-9]+]] = (d0) -> (d0 + 1)

// UNROLL-BY-4-DAG: [[MAP0:#map[0-9]+]] = (d0) -> (d0 + 1)
// UNROLL-BY-4-DAG: [[MAP1:#map[0-9]+]] = (d0) -> (d0 + 2)
// UNROLL-BY-4-DAG: [[MAP2:#map[0-9]+]] = (d0) -> (d0 + 3)
// UNROLL-BY-4-DAG: [[MAP3:#map[0-9]+]] = (d0, d1) -> (d0 + 1)
// UNROLL-BY-4-DAG: [[MAP4:#map[0-9]+]] = (d0, d1) -> (d0 + 3)
// UNROLL-BY-4-DAG: [[MAP5:#map[0-9]+]] = (d0)[s0] -> (d0 + s0 + 1)
// UNROLL-BY-4-DAG: [[MAP6:#map[0-9]+]] = (d0, d1) -> (d0 * 16 + d1)
// UNROLL-BY-4-DAG: [[MAP11:#map[0-9]+]] = (d0) -> (d0)
// UNROLL-BY-4-DAG: [[MAP_TRIP_COUNT_MULTIPLE_FOUR:#map[0-9]+]] = ()[s0, s1, s2] -> (s0 + ((-s0 + s1) floordiv 4) * 4, s0 + ((-s0 + s2) floordiv 4) * 4, s0 + ((-s0 + 1024) floordiv 4) * 4)

// UNROLL-FULL-LABEL: func @loop_nest_simplest() {
func @loop_nest_simplest() {
  // UNROLL-FULL: for %i0 = 0 to 100 step 2 {
  for %i = 0 to 100 step 2 {
    // UNROLL-FULL: %c1_i32 = constant 1 : i32
    // UNROLL-FULL-NEXT: %c1_i32_0 = constant 1 : i32
    // UNROLL-FULL-NEXT: %c1_i32_1 = constant 1 : i32
    // UNROLL-FULL-NEXT: %c1_i32_2 = constant 1 : i32
    for %j = 0 to 4 {
      %x = constant 1 : i32
    }
  }       // UNROLL-FULL:  }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }

// UNROLL-FULL-LABEL: func @loop_nest_simple_iv_use() {
func @loop_nest_simple_iv_use() {
  // UNROLL-FULL: %c0 = constant 0 : index
  // UNROLL-FULL-NEXT: for %i0 = 0 to 100 step 2 {
  for %i = 0 to 100 step 2 {
    // UNROLL-FULL: %0 = "addi32"(%c0, %c0) : (index, index) -> i32
    // UNROLL-FULL: %1 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT:  %2 = "addi32"(%1, %1) : (index, index) -> i32
    // UNROLL-FULL: %3 = affine.apply [[MAP1]](%c0)
    // UNROLL-FULL-NEXT:  %4 = "addi32"(%3, %3) : (index, index) -> i32
    // UNROLL-FULL: %5 = affine.apply [[MAP2]](%c0)
    // UNROLL-FULL-NEXT:  %6 = "addi32"(%5, %5) : (index, index) -> i32
    for %j = 0 to 4 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
    }
  }       // UNROLL-FULL:  }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }

// Operations in the loop body have results that are used therein.
// UNROLL-FULL-LABEL: func @loop_nest_body_def_use() {
func @loop_nest_body_def_use() {
  // UNROLL-FULL: %c0 = constant 0 : index
  // UNROLL-FULL-NEXT: for %i0 = 0 to 100 step 2 {
  for %i = 0 to 100 step 2 {
    // UNROLL-FULL: %c0_0 = constant 0 : index
    %c0 = constant 0 : index
    // UNROLL-FULL:      %0 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %1 = "addi32"(%0, %c0_0) : (index, index) -> index
    // UNROLL-FULL-NEXT: %2 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %3 = affine.apply [[MAP0]](%2)
    // UNROLL-FULL-NEXT: %4 = "addi32"(%3, %c0_0) : (index, index) -> index
    // UNROLL-FULL-NEXT: %5 = affine.apply [[MAP1]](%c0)
    // UNROLL-FULL-NEXT: %6 = affine.apply [[MAP0]](%5)
    // UNROLL-FULL-NEXT: %7 = "addi32"(%6, %c0_0) : (index, index) -> index
    // UNROLL-FULL-NEXT: %8 = affine.apply [[MAP2]](%c0)
    // UNROLL-FULL-NEXT: %9 = affine.apply [[MAP0]](%8)
    // UNROLL-FULL-NEXT: %10 = "addi32"(%9, %c0_0) : (index, index) -> index
    for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
      %y = "addi32"(%x, %c0) : (index, index) -> index
    }
  }       // UNROLL-FULL:  }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }

// UNROLL-FULL-LABEL: func @loop_nest_strided() {
func @loop_nest_strided() {
  // UNROLL-FULL: %c2 = constant 2 : index
  // UNROLL-FULL-NEXT: %c2_0 = constant 2 : index
  // UNROLL-FULL-NEXT: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-FULL:      %0 = affine.apply [[MAP0]](%c2_0)
    // UNROLL-FULL-NEXT: %1 = "addi32"(%0, %0) : (index, index) -> index
    // UNROLL-FULL-NEXT: %2 = affine.apply [[MAP1]](%c2_0)
    // UNROLL-FULL-NEXT: %3 = affine.apply [[MAP0]](%2)
    // UNROLL-FULL-NEXT: %4 = "addi32"(%3, %3) : (index, index) -> index
    for %j = 2 to 6 step 2 {
      %x = "affine.apply" (%j) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
      %y = "addi32"(%x, %x) : (index, index) -> index
    }
    // UNROLL-FULL:      %5 = affine.apply [[MAP0]](%c2)
    // UNROLL-FULL-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> index
    // UNROLL-FULL-NEXT: %7 = affine.apply [[MAP1]](%c2)
    // UNROLL-FULL-NEXT: %8 = affine.apply [[MAP0]](%7)
    // UNROLL-FULL-NEXT: %9 = "addi32"(%8, %8) : (index, index) -> index
    // UNROLL-FULL-NEXT: %10 = affine.apply [[MAP3]](%c2)
    // UNROLL-FULL-NEXT: %11 = affine.apply [[MAP0]](%10)
    // UNROLL-FULL-NEXT: %12 = "addi32"(%11, %11) : (index, index) -> index
    for %k = 2 to 7 step 2 {
      %z = "affine.apply" (%k) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
      %w = "addi32"(%z, %z) : (index, index) -> index
    }
  }       // UNROLL-FULL:  }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }

// UNROLL-FULL-LABEL: func @loop_nest_multiple_results() {
func @loop_nest_multiple_results() {
  // UNROLL-FULL: %c0 = constant 0 : index
  // UNROLL-FULL-NEXT: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-FULL: %0 = affine.apply [[MAP4]](%i0, %c0)
    // UNROLL-FULL-NEXT: %1 = "addi32"(%0, %0) : (index, index) -> index
    // UNROLL-FULL-NEXT: %2 = affine.apply #map{{.*}}(%i0, %c0)
    // UNROLL-FULL-NEXT: %3 = "fma"(%2, %0, %0) : (index, index, index) -> (index, index)
    // UNROLL-FULL-NEXT: %4 = affine.apply #map{{.*}}(%c0)
    // UNROLL-FULL-NEXT: %5 = affine.apply #map{{.*}}(%i0, %4)
    // UNROLL-FULL-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> index
    // UNROLL-FULL-NEXT: %7 = affine.apply #map{{.*}}(%i0, %4)
    // UNROLL-FULL-NEXT: %8 = "fma"(%7, %5, %5) : (index, index, index) -> (index, index)
    for %j = 0 to 2 step 1 {
      %x = affine.apply (d0, d1) -> (d0 + 1) (%i, %j)
      %y = "addi32"(%x, %x) : (index, index) -> index
      %z = affine.apply (d0, d1) -> (d0 + 3) (%i, %j)
      %w = "fma"(%z, %x, %x) : (index, index, index) -> (index, index)
    }
  }       // UNROLL-FULL:  }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }


// Imperfect loop nest. Unrolling innermost here yields a perfect nest.
// UNROLL-FULL-LABEL: func @loop_nest_seq_imperfect(%arg0: memref<128x128xf32>) {
func @loop_nest_seq_imperfect(%a : memref<128x128xf32>) {
  // UNROLL-FULL: %c0 = constant 0 : index
  // UNROLL-FULL-NEXT: %c128 = constant 128 : index
  %c128 = constant 128 : index
  // UNROLL-FULL: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-FULL: %0 = "vld"(%i0) : (index) -> i32
    %ld = "vld"(%i) : (index) -> i32
    // UNROLL-FULL: %1 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %2 = "vmulf"(%c0, %1) : (index, index) -> index
    // UNROLL-FULL-NEXT: %3 = "vaddf"(%2, %2) : (index, index) -> index
    // UNROLL-FULL-NEXT: %4 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %5 = affine.apply [[MAP0]](%4)
    // UNROLL-FULL-NEXT: %6 = "vmulf"(%4, %5) : (index, index) -> index
    // UNROLL-FULL-NEXT: %7 = "vaddf"(%6, %6) : (index, index) -> index
    // UNROLL-FULL-NEXT: %8 = affine.apply [[MAP1]](%c0)
    // UNROLL-FULL-NEXT: %9 = affine.apply [[MAP0]](%8)
    // UNROLL-FULL-NEXT: %10 = "vmulf"(%8, %9) : (index, index) -> index
    // UNROLL-FULL-NEXT: %11 = "vaddf"(%10, %10) : (index, index) -> index
    // UNROLL-FULL-NEXT: %12 = affine.apply [[MAP2]](%c0)
    // UNROLL-FULL-NEXT: %13 = affine.apply [[MAP0]](%12)
    // UNROLL-FULL-NEXT: %14 = "vmulf"(%12, %13) : (index, index) -> index
    // UNROLL-FULL-NEXT: %15 = "vaddf"(%14, %14) : (index, index) -> index
    for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
       %y = "vmulf"(%j, %x) : (index, index) -> index
       %z = "vaddf"(%y, %y) : (index, index) -> index
    }
    // UNROLL-FULL: %16 = "scale"(%c128, %i0) : (index, index) -> index
    %addr = "scale"(%c128, %i) : (index, index) -> index
    // UNROLL-FULL: "vst"(%16, %i0) : (index, index) -> ()
    "vst"(%addr, %i) : (index, index) -> ()
  }       // UNROLL-FULL }
  return  // UNROLL-FULL:  return
}

// UNROLL-FULL-LABEL: func @loop_nest_seq_multiple() {
func @loop_nest_seq_multiple() {
  // UNROLL-FULL: c0 = constant 0 : index
  // UNROLL-FULL-NEXT: %c0_0 = constant 0 : index
  // UNROLL-FULL-NEXT: %0 = affine.apply [[MAP0]](%c0_0)
  // UNROLL-FULL-NEXT: "mul"(%0, %0) : (index, index) -> ()
  // UNROLL-FULL-NEXT: %1 = affine.apply [[MAP0]](%c0_0)
  // UNROLL-FULL-NEXT: %2 = affine.apply [[MAP0]](%1)
  // UNROLL-FULL-NEXT: "mul"(%2, %2) : (index, index) -> ()
  // UNROLL-FULL-NEXT: %3 = affine.apply [[MAP1]](%c0_0)
  // UNROLL-FULL-NEXT: %4 = affine.apply [[MAP0]](%3)
  // UNROLL-FULL-NEXT: "mul"(%4, %4) : (index, index) -> ()
  // UNROLL-FULL-NEXT: %5 = affine.apply [[MAP2]](%c0_0)
  // UNROLL-FULL-NEXT: %6 = affine.apply [[MAP0]](%5)
  // UNROLL-FULL-NEXT: "mul"(%6, %6) : (index, index) -> ()
  for %j = 0 to 4 {
    %x = "affine.apply" (%j) { map: (d0) -> (d0 + 1) } :
      (index) -> (index)
    "mul"(%x, %x) : (index, index) -> ()
  }

  // UNROLL-FULL: %c99 = constant 99 : index
  %k = constant 99 : index
  // UNROLL-FULL: for %i0 = 0 to 100 step 2 {
  for %m = 0 to 100 step 2 {
    // UNROLL-FULL: %7 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %8 = affine.apply [[MAP6]](%c0)[%c99]
    // UNROLL-FULL-NEXT: %9 = affine.apply [[MAP0]](%c0)
    // UNROLL-FULL-NEXT: %10 = affine.apply [[MAP0]](%9)
    // UNROLL-FULL-NEXT: %11 = affine.apply [[MAP6]](%9)[%c99]
    // UNROLL-FULL-NEXT: %12 = affine.apply [[MAP1]](%c0)
    // UNROLL-FULL-NEXT: %13 = affine.apply [[MAP0]](%12)
    // UNROLL-FULL-NEXT: %14 = affine.apply [[MAP6]](%12)[%c99]
    // UNROLL-FULL-NEXT: %15 = affine.apply [[MAP2]](%c0)
    // UNROLL-FULL-NEXT: %16 = affine.apply [[MAP0]](%15)
    // UNROLL-FULL-NEXT: %17 = affine.apply [[MAP6]](%15)[%c99]
    for %n = 0 to 4 {
      %y = "affine.apply" (%n) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
      %z = "affine.apply" (%n, %k) { map: (d0) [s0] -> (d0 + s0 + 1) } :
        (index, index) -> (index)
    }     // UNROLL-FULL }
  }       // UNROLL-FULL }
  return  // UNROLL-FULL:  return
}         // UNROLL-FULL }

// UNROLL-FULL-LABEL: func @loop_nest_unroll_full() {
func @loop_nest_unroll_full() {
  // UNROLL-FULL-NEXT: %0 = "foo"() : () -> i32
  // UNROLL-FULL-NEXT: %1 = "bar"() : () -> i32
  // UNROLL-FULL-NEXT:  return
  for %i = 0 to 1 {
    %x = "foo"() : () -> i32
    %y = "bar"() : () -> i32
  }
  return
} // UNROLL-FULL }

// SHORT-LABEL: func @loop_nest_outer_unroll() {
func @loop_nest_outer_unroll() {
  // SHORT:      for %i0 = 0 to 4 {
  // SHORT-NEXT:   %0 = affine.apply [[MAP0]](%i0)
  // SHORT-NEXT:   %1 = "addi32"(%0, %0) : (index, index) -> index
  // SHORT-NEXT: }
  // SHORT-NEXT: for %i1 = 0 to 4 {
  // SHORT-NEXT:   %2 = affine.apply [[MAP0]](%i1)
  // SHORT-NEXT:   %3 = "addi32"(%2, %2) : (index, index) -> index
  // SHORT-NEXT: }
  for %i = 0 to 2 {
    for %j = 0 to 4 {
      %x = "affine.apply" (%j) { map: (d0) -> (d0 + 1) } :
        (index) -> (index)
      %y = "addi32"(%x, %x) : (index, index) -> index
    }
  }
  return  // SHORT:  return
}         // SHORT }

// We are doing a minimal FileCheck here. We just need this test case to
// successfully run. Both %x and %y will get unrolled here as the min trip
// count threshold set to 2.
// SHORT-LABEL: func @loop_nest_seq_long() -> i32 {
func @loop_nest_seq_long() -> i32 {
  %A = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
  %B = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
  %C = alloc() : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>

  %zero = constant 0 : i32
  %one = constant 1 : i32
  %two = constant 2 : i32

  %zero_idx = constant 0 : index

  // CHECK: for %i0 = 0 to 512
  for %n0 = 0 to 512 {
    // CHECK: for %i1 = 0 to 8
    for %n1 = 0 to 8 {
      store %one,  %A[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
      store %two,  %B[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
      store %zero, %C[%n0, %n1] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
    }
  }

  for %x = 0 to 2 {
    for %y = 0 to 2 {
      // CHECK: for %i2
      for %i2 = 0 to 8 {
        // CHECK-NOT: for %i3
        // CHECK: %{{[0-9]+}} = affine.apply
        %b2 = "affine.apply" (%y, %i2) {map: (d0, d1) -> (16*d0 + d1)} : (index, index) -> index
        %z = load %B[%x, %b2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
        "op1"(%z) : (i32) -> ()
      }
      for %j1 = 0 to 8 {
        for %j2 = 0 to 8 {
          %a2 = "affine.apply" (%y, %j2) {map: (d0, d1) -> (16*d0 + d1)} : (index, index) -> index
          %v203 = load %A[%j1, %a2] : memref<512 x 512 x i32, (d0, d1) -> (d0, d1), 2>
          "op2"(%v203) : (i32) -> ()
        }
        for %k2 = 0 to 8 {
          %s0 = "op3"() : () -> i32
          %c2 = "affine.apply" (%x, %k2) {map: (d0, d1) -> (16*d0 + d1)} : (index, index) -> index
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

// UNROLL-BY-4-LABEL: func @unroll_unit_stride_no_cleanup() {
func @unroll_unit_stride_no_cleanup() {
  // UNROLL-BY-4: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 0 to 8 step 4 {
    // UNROLL-BY-4-NEXT: %0 = "addi32"([[L1]], [[L1]]) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %2 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %3 = "addi32"(%2, %2) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %8 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %9 = "addi32"(%8, %8) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 0 to 8 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
    // empty loop
    // UNROLL-BY-4: for %i2 = 0 to 8 {
    for %k = 0 to 8 {
    }
  }
  return
}

// UNROLL-BY-4-LABEL: func @unroll_unit_stride_cleanup() {
func @unroll_unit_stride_cleanup() {
  // UNROLL-BY-4: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 0 to 8 step 4 {
    // UNROLL-BY-4-NEXT:   %0 = "addi32"([[L1]], [[L1]]) : (index, index) -> i32
    // UNROLL-BY-4-NEXT:   %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT:   %2 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT:   %3 = "addi32"(%2, %2) : (index, index) -> i32
    // UNROLL-BY-4-NEXT:   %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT:   %5 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT:   %6 = "addi32"(%5, %5) : (index, index) -> i32
    // UNROLL-BY-4-NEXT:   %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT:   %8 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT:   %9 = "addi32"(%8, %8) : (index, index) -> i32
    // UNROLL-BY-4-NEXT:   %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: for [[L2:%i[0-9]+]] = 8 to 10 {
    // UNROLL-BY-4-NEXT:   %11 = "addi32"([[L2]], [[L2]]) : (index, index) -> i32
    // UNROLL-BY-4-NEXT:   %12 = "addi32"(%11, %11) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 0 to 10 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}

// UNROLL-BY-4-LABEL: func @unroll_non_unit_stride_cleanup() {
func @unroll_non_unit_stride_cleanup() {
  // UNROLL-BY-4: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-BY-4: for [[L1:%i[0-9]+]] = 2 to 42 step 20 {
    // UNROLL-BY-4-NEXT: %0 = "addi32"([[L1]], [[L1]]) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %1 = "addi32"(%0, %0) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %2 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %3 = "addi32"(%2, %2) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%6, %6) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: %8 = affine.apply #map{{[0-9]+}}([[L1]])
    // UNROLL-BY-4-NEXT: %9 = "addi32"(%8, %8) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %10 = "addi32"(%9, %9) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: for [[L2:%i[0-9]+]] = 42 to 48 step 5 {
    // UNROLL-BY-4-NEXT: %11 = "addi32"([[L2]], [[L2]]) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %12 = "addi32"(%11, %11) : (i32, i32) -> i32
    // UNROLL-BY-4-NEXT: }
    for %j = 2 to 48 step 5 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
      %y = "addi32"(%x, %x) : (i32, i32) -> i32
    }
  }
  return
}

// Both the unrolled loop and the cleanup loop are single iteration loops.
// UNROLL-BY-4-LABEL: func @loop_nest_single_iteration_after_unroll
func @loop_nest_single_iteration_after_unroll(%N: index) {
  // UNROLL-BY-4: %c0 = constant 0 : index
  // UNROLL-BY-4: %c4 = constant 4 : index
  // UNROLL-BY-4: for %i0 = 0 to %arg0 {
  for %i = 0 to %N {
    // UNROLL-BY-4: %0 = "addi32"(%c0, %c0) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %1 = affine.apply [[MAP0]](%c0)
    // UNROLL-BY-4-NEXT: %2 = "addi32"(%1, %1) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %3 = affine.apply [[MAP1]](%c0)
    // UNROLL-BY-4-NEXT: %4 = "addi32"(%3, %3) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %5 = affine.apply [[MAP2]](%c0)
    // UNROLL-BY-4-NEXT: %6 = "addi32"(%5, %5) : (index, index) -> i32
    // UNROLL-BY-4-NEXT: %7 = "addi32"(%c4, %c4) : (index, index) -> i32
    // UNROLL-BY-4-NOT: for
    for %j = 0 to 5 {
      %x = "addi32"(%j, %j) : (index, index) -> i32
    } // UNROLL-BY-4-NOT: }
  } // UNROLL-BY-4:  }
  return
}

// Test cases with loop bound operands.

// No cleanup will be generated here.
// UNROLL-BY-4-LABEL: func @loop_nest_operand1() {
func @loop_nest_operand1() {
// UNROLL-BY-4:      for %i0 = 0 to 100 step 2 {
// UNROLL-BY-4-NEXT:   for %i1 = 0 to #map{{[0-9]+}}(%i0) step 4
// UNROLL-BY-4-NEXT:      %0 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:      %1 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:      %2 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:      %3 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:   }
// UNROLL-BY-4-NEXT: }
// UNROLL-BY-4-NEXT: return
  for %i = 0 to 100 step 2 {
    for %j = 0 to (d0) -> (d0 - d0 mod 4) (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// No cleanup will be generated here.
// UNROLL-BY-4-LABEL: func @loop_nest_operand2() {
func @loop_nest_operand2() {
// UNROLL-BY-4:      for %i0 = 0 to 100 step 2 {
// UNROLL-BY-4-NEXT:   for %i1 = [[MAP11]](%i0) to #map{{[0-9]+}}(%i0) step 4 {
// UNROLL-BY-4-NEXT:     %0 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:     %1 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:     %2 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:     %3 = "foo"() : () -> i32
// UNROLL-BY-4-NEXT:   }
// UNROLL-BY-4-NEXT: }
// UNROLL-BY-4-NEXT: return
  for %i = 0 to 100 step 2 {
    for %j = (d0) -> (d0) (%i) to (d0) -> (5*d0 + 4) (%i) {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// Difference between loop bounds is constant, but not a multiple of unroll
// factor. The cleanup loop happens to be a single iteration one and is promoted.
// UNROLL-BY-4-LABEL: func @loop_nest_operand3() {
func @loop_nest_operand3() {
  // UNROLL-BY-4: for %i0 = 0 to 100 step 2 {
  for %i = 0 to 100 step 2 {
    // UNROLL-BY-4: for %i1 = [[MAP11]](%i0) to #map{{[0-9]+}}(%i0) step 4 {
    // UNROLL-BY-4-NEXT: %0 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %1 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %2 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %3 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: }
    // UNROLL-BY-4-NEXT: %4 = "foo"() : () -> i32
    for %j = (d0) -> (d0) (%i) to (d0) -> (d0 + 9) (%i) {
      %x = "foo"() : () -> i32
    }
  } // UNROLL-BY-4: }
  return
}

// UNROLL-BY-4-LABEL: func @loop_nest_symbolic_bound(%arg0: index) {
func @loop_nest_symbolic_bound(%N : index) {
  // UNROLL-BY-4: for %i0 = 0 to 100 {
  for %i = 0 to 100 {
    // UNROLL-BY-4: for %i1 = 0 to #map{{[0-9]+}}()[%arg0] step 4 {
    // UNROLL-BY-4: %0 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %1 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %2 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: %3 = "foo"() : () -> i32
    // UNROLL-BY-4-NEXT: }
    // A cleanup loop will be be generated here.
    // UNROLL-BY-4-NEXT: for %i2 = #map{{[0-9]+}}()[%arg0] to %arg0 {
    // UNROLL-BY-4-NEXT: %4 = "foo"() : () -> i32
    // UNROLL-BY-4_NEXT: }
    for %j = 0 to %N {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// UNROLL-BY-4-LABEL: func @loop_nest_symbolic_and_min_upper_bound
func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  for %i = %M to min ()[s0, s1] -> (s0, s1, 1024)()[%N, %K] {
    "foo"() : () -> ()
  }
  return
}
// CHECK-NEXT:  for %i0 = %arg0 to min [[MAP_TRIP_COUNT_MULTIPLE_FOUR]]()[%arg0, %arg1, %arg2] step 4 {
// CHECK-NEXT:    "foo"() : () -> ()
// CHECK-NEXT:    "foo"() : () -> ()
// CHECK-NEXT:    "foo"() : () -> ()
// CHECK-NEXT:    "foo"() : () -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  for %i1 = max [[MAP_TRIP_COUNT_MULTIPLE_FOUR]]()[%arg0, %arg1, %arg2] to min #map28()[%arg1, %arg2] {
// CHECK-NEXT:    "foo"() : () -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// The trip count here is a multiple of four, but this can be inferred only
// through composition. Check for no cleanup loop.
// UNROLL-BY-4-LABEL: func @loop_nest_non_trivial_multiple_unroll_factor
func @loop_nest_non_trivial_multiple_unroll_factor(%M : index, %N : index) {
  %T = affine.apply (d0) -> (4*d0 + 1)(%M)
  %K = affine.apply (d0) -> (d0 - 1) (%T)
  for %i = 0 to min (d0, d1) -> (4 * d0, d1, 1024)(%N, %K) {
    "foo"() : () -> ()
  }
  return
}
// UNROLL-BY-4: for %i0 = 0 to min
// UNROLL-BY-4-NOT: for
// UNROLL-BY-4: return

// UNROLL-BY-4-LABEL: func @loop_nest_non_trivial_multiple_unroll_factor_2
func @loop_nest_non_trivial_multiple_unroll_factor_2(%M : index, %N : index) {
  %K = affine.apply (d0) -> (4*d0) (%M)
  for %i = 0 to min ()[s0, s1] -> (4 * s0, s1, 1024)()[%N, %K] {
    "foo"() : () -> ()
  }
  // UNROLL-BY-4: for %i0 = 0 to min
  // UNROLL-BY-4-NEXT: "foo"
  // UNROLL-BY-4-NEXT: "foo"
  // UNROLL-BY-4-NEXT: "foo"
  // UNROLL-BY-4-NEXT: "foo"
  // UNROLL-BY-4-NOT for
  // UNROLL-BY-4: return
  return
}

// UNROLL-BY-1-LABEL: func @unroll_by_one_should_promote_single_iteration_loop()
func @unroll_by_one_should_promote_single_iteration_loop() {
  for %i = 0 to 1 {
    %x = "foo"(%i) : (index) -> i32
  }
  return
// UNROLL-BY-1-NEXT: %c0 = constant 0 : index
// UNROLL-BY-1-NEXT: %0 = "foo"(%c0) : (index) -> i32
// UNROLL-BY-1-NEXT: return
}
