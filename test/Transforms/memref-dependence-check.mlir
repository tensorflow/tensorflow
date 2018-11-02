// RUN: mlir-opt %s -memref-dependence-check  -split-input-file -verify | FileCheck %s

// TODO(andydavis) Add test cases for self-edges and a dependence cycle.

// -----
// CHECK-LABEL: mlfunc @different_memrefs() {
mlfunc @different_memrefs() {
  %m.a = alloc() : memref<100xf32>
  %m.b = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1.0 : f32
  store %c1, %m.a[%c0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  %v0 = load %m.b[%c0] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_different_elements() {
mlfunc @store_load_different_elements() {
  %m = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c7 = constant 7.0 : f32
  store %c7, %m[%c0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  %v0 = load %m[%c1] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @load_store_different_elements() {
mlfunc @load_store_different_elements() {
  %m = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c7 = constant 7.0 : f32
  %v0 = load %m[%c1] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  store %c7, %m[%c0] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_same_element() {
mlfunc @store_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  store %c7, %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %v0 = load %m[%c11] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @load_store_same_element() {
mlfunc @load_store_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  %v0 = load %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  store %c7, %m[%c11] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @load_load_same_element() {
mlfunc @load_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  %v0 = load %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  %v1 = load %m[%c11] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_same_symbol(%arg0 : index) {
mlfunc @store_load_same_symbol(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  store %c7, %m[%arg0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %v0 = load %m[%arg0] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_different_symbols(%arg0 : index, %arg1 : index) {
mlfunc @store_load_different_symbols(%arg0 : index, %arg1 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  store %c7, %m[%arg0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %v0 = load %m[%arg1] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_diff_element_affine_apply_const() {
mlfunc @store_load_diff_element_affine_apply_const() {
  %m = alloc() : memref<100xf32>
  %c1 = constant 1 : index
  %c7 = constant 7.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%c1)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  %a1 = affine_apply (d0) -> (d0 + 1) (%c1)
  %v0 = load %m[%a1] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_same_element_affine_apply_const() {
mlfunc @store_load_same_element_affine_apply_const() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c9 = constant 9 : index
  %c11 = constant 11 : index  
  %a0 = affine_apply (d0) -> (d0 + 1) (%c9)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %a1 = affine_apply (d0) -> (d0 - 1) (%c11)
  %v0 = load %m[%a1] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_affine_apply_symbol(%arg0 : index) {
mlfunc @store_load_affine_apply_symbol(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%arg0)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %a1 = affine_apply (d0) -> (d0) (%arg0)
  %v0 = load %m[%a1] : memref<100xf32>
  return
}

// -----
// Note: has single equality x - y - 1 = 0, which has solns for (1, 0) (0, -1)
// CHECK-LABEL: mlfunc @store_load_affine_apply_symbol_offset(%arg0 : index) {
mlfunc @store_load_affine_apply_symbol_offset(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%arg0)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  %a1 = affine_apply (d0) -> (d0 + 1) (%arg0)
  %v0 = load %m[%a1] : memref<100xf32>
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_after_range() {
mlfunc @store_range_load_after_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  for %i0 = 0 to 9 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
    %a1 = affine_apply (d0) -> (d0) (%c10)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_last_in_range() {
mlfunc @store_range_load_last_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  for %i0 = 0 to 9 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
    %a1 = affine_apply (d0) -> (d0 - 1) (%c10)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_before_range() {
mlfunc @store_range_load_before_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  for %i0 = 1 to 10 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
    %a1 = affine_apply (d0) -> (d0) (%c0)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_first_in_range() {
mlfunc @store_range_load_first_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  for %i0 = 1 to 10 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
    %a1 = affine_apply (d0) -> (d0 + 1) (%c0)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_diff_ranges_diff_1d_loop_nests() {
mlfunc @store_load_diff_ranges_diff_1d_loop_nests() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
  }
  for %i1 = 5 to 10 {
    %a1 = affine_apply (d0) -> (d0) (%i1)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_overlapping_ranges_diff_1d_loop_nests() {
mlfunc @store_load_overlapping_ranges_diff_1d_loop_nests() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
  }
  for %i1 = 5 to 10 {
    %a1 = affine_apply (d0) -> (d0 - 1) (%i1)
    %v0 = load %m[%a1] : memref<100xf32>
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_diff_inner_ranges_diff_2d_loop_nests() {
mlfunc @store_load_diff_inner_ranges_diff_2d_loop_nests() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    for %i1 = 0 to 4 {
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
       store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
    }
  }
  for %i2 = 0 to 4 {
    for %i3 = 5 to 6 {
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i2, %i3)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_overlapping_inner_ranges_diff_2d_loop_nests() {
mlfunc @store_load_overlapping_inner_ranges_diff_2d_loop_nests() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    for %i1 = 0 to 4 {
      %a0 = affine_apply (d0, d1) -> (d0, d1 + 1) (%i0, %i1)
       store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
    }
  }
  for %i2 = 0 to 4 {
    for %i3 = 5 to 6 {
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i2, %i3)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_diff_outer_ranges_diff_2d_loop_nests() {
mlfunc @store_load_diff_outer_ranges_diff_2d_loop_nests() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    for %i1 = 0 to 4 {
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
       store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
    }
  }
  for %i2 = 5 to 7 {
    for %i3 = 0 to 4 {
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i2, %i3)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_overlapping_outer_ranges_diff_2d_loop_nests() {
mlfunc @store_load_overlapping_outer_ranges_diff_2d_loop_nests() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 4 {
    for %i1 = 0 to 4 {
      %a0 = affine_apply (d0, d1) -> (d0 + 1, d1) (%i0, %i1)
       store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from memref access 0 to access 1 = true}}
    }
  }
  for %i2 = 5 to 7 {
    for %i3 = 0 to 4 {
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i2, %i3)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }
  return
}
