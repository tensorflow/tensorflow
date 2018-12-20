// RUN: mlir-opt %s -memref-dependence-check  -split-input-file -verify | FileCheck %s

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: mlfunc @store_may_execute_before_load() {
mlfunc @store_may_execute_before_load() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c0 = constant 4 : index
  // There is a dependence from store 0 to load 1 at depth 1 because the
  // ancestor IfStmt of the store, dominates the ancestor ForSmt of the load,
  // and thus the store "may" conditionally execute before the load.
  if #set0(%c0) {
    for %i0 = 0 to 10 {
      store %cf7, %m[%i0] : memref<10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 1 at depth 1 = true}}
    }
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
    // expected-note@-1 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 1 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 0 at depth 1 = false}}
  }
  return
}

// -----

// CHECK-LABEL: mlfunc @dependent_loops() {
mlfunc @dependent_loops() {
  %0 = alloc() : memref<10xf32>
  %cst = constant 7.000000e+00 : f32
  // There is a dependence from 0 to 1 at depth 1 (common surrounding loops 0)
  // because the first loop with the store dominates the second loop.
  for %i0 = 0 to 10 {
    store %cst, %0[%i0] : memref<10xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = true}}
  }
  for %i1 = 0 to 10 {
    %1 = load %0[%i1] : memref<10xf32>
    // expected-note@-1 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 1 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 0 at depth 1 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @different_memrefs() {
mlfunc @different_memrefs() {
  %m.a = alloc() : memref<100xf32>
  %m.b = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1.0 : f32
  store %c1, %m.a[%c0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  %v0 = load %m.b[%c0] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
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
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  %v0 = load %m[%c1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
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
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  store %c7, %m[%c0] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_same_element() {
mlfunc @store_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  store %c7, %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = load %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @load_load_same_element() {
mlfunc @load_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  %v0 = load %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  %v1 = load %m[%c11] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_same_symbol(%arg0 : index) {
mlfunc @store_load_same_symbol(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  store %c7, %m[%arg0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = load %m[%arg0] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_different_symbols(%arg0 : index, %arg1 : index) {
mlfunc @store_load_different_symbols(%arg0 : index, %arg1 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  store %c7, %m[%arg0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = load %m[%arg1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_diff_element_affine_apply_const() {
mlfunc @store_load_diff_element_affine_apply_const() {
  %m = alloc() : memref<100xf32>
  %c1 = constant 1 : index
  %c8 = constant 8.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%c1)
  store %c8, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  %a1 = affine_apply (d0) -> (d0 + 1) (%c1)
  %v0 = load %m[%a1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
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
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = true}} 
  %a1 = affine_apply (d0) -> (d0 - 1) (%c11)
  %v0 = load %m[%a1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_affine_apply_symbol(%arg0 : index) {
mlfunc @store_load_affine_apply_symbol(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%arg0)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = true}}
  %a1 = affine_apply (d0) -> (d0) (%arg0)
  %v0 = load %m[%a1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_affine_apply_symbol_offset(%arg0 : index) {
mlfunc @store_load_affine_apply_symbol_offset(%arg0 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine_apply (d0) -> (d0) (%arg0)
  store %c7, %m[%a0] : memref<100xf32>
  // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 0 to 1 at depth 1 = false}}
  %a1 = affine_apply (d0) -> (d0 + 1) (%arg0)
  %v0 = load %m[%a1] : memref<100xf32>
  // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
  // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_after_range() {
mlfunc @store_range_load_after_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  for %i0 = 0 to 10 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine_apply (d0) -> (d0) (%c10)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_load_func_symbol(%arg0 : index, %arg1 : index) {
mlfunc @store_load_func_symbol(%arg0 : index, %arg1 : index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  for %i0 = 0 to %arg1 {
    %a0 = affine_apply (d0) -> (d0) (%arg0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = [1, +inf]}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = [1, +inf]}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine_apply (d0) -> (d0) (%arg0)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = [1, +inf]}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_last_in_range() {
mlfunc @store_range_load_last_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  for %i0 = 0 to 10 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    // For dependence from 0 to 1, we do not have a loop carried dependence
    // because only the final write in the loop accesses the same element as the
    // load, so this dependence appears only at depth 2 (loop independent).
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine_apply (d0) -> (d0 - 1) (%c10)
    // For dependence from 1 to 0, we have write-after-read (WAR) dependences
    // for all loads in the loop to the store on the last iteration.
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = [1, 9]}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_before_range() {
mlfunc @store_range_load_before_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  for %i0 = 1 to 11 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine_apply (d0) -> (d0) (%c0)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_range_load_first_in_range() {
mlfunc @store_range_load_first_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  for %i0 = 1 to 11 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    // Dependence from 0 to 1 at depth 1 is a range because all loads at
    // constant index zero are reads after first store at index zero during
    // first iteration of the loop.
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = [1, 9]}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine_apply (d0) -> (d0 + 1) (%c0)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @store_plus_3() {
mlfunc @store_plus_3() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 1 to 11 {
    %a0 = affine_apply (d0) -> (d0 + 3) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = [3, 3]}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine_apply (d0) -> (d0) (%i0)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @load_minus_2() {
mlfunc @load_minus_2() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 2 to 11 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    store %c7, %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = [2, 2]}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine_apply (d0) -> (d0 - 2) (%i0)
    %v0 = load %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @perfectly_nested_loops_loop_independent() {
mlfunc @perfectly_nested_loops_loop_independent() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 11 {
    for %i1 = 0 to 11 {
      // Dependence from access 0 to 1 is loop independent at depth = 3.
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 0 to 1 at depth 3 = true}}
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @perfectly_nested_loops_loop_carried_at_depth1() {
mlfunc @perfectly_nested_loops_loop_carried_at_depth1() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 9 {
    for %i1 = 0 to 9 {
      // Dependence from access 0 to 1 is loop carried at depth 1.
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = [2, 2][0, 0]}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 0 to 1 at depth 3 = false}}
      %a1 = affine_apply (d0, d1) -> (d0 - 2, d1) (%i0, %i1)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @perfectly_nested_loops_loop_carried_at_depth2() {
mlfunc @perfectly_nested_loops_loop_carried_at_depth2() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      // Dependence from access 0 to 1 is loop carried at depth 2.
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = [0, 0][3, 3]}}
      // expected-note@-6 {{dependence from 0 to 1 at depth 3 = false}}
      %a1 = affine_apply (d0, d1) -> (d0, d1 - 3) (%i0, %i1)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @one_common_loop() {
mlfunc @one_common_loop() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  // There is a loop-independent dependence from access 0 to 1 at depth 2.
  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %a0 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      store %c7, %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = true}}
    }
    for %i2 = 0 to 9 {
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i2)
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @dependence_cycle() {
mlfunc @dependence_cycle() {
  %m.a = alloc() : memref<100xf32>
  %m.b = alloc() : memref<100xf32>

  // Dependences:
  // *) loop-independent dependence from access 1 to 2 at depth 2.
  // *) loop-carried dependence from access 3 to 0 at depth 1.
  for %i0 = 0 to 9 {
    %a0 = affine_apply (d0) -> (d0) (%i0)
    %v0 = load %m.a[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    // expected-note@-5 {{dependence from 0 to 2 at depth 1 = false}}
    // expected-note@-6 {{dependence from 0 to 2 at depth 2 = false}}
    // expected-note@-7 {{dependence from 0 to 3 at depth 1 = false}}
    // expected-note@-8 {{dependence from 0 to 3 at depth 2 = false}}
    %a1 = affine_apply (d0) -> (d0) (%i0)
    store %v0, %m.b[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
    // expected-note@-5 {{dependence from 1 to 2 at depth 1 = false}}
    // expected-note@-6 {{dependence from 1 to 2 at depth 2 = true}}
    // expected-note@-7 {{dependence from 1 to 3 at depth 1 = false}}
    // expected-note@-8 {{dependence from 1 to 3 at depth 2 = false}}
    %a2 = affine_apply (d0) -> (d0) (%i0)
    %v1 = load %m.b[%a2] : memref<100xf32>
    // expected-note@-1 {{dependence from 2 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 2 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 2 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 2 to 1 at depth 2 = false}}
    // expected-note@-5 {{dependence from 2 to 2 at depth 1 = false}}
    // expected-note@-6 {{dependence from 2 to 2 at depth 2 = false}}
    // expected-note@-7 {{dependence from 2 to 3 at depth 1 = false}}
    // expected-note@-8 {{dependence from 2 to 3 at depth 2 = false}}
    %a3 = affine_apply (d0) -> (d0 + 1) (%i0)
    store %v1, %m.a[%a3] : memref<100xf32>
    // expected-note@-1 {{dependence from 3 to 0 at depth 1 = [1, 1]}}
    // expected-note@-2 {{dependence from 3 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 3 to 1 at depth 1 = false}}
    // expected-note@-4 {{dependence from 3 to 1 at depth 2 = false}}
    // expected-note@-5 {{dependence from 3 to 2 at depth 1 = false}}
    // expected-note@-6 {{dependence from 3 to 2 at depth 2 = false}}
    // expected-note@-7 {{dependence from 3 to 3 at depth 1 = false}}
    // expected-note@-8 {{dependence from 3 to 3 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @negative_and_positive_direction_vectors(%arg0 : index, %arg1 : index) {
mlfunc @negative_and_positive_direction_vectors(%arg0 : index, %arg1 : index) {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to %arg0 {
    for %i1 = 0 to %arg1 {
      %a0 = affine_apply (d0, d1) -> (d0 - 1, d1 + 1) (%i0, %i1)
      %v0 = load %m[%a0#0, %a0#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 0 to 1 at depth 3 = false}}
      %a1 = affine_apply (d0, d1) -> (d0, d1) (%i0, %i1)
      store %c7, %m[%a1#0, %a1#1] : memref<10x10xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = [1, 1][-1, -1]}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 1 = false}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @war_raw_waw_deps() {
mlfunc @war_raw_waw_deps() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %a0 = affine_apply (d0) -> (d0 + 1) (%i1)
      %v0 = load %m[%a0] : memref<100xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = [1, 9][1, 1]}}
      // expected-note@-5 {{dependence from 0 to 1 at depth 2 = [0, 0][1, 1]}}
      // expected-note@-6 {{dependence from 0 to 1 at depth 3 = false}}
      %a1 = affine_apply (d0) -> (d0) (%i1)
      store %c7, %m[%a1] : memref<100xf32>
      // expected-note@-1 {{dependence from 1 to 0 at depth 1 = [1, 9][-1, -1]}}
      // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 1 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 1 to 1 at depth 1 = [1, 9][0, 0]}}
      // expected-note@-5 {{dependence from 1 to 1 at depth 2 = false}}
      // expected-note@-6 {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @mod_deps() {
mlfunc @mod_deps() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  for %i0 = 0 to 10 {
    %a0 = affine_apply (d0) -> (d0 mod 2) (%i0)
    // Results are conservative here since constraint information after
    // flattening isn't being completely added. Will be done in the next CL.
    // The third and the fifth dependence below shouldn't have existed.
    %v0 = load %m[%a0] : memref<100xf32>
    // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
    // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 0 to 1 at depth 1 = [1, 9]}}
    // expected-note@-4 {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine_apply (d0) -> ( (d0 + 1) mod 2) (%i0)
    store %c7, %m[%a1] : memref<100xf32>
    // expected-note@-1 {{dependence from 1 to 0 at depth 1 = [1, 9]}}
    // expected-note@-2 {{dependence from 1 to 0 at depth 2 = false}}
    // expected-note@-3 {{dependence from 1 to 1 at depth 1 = [2, 9]}}
    // expected-note@-4 {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: mlfunc @loop_nest_depth() {
mlfunc @loop_nest_depth() {
  %0 = alloc() : memref<100x100xf32>
  %c7 = constant 7.0 : f32

  for %i0 = 0 to 128 {
    for %i1 = 0 to 8 {
      store %c7, %0[%i0, %i1] : memref<100x100xf32>
      // expected-note@-1 {{dependence from 0 to 0 at depth 1 = false}}
      // expected-note@-2 {{dependence from 0 to 0 at depth 2 = false}}
      // expected-note@-3 {{dependence from 0 to 0 at depth 3 = false}}
      // expected-note@-4 {{dependence from 0 to 1 at depth 1 = true}}
    }
  }
  for %i2 = 0 to 8 {
    for %i3 = 0 to 8 {
      for %i4 = 0 to 8 {
        for %i5 = 0 to 16 {
          %8 = affine_apply (d0, d1) -> (d0 * 16 + d1)(%i4, %i5)
          %9 = load %0[%8, %i3] : memref<100x100xf32>
          // expected-note@-1 {{dependence from 1 to 0 at depth 1 = false}}
          // expected-note@-2 {{dependence from 1 to 1 at depth 1 = false}}
          // expected-note@-3 {{dependence from 1 to 1 at depth 2 = false}}
          // expected-note@-4 {{dependence from 1 to 1 at depth 3 = false}}
          // expected-note@-5 {{dependence from 1 to 1 at depth 4 = false}}
          // expected-note@-6 {{dependence from 1 to 1 at depth 5 = false}}
        }
      }
    }
  }
  return
}
