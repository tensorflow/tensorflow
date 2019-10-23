// RUN: mlir-opt %s -test-memref-dependence-check  -split-input-file -verify-diagnostics | FileCheck %s

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: func @store_may_execute_before_load() {
func @store_may_execute_before_load() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c0 = constant 4 : index
  // There is a dependence from store 0 to load 1 at depth 1 because the
  // ancestor IfOp of the store, dominates the ancestor ForSmt of the load,
  // and thus the store "may" conditionally execute before the load.
  affine.if #set0(%c0) {
    affine.for %i0 = 0 to 10 {
      affine.store %cf7, %m[%i0] : memref<10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
    }
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  }
  return
}

// -----

// CHECK-LABEL: func @dependent_loops() {
func @dependent_loops() {
  %0 = alloc() : memref<10xf32>
  %cst = constant 7.000000e+00 : f32
  // There is a dependence from 0 to 1 at depth 1 (common surrounding loops 0)
  // because the first loop with the store dominates the second loop.
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %0[%i0] : memref<10xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %0[%i1] : memref<10xf32>
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @different_memrefs() {
func @different_memrefs() {
  %m.a = alloc() : memref<100xf32>
  %m.b = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1.0 : f32
  affine.store %c1, %m.a[%c0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  %v0 = affine.load %m.b[%c0] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_different_elements() {
func @store_load_different_elements() {
  %m = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c7 = constant 7.0 : f32
  affine.store %c7, %m[%c0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  %v0 = affine.load %m[%c1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @load_store_different_elements() {
func @load_store_different_elements() {
  %m = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c7 = constant 7.0 : f32
  %v0 = affine.load %m[%c1] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  affine.store %c7, %m[%c0] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_same_element() {
func @store_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  affine.store %c7, %m[%c11] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = affine.load %m[%c11] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @load_load_same_element() {
func @load_load_same_element() {
  %m = alloc() : memref<100xf32>
  %c11 = constant 11 : index
  %c7 = constant 7.0 : f32
  %v0 = affine.load %m[%c11] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  %v1 = affine.load %m[%c11] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_same_symbol(%arg0: index) {
func @store_load_same_symbol(%arg0: index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.store %c7, %m[%arg0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = affine.load %m[%arg0] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_different_symbols(%arg0: index, %arg1: index) {
func @store_load_different_symbols(%arg0: index, %arg1: index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.store %c7, %m[%arg0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  %v0 = affine.load %m[%arg1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_diff_element_affine_apply_const() {
func @store_load_diff_element_affine_apply_const() {
  %m = alloc() : memref<100xf32>
  %c1 = constant 1 : index
  %c8 = constant 8.0 : f32
  %a0 = affine.apply (d0) -> (d0) (%c1)
  affine.store %c8, %m[%a0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  %a1 = affine.apply (d0) -> (d0 + 1) (%c1)
  %v0 = affine.load %m[%a1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_same_element_affine_apply_const() {
func @store_load_same_element_affine_apply_const() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c9 = constant 9 : index
  %c11 = constant 11 : index
  %a0 = affine.apply (d0) -> (d0 + 1) (%c9)
  affine.store %c7, %m[%a0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  %a1 = affine.apply (d0) -> (d0 - 1) (%c11)
  %v0 = affine.load %m[%a1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_affine_apply_symbol(%arg0: index) {
func @store_load_affine_apply_symbol(%arg0: index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine.apply (d0) -> (d0) (%arg0)
  affine.store %c7, %m[%a0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
  %a1 = affine.apply (d0) -> (d0) (%arg0)
  %v0 = affine.load %m[%a1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_load_affine_apply_symbol_offset(%arg0: index) {
func @store_load_affine_apply_symbol_offset(%arg0: index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %a0 = affine.apply (d0) -> (d0) (%arg0)
  affine.store %c7, %m[%a0] : memref<100xf32>
  // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
  %a1 = affine.apply (d0) -> (d0 + 1) (%arg0)
  %v0 = affine.load %m[%a1] : memref<100xf32>
  // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
  // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
  return
}

// -----
// CHECK-LABEL: func @store_range_load_after_range() {
func @store_range_load_after_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  affine.for %i0 = 0 to 10 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine.apply (d0) -> (d0) (%c10)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @store_load_func_symbol(%arg0: index, %arg1: index) {
func @store_load_func_symbol(%arg0: index, %arg1: index) {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  affine.for %i0 = 0 to %arg1 {
    %a0 = affine.apply (d0) -> (d0) (%arg0)
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = [1, +inf]}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [1, +inf]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine.apply (d0) -> (d0) (%arg0)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, +inf]}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @store_range_load_last_in_range() {
func @store_range_load_last_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c10 = constant 10 : index
  affine.for %i0 = 0 to 10 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    // For dependence from 0 to 1, we do not have a loop carried dependence
    // because only the final write in the loop accesses the same element as the
    // load, so this dependence appears only at depth 2 (loop independent).
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine.apply (d0) -> (d0 - 1) (%c10)
    // For dependence from 1 to 0, we have write-after-read (WAR) dependences
    // for all loads in the loop to the store on the last iteration.
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, 9]}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @store_range_load_before_range() {
func @store_range_load_before_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  affine.for %i0 = 1 to 11 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine.apply (d0) -> (d0) (%c0)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @store_range_load_first_in_range() {
func @store_range_load_first_in_range() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  %c0 = constant 0 : index
  affine.for %i0 = 1 to 11 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    // Dependence from 0 to 1 at depth 1 is a range because all loads at
    // constant index zero are reads after first store at index zero during
    // first iteration of the loop.
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [1, 9]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = true}}
    %a1 = affine.apply (d0) -> (d0 + 1) (%c0)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @store_plus_3() {
func @store_plus_3() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 1 to 11 {
    %a0 = affine.apply (d0) -> (d0 + 3) (%i0)
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [3, 3]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine.apply (d0) -> (d0) (%i0)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @load_minus_2() {
func @load_minus_2() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 2 to 11 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    affine.store %c7, %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [2, 2]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine.apply (d0) -> (d0 - 2) (%i0)
    %v0 = affine.load %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @perfectly_nested_loops_loop_independent() {
func @perfectly_nested_loops_loop_independent() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to 11 {
    affine.for %i1 = 0 to 11 {
      // Dependence from access 0 to 1 is loop independent at depth = 3.
      %a00 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a01 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      affine.store %c7, %m[%a00, %a01] : memref<10x10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 3 = true}}
      %a10 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a11 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      %v0 = affine.load %m[%a10, %a11] : memref<10x10xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @perfectly_nested_loops_loop_carried_at_depth1() {
func @perfectly_nested_loops_loop_carried_at_depth1() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to 9 {
    affine.for %i1 = 0 to 9 {
      // Dependence from access 0 to 1 is loop carried at depth 1.
      %a00 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a01 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      affine.store %c7, %m[%a00, %a01] : memref<10x10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = [2, 2][0, 0]}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 3 = false}}
      %a10 = affine.apply (d0, d1) -> (d0 - 2) (%i0, %i1)
      %a11 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      %v0 = affine.load %m[%a10, %a11] : memref<10x10xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @perfectly_nested_loops_loop_carried_at_depth2() {
func @perfectly_nested_loops_loop_carried_at_depth2() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // Dependence from access 0 to 1 is loop carried at depth 2.
      %a00 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a01 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      affine.store %c7, %m[%a00, %a01] : memref<10x10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = [0, 0][3, 3]}}
      // expected-remark@above {{dependence from 0 to 1 at depth 3 = false}}
      %a10 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a11 = affine.apply (d0, d1) -> (d1 - 3) (%i0, %i1)
      %v0 = affine.load %m[%a10, %a11] : memref<10x10xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @one_common_loop() {
func @one_common_loop() {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  // There is a loop-independent dependence from access 0 to 1 at depth 2.
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %a00 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a01 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      affine.store %c7, %m[%a00, %a01] : memref<10x10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = true}}
    }
    affine.for %i2 = 0 to 9 {
      %a10 = affine.apply (d0, d1) -> (d0) (%i0, %i2)
      %a11 = affine.apply (d0, d1) -> (d1) (%i0, %i2)
      %v0 = affine.load %m[%a10, %a11] : memref<10x10xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @dependence_cycle() {
func @dependence_cycle() {
  %m.a = alloc() : memref<100xf32>
  %m.b = alloc() : memref<100xf32>

  // Dependences:
  // *) loop-independent dependence from access 1 to 2 at depth 2.
  // *) loop-carried dependence from access 3 to 0 at depth 1.
  affine.for %i0 = 0 to 9 {
    %a0 = affine.apply (d0) -> (d0) (%i0)
    %v0 = affine.load %m.a[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 2 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 2 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 3 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 3 at depth 2 = false}}
    %a1 = affine.apply (d0) -> (d0) (%i0)
    affine.store %v0, %m.b[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 2 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 2 at depth 2 = true}}
    // expected-remark@above {{dependence from 1 to 3 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 3 at depth 2 = false}}
    %a2 = affine.apply (d0) -> (d0) (%i0)
    %v1 = affine.load %m.b[%a2] : memref<100xf32>
    // expected-remark@above {{dependence from 2 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 2 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 2 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 2 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 2 to 2 at depth 1 = false}}
    // expected-remark@above {{dependence from 2 to 2 at depth 2 = false}}
    // expected-remark@above {{dependence from 2 to 3 at depth 1 = false}}
    // expected-remark@above {{dependence from 2 to 3 at depth 2 = false}}
    %a3 = affine.apply (d0) -> (d0 + 1) (%i0)
    affine.store %v1, %m.a[%a3] : memref<100xf32>
    // expected-remark@above {{dependence from 3 to 0 at depth 1 = [1, 1]}}
    // expected-remark@above {{dependence from 3 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 3 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 3 to 1 at depth 2 = false}}
    // expected-remark@above {{dependence from 3 to 2 at depth 1 = false}}
    // expected-remark@above {{dependence from 3 to 2 at depth 2 = false}}
    // expected-remark@above {{dependence from 3 to 3 at depth 1 = false}}
    // expected-remark@above {{dependence from 3 to 3 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @negative_and_positive_direction_vectors(%arg0: index, %arg1: index) {
func @negative_and_positive_direction_vectors(%arg0: index, %arg1: index) {
  %m = alloc() : memref<10x10xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to %arg0 {
    affine.for %i1 = 0 to %arg1 {
      %a00 = affine.apply (d0, d1) -> (d0 - 1) (%i0, %i1)
      %a01 = affine.apply (d0, d1) -> (d1 + 1) (%i0, %i1)
      %v0 = affine.load %m[%a00, %a01] : memref<10x10xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 3 = false}}
      %a10 = affine.apply (d0, d1) -> (d0) (%i0, %i1)
      %a11 = affine.apply (d0, d1) -> (d1) (%i0, %i1)
      affine.store %c7, %m[%a10, %a11] : memref<10x10xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, 1][-1, -1]}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @war_raw_waw_deps() {
func @war_raw_waw_deps() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %a0 = affine.apply (d0) -> (d0 + 1) (%i1)
      %v0 = affine.load %m[%a0] : memref<100xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = [1, 9][1, 1]}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = [0, 0][1, 1]}}
      // expected-remark@above {{dependence from 0 to 1 at depth 3 = false}}
      %a1 = affine.apply (d0) -> (d0) (%i1)
      affine.store %c7, %m[%a1] : memref<100xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, 9][-1, -1]}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = [1, 9][0, 0]}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----
// CHECK-LABEL: func @mod_deps() {
func @mod_deps() {
  %m = alloc() : memref<100xf32>
  %c7 = constant 7.0 : f32
  affine.for %i0 = 0 to 10 {
    %a0 = affine.apply (d0) -> (d0 mod 2) (%i0)
    // Results are conservative here since we currently don't have a way to
    // represent strided sets in FlatAffineConstraints.
    %v0 = affine.load %m[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [1, 9]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %a1 = affine.apply (d0) -> ( (d0 + 1) mod 2) (%i0)
    affine.store %c7, %m[%a1] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, 9]}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = [2, 9]}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----
// CHECK-LABEL: func @loop_nest_depth() {
func @loop_nest_depth() {
  %0 = alloc() : memref<100x100xf32>
  %c7 = constant 7.0 : f32

  affine.for %i0 = 0 to 128 {
    affine.for %i1 = 0 to 8 {
      affine.store %c7, %0[%i0, %i1] : memref<100x100xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
    }
  }
  affine.for %i2 = 0 to 8 {
    affine.for %i3 = 0 to 8 {
      affine.for %i4 = 0 to 8 {
        affine.for %i5 = 0 to 16 {
          %8 = affine.apply (d0, d1) -> (d0 * 16 + d1)(%i4, %i5)
          %9 = affine.load %0[%8, %i3] : memref<100x100xf32>
          // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
          // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
          // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
          // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
          // expected-remark@above {{dependence from 1 to 1 at depth 4 = false}}
          // expected-remark@above {{dependence from 1 to 1 at depth 5 = false}}
        }
      }
    }
  }
  return
}

// -----
// Test case to exercise sanity when flattening multiple expressions involving
// mod/div's successively.
// CHECK-LABEL: func @mod_div_3d() {
func @mod_div_3d() {
  %M = alloc() : memref<2x2x2xi32>
  %c0 = constant 0 : i32
  affine.for %i0 = 0 to 8 {
    affine.for %i1 = 0 to 8 {
      affine.for %i2 = 0 to 8 {
        %idx0 = affine.apply (d0, d1, d2) -> (d0 floordiv 4) (%i0, %i1, %i2)
        %idx1 = affine.apply (d0, d1, d2) -> (d1 mod 2) (%i0, %i1, %i2)
        %idx2 = affine.apply (d0, d1, d2) -> (d2 floordiv 4) (%i0, %i1, %i2)
        affine.store %c0, %M[%idx0, %idx1, %idx2] : memref<2 x 2 x 2 x i32>
        // expected-remark@above {{dependence from 0 to 0 at depth 1 = [1, 3][-7, 7][-3, 3]}}
        // expected-remark@above {{dependence from 0 to 0 at depth 2 = [0, 0][2, 7][-3, 3]}}
        // expected-remark@above {{dependence from 0 to 0 at depth 3 = [0, 0][0, 0][1, 3]}}
        // expected-remark@above {{dependence from 0 to 0 at depth 4 = false}}
      }
    }
  }
  return
}

// -----
// This test case arises in the context of a 6-d to 2-d reshape.
// CHECK-LABEL: func @delinearize_mod_floordiv
func @delinearize_mod_floordiv() {
  %c0 = constant 0 : index
  %val = constant 0 : i32
  %in = alloc() : memref<2x2x3x3x16x1xi32>
  %out = alloc() : memref<64x9xi32>

  affine.for %i0 = 0 to 2 {
    affine.for %i1 = 0 to 2 {
      affine.for %i2 = 0 to 3 {
        affine.for %i3 = 0 to 3 {
          affine.for %i4 = 0 to 16 {
            affine.for %i5 = 0 to 1 {
              affine.store %val, %in[%i0, %i1, %i2, %i3, %i4, %i5] : memref<2x2x3x3x16x1xi32>
// expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 4 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 5 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 6 = false}}
// expected-remark@above {{dependence from 0 to 0 at depth 7 = false}}
// expected-remark@above {{dependence from 0 to 1 at depth 1 = true}}
// expected-remark@above {{dependence from 0 to 2 at depth 1 = false}}
            }
          }
        }
      }
    }
  }

  affine.for %ii = 0 to 64 {
    affine.for %jj = 0 to 9 {
      %a0 = affine.apply (d0, d1) -> (d0 * (9 * 1024) + d1 * 128) (%ii, %jj)
      %a10 = affine.apply (d0) ->
        (d0 floordiv (2 * 3 * 3 * 128 * 128)) (%a0)
      %a11 = affine.apply (d0) ->
        ((d0 mod 294912) floordiv (3 * 3 * 128 * 128)) (%a0)
      %a12 = affine.apply (d0) ->
        ((((d0 mod 294912) mod 147456) floordiv 1152) floordiv 8) (%a0)
      %a13 = affine.apply (d0) ->
        ((((d0 mod 294912) mod 147456) mod 1152) floordiv 384) (%a0)
      %a14 = affine.apply (d0) ->
        (((((d0 mod 294912) mod 147456) mod 1152) mod 384) floordiv 128) (%a0)
      %a15 = affine.apply (d0) ->
        ((((((d0 mod 294912) mod 147456) mod 1152) mod 384) mod 128)
          floordiv 128) (%a0)
      %v0 = affine.load %in[%a10, %a11, %a13, %a14, %a12, %a15] : memref<2x2x3x3x16x1xi32>
// expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
// expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
// expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
// expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
// expected-remark@above {{dependence from 1 to 2 at depth 1 = false}}
// expected-remark@above {{dependence from 1 to 2 at depth 2 = false}}
// expected-remark@above {{dependence from 1 to 2 at depth 3 = false}}
// TODO(andydavis): the dep tester shouldn't be printing out these messages
// below; they are redundant.
      affine.store %v0, %out[%ii, %jj] : memref<64x9xi32>
// expected-remark@above {{dependence from 2 to 0 at depth 1 = false}}
// expected-remark@above {{dependence from 2 to 1 at depth 1 = false}}
// expected-remark@above {{dependence from 2 to 1 at depth 2 = false}}
// expected-remark@above {{dependence from 2 to 1 at depth 3 = false}}
// expected-remark@above {{dependence from 2 to 2 at depth 1 = false}}
// expected-remark@above {{dependence from 2 to 2 at depth 2 = false}}
// expected-remark@above {{dependence from 2 to 2 at depth 3 = false}}
    }
  }
  return
}

// TODO(bondhugula): add more test cases involving mod's/div's.

// -----

// Load and store ops access the same elements in strided loop.
// CHECK-LABEL: func @strided_loop_with_dependence_at_depth2
func @strided_loop_with_dependence_at_depth2() {
  %0 = alloc() : memref<10xf32>
  %cf0 = constant 0.0 : f32
  affine.for %i0 = 0 to 8 step 2 {
    affine.store %cf0, %0[%i0] : memref<10xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = true}}
    %v0 = affine.load %0[%i0] : memref<10xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----

// Load and store ops access alternating memref elements: no dependence.
// CHECK-LABEL: func @strided_loop_with_no_dependence
func @strided_loop_with_no_dependence() {
  %0 = alloc() : memref<10xf32>
  %cf0 = constant 0.0 : f32
  affine.for %i0 = 0 to 8 step 2 {
    %a0 = affine.apply (d0) -> (d0 + 1)(%i0)
    affine.store %cf0, %0[%a0] : memref<10xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %v0 = affine.load %0[%i0] : memref<10xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----

// Affine.Store op accesses memref elements at offset causing loop-carried dependence.
// CHECK-LABEL: func @strided_loop_with_loop_carried_dependence_at_depth1
func @strided_loop_with_loop_carried_dependence_at_depth1() {
  %0 = alloc() : memref<10xf32>
  %cf0 = constant 0.0 : f32
  affine.for %i0 = 0 to 8 step 2 {
    %a0 = affine.apply (d0) -> (d0 + 4)(%i0)
    affine.store %cf0, %0[%a0] : memref<10xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = [4, 4]}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    %v0 = affine.load %0[%i0] : memref<10xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}

// -----

// Test that the loop carried dependence from load to store on '%i0' is
// properly computed when the load and store are at different loop depths.
// CHECK-LABEL: func @test_dep_store_depth1_load_depth2
func @test_dep_store_depth1_load_depth2() {
  %0 = alloc() : memref<100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    %a0 = affine.apply (d0) -> (d0 - 1)(%i0)
    affine.store %cst, %0[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    affine.for %i1 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 1)(%i0) {
      %1 = affine.load %0[%i1] : memref<100xf32>
      // expected-remark@above {{dependence from 1 to 0 at depth 1 = [1, 1]}}
      // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
      // expected-remark@above {{dependence from 1 to 1 at depth 3 = false}}
    }
  }
  return
}

// -----

// Test that the loop carried dependence from store to load on '%i0' is
// properly computed when the load and store are at different loop depths.
// CHECK-LABEL: func @test_dep_store_depth2_load_depth1
func @test_dep_store_depth2_load_depth1() {
  %0 = alloc() : memref<100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 1)(%i0) {
      affine.store %cst, %0[%i1] : memref<100xf32>
      // expected-remark@above {{dependence from 0 to 0 at depth 1 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 2 = false}}
      // expected-remark@above {{dependence from 0 to 0 at depth 3 = false}}
      // expected-remark@above {{dependence from 0 to 1 at depth 1 = [2, 2]}}
      // expected-remark@above {{dependence from 0 to 1 at depth 2 = false}}
    }
    %a0 = affine.apply (d0) -> (d0 - 2)(%i0)
    %1 = affine.load %0[%a0] : memref<100xf32>
    // expected-remark@above {{dependence from 1 to 0 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 0 at depth 2 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 1 = false}}
    // expected-remark@above {{dependence from 1 to 1 at depth 2 = false}}
  }
  return
}
