// RUN: mlir-opt %s -test-loop-fusion -test-loop-fusion-slice-computation -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func @slice_depth1_loop_nest() {
func @slice_depth1_loop_nest() {
  %0 = alloc() : memref<100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 1) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] )}}
    affine.store %cst, %0[%i0] : memref<100xf32>
  }
  affine.for %i1 = 0 to 5 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] )}}
    %1 = affine.load %0[%i1] : memref<100xf32>
  }
  return
}

// -----

// Loop %i0 writes to locations [2, 17] and loop %i0 reads from locations [3, 6]
// Slice loop bounds should be adjusted such that the load/store are for the
// same location.
// CHECK-LABEL: func @slice_depth1_loop_nest_with_offsets() {
func @slice_depth1_loop_nest_with_offsets() {
  %0 = alloc() : memref<100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 2) loop bounds: [(d0) -> (d0 + 3), (d0) -> (d0 + 4)] )}}
    %a0 = affine.apply (d0) -> (d0 + 2)(%i0)
    affine.store %cst, %0[%a0] : memref<100xf32>
  }
  affine.for %i1 = 4 to 8 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0) -> (d0 - 3), (d0) -> (d0 - 2)] )}}
    %a1 = affine.apply (d0) -> (d0 - 1)(%i1)
    %1 = affine.load %0[%a1] : memref<100xf32>
  }
  return
}

// -----

// Slices at loop depth 1 should only slice the loop bounds of the first loop.
// Slices at loop detph 2 should slice loop bounds of both loops.
// CHECK-LABEL: func @slice_depth2_loop_nest() {
func @slice_depth2_loop_nest() {
  %0 = alloc() : memref<100x100xf32>
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 1) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] [(d0) -> (0), (d0) -> (8)] )}}
    // expected-remark@-2 {{slice ( src loop: 1, dst loop: 0, depth: 2 : insert point: (2, 1) loop bounds: [(d0, d1) -> (d0), (d0, d1) -> (d0 + 1)] [(d0, d1) -> (d1), (d0, d1) -> (d1 + 1)] )}}
    affine.for %i1 = 0 to 16 {
      affine.store %cst, %0[%i0, %i1] : memref<100x100xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] [(d0) -> (0), (d0) -> (8)] )}}
    // expected-remark@-2 {{slice ( src loop: 0, dst loop: 1, depth: 2 : insert point: (2, 0) loop bounds: [(d0, d1) -> (d0), (d0, d1) -> (d0 + 1)] [(d0, d1) -> (d1), (d0, d1) -> (d1 + 1)] )}}
    affine.for %i3 = 0 to 8 {
      %1 = affine.load %0[%i2, %i3] : memref<100x100xf32>
    }
  }
  return
}

// -----

// The load at depth 1 in loop nest %i2 prevents slicing loop nest %i0 at depths
// greater than 1. However, loop nest %i2 can be sliced into loop nest %i0 at
// depths 1 and 2 because the dependent store in loop nest %i0 is at depth 2.
// CHECK-LABEL: func @slice_depth2_loop_nest_two_loads() {
func @slice_depth2_loop_nest_two_loads() {
  %0 = alloc() : memref<100x100xf32>
  %c0 = constant 0 : index
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 1) loop bounds: [(d0)[s0] -> (d0), (d0)[s0] -> (d0 + 1)] [(d0)[s0] -> (0), (d0)[s0] -> (8)] )}}
    // expected-remark@-2 {{slice ( src loop: 1, dst loop: 0, depth: 2 : insert point: (2, 1) loop bounds: [(d0, d1)[s0] -> (d0), (d0, d1)[s0] -> (d0 + 1)] [(d0, d1)[s0] -> (0), (d0, d1)[s0] -> (8)] )}}
    affine.for %i1 = 0 to 16 {
      affine.store %cst, %0[%i0, %i1] : memref<100x100xf32>
    }
  }
  affine.for %i2 = 0 to 10 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0)[s0] -> (d0), (d0)[s0] -> (d0 + 1)] [(d0)[s0] -> (0), (d0)[s0] -> (8)] )}}
    affine.for %i3 = 0 to 8 {
      %1 = affine.load %0[%i2, %i3] : memref<100x100xf32>
    }
    %2 = affine.load %0[%i2, %c0] : memref<100x100xf32>
  }
  return
}

// -----

// The store at depth 1 in loop nest %i0 prevents slicing loop nest %i2 at
// depths greater than 1 into loop nest %i0. However, loop nest %i0 can be
// sliced into loop nest %i2 at depths 1 and 2 because the dependent load in
// loop nest %i2 is at depth 2.
// CHECK-LABEL: func @slice_depth2_loop_nest_two_stores() {
func @slice_depth2_loop_nest_two_stores() {
  %0 = alloc() : memref<100x100xf32>
  %c0 = constant 0 : index
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 2) loop bounds: [(d0)[s0] -> (d0), (d0)[s0] -> (d0 + 1)] [(d0)[s0] -> (0), (d0)[s0] -> (8)] )}}
    affine.for %i1 = 0 to 16 {
      affine.store %cst, %0[%i0, %i1] : memref<100x100xf32>
    }
    affine.store %cst, %0[%i0, %c0] : memref<100x100xf32>
  }
  affine.for %i2 = 0 to 10 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0)[s0] -> (d0), (d0)[s0] -> (d0 + 1)] [(d0)[s0] -> (0), (d0)[s0] -> (16)] )}}
    // expected-remark@-2 {{slice ( src loop: 0, dst loop: 1, depth: 2 : insert point: (2, 0) loop bounds: [(d0, d1)[s0] -> (d0), (d0, d1)[s0] -> (d0 + 1)] [(d0, d1)[s0] -> (0), (d0, d1)[s0] -> (16)] )}}
    affine.for %i3 = 0 to 8 {
      %1 = affine.load %0[%i2, %i3] : memref<100x100xf32>
    }
  }
  return
}

// -----

// Test loop nest which has a smaller outer trip count than its inner loop.
// CHECK-LABEL: func @slice_loop_nest_with_smaller_outer_trip_count() {
func @slice_loop_nest_with_smaller_outer_trip_count() {
  %0 = alloc() : memref<100x100xf32>
  %c0 = constant 0 : index
  %cst = constant 7.000000e+00 : f32
  affine.for %i0 = 0 to 16 {
    // expected-remark@-1 {{slice ( src loop: 1, dst loop: 0, depth: 1 : insert point: (1, 1) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] [(d0) -> (0), (d0) -> (10)] )}}
    // expected-remark@-2 {{slice ( src loop: 1, dst loop: 0, depth: 2 : insert point: (2, 1) loop bounds: [(d0, d1) -> (d0), (d0, d1) -> (d0 + 1)] [(d0, d1) -> (d1), (d0, d1) -> (d1 + 1)] )}}
    affine.for %i1 = 0 to 16 {
      affine.store %cst, %0[%i0, %i1] : memref<100x100xf32>
    }
  }
  affine.for %i2 = 0 to 8 {
    // expected-remark@-1 {{slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0) loop bounds: [(d0) -> (d0), (d0) -> (d0 + 1)] [(d0) -> (0), (d0) -> (10)] )}}
    // expected-remark@-2 {{slice ( src loop: 0, dst loop: 1, depth: 2 : insert point: (2, 0) loop bounds: [(d0, d1) -> (d0), (d0, d1) -> (d0 + 1)] [(d0, d1) -> (d1), (d0, d1) -> (d1 + 1)] )}}
    affine.for %i3 = 0 to 10 {
      %1 = affine.load %0[%i2, %i3] : memref<100x100xf32>
    }
  }
  return
}