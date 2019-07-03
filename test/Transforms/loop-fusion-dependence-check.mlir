// RUN: mlir-opt %s -test-loop-fusion -test-loop-fusion-dependence-check -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func @cannot_fuse_would_create_cycle() {
func @cannot_fuse_would_create_cycle() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // 1) loop0 -> loop1 on memref '%a'
  // 2) loop0 -> loop2 on memref '%b'
  // 3) loop1 -> loop2 on memref '%c'

  // Fusing loop nest '%i0' and loop nest '%i2' would create a cycle.
  affine.for %i0 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 2 at depth 0}}
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %cf7, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %a[%i1] : memref<10xf32>
    %v1 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 2 into loop nest 0 at depth 0}}
    %v2 = affine.load %b[%i2] : memref<10xf32>
    affine.store %cf7, %c[%i2] : memref<10xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @can_fuse_rar_dependence() {
func @can_fuse_rar_dependence() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // Make dependence from 0 to 1 on '%a' read-after-read.
  // 1) loop0 -> loop1 on memref '%a'
  // 2) loop0 -> loop2 on memref '%b'
  // 3) loop1 -> loop2 on memref '%c'

  // Should fuse: no fusion preventing remarks should be emitted for this test.
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %cf7, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v1 = affine.load %a[%i1] : memref<10xf32>
    %v2 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v3 = affine.load %b[%i2] : memref<10xf32>
    affine.store %cf7, %c[%i2] : memref<10xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @can_fuse_different_memrefs() {
func @can_fuse_different_memrefs() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>
  %d = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // Make dependence from 0 to 1 on unrelated memref '%d'.
  // 1) loop0 -> loop1 on memref '%a'
  // 2) loop0 -> loop2 on memref '%b'
  // 3) loop1 -> loop2 on memref '%c'

  // Should fuse: no fusion preventing remarks should be emitted for this test.
  affine.for %i0 = 0 to 10 {
    %v0 = affine.load %a[%i0] : memref<10xf32>
    affine.store %cf7, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    affine.store %cf7, %d[%i1] : memref<10xf32>
    %v1 = affine.load %c[%i1] : memref<10xf32>
  }
  affine.for %i2 = 0 to 10 {
    %v2 = affine.load %b[%i2] : memref<10xf32>
    affine.store %cf7, %c[%i2] : memref<10xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_intermediate_store() {
func @should_not_fuse_across_intermediate_store() {
  %0 = alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 0}}
    %v0 = affine.load %0[%i0] : memref<10xf32>
    "op0"(%v0) : (f32) -> ()
  }

  // Should not fuse loop nests '%i0' and '%i1' across top-level store.
  affine.store %cf7, %0[%c0] : memref<10xf32>

  affine.for %i1 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 0}}
    %v1 = affine.load %0[%i1] : memref<10xf32>
    "op1"(%v1) : (f32) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_intermediate_load() {
func @should_not_fuse_across_intermediate_load() {
  %0 = alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 0}}
    affine.store %cf7, %0[%i0] : memref<10xf32>
  }

  // Should not fuse loop nests '%i0' and '%i1' across top-level load.
  %v0 = affine.load %0[%c0] : memref<10xf32>
  "op0"(%v0) : (f32) -> ()

  affine.for %i1 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 0}}
    affine.store %cf7, %0[%i1] : memref<10xf32>
  }

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_ssa_value_def() {
func @should_not_fuse_across_ssa_value_def() {
  %0 = alloc() : memref<10xf32>
  %1 = alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 0}}
    %v0 = affine.load %0[%i0] : memref<10xf32>
    affine.store %v0, %1[%i0] : memref<10xf32>
  }

  // Loop nest '%i0" cannot be fused past load from '%1' due to RAW dependence.
  %v1 = affine.load %1[%c0] : memref<10xf32>
  "op0"(%v1) : (f32) -> ()

  // Loop nest '%i1' cannot be fused past SSA value def '%c2' which it uses.
  %c2 = constant 2 : index

  affine.for %i1 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 0}}
    affine.store %cf7, %0[%c2] : memref<10xf32>
  }

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_store_before_load() {
func @should_not_fuse_store_before_load() {
  %0 = alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 2 at depth 0}}
    affine.store %cf7, %0[%i0] : memref<10xf32>
    %v0 = affine.load %0[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 10 {
    %v1 = affine.load %0[%i1] : memref<10xf32>
  }

  affine.for %i2 = 0 to 10 {
    // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 2 into loop nest 0 at depth 0}}
    affine.store %cf7, %0[%i2] : memref<10xf32>
    %v2 = affine.load %0[%i2] : memref<10xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_load_at_depth1() {
func @should_not_fuse_across_load_at_depth1() {
  %0 = alloc() : memref<10x10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 1}}
      affine.store %cf7, %0[%i0, %i1] : memref<10x10xf32>
    }

    %v1 = affine.load %0[%i0, %c0] : memref<10x10xf32>

    affine.for %i3 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 1}}
      affine.store %cf7, %0[%i0, %i3] : memref<10x10xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_load_in_loop_at_depth1() {
func @should_not_fuse_across_load_in_loop_at_depth1() {
  %0 = alloc() : memref<10x10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 2 at depth 1}}
      affine.store %cf7, %0[%i0, %i1] : memref<10x10xf32>
    }

    affine.for %i2 = 0 to 10 {
      %v1 = affine.load %0[%i0, %i2] : memref<10x10xf32>
    }

    affine.for %i3 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 2 into loop nest 0 at depth 1}}
      affine.store %cf7, %0[%i0, %i3] : memref<10x10xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_store_at_depth1() {
func @should_not_fuse_across_store_at_depth1() {
  %0 = alloc() : memref<10x10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 1}}
      %v0 = affine.load %0[%i0, %i1] : memref<10x10xf32>
    }

    affine.store %cf7, %0[%i0, %c0] : memref<10x10xf32>

    affine.for %i3 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 1}}
      %v1 = affine.load %0[%i0, %i3] : memref<10x10xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_store_in_loop_at_depth1() {
func @should_not_fuse_across_store_in_loop_at_depth1() {
  %0 = alloc() : memref<10x10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 2 at depth 1}}
      %v0 = affine.load %0[%i0, %i1] : memref<10x10xf32>
    }

    affine.for %i2 = 0 to 10 {
      affine.store %cf7, %0[%i0, %i2] : memref<10x10xf32>
    }

    affine.for %i3 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 2 into loop nest 0 at depth 1}}
      %v1 = affine.load %0[%i0, %i3] : memref<10x10xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_across_ssa_value_def_at_depth1() {
func @should_not_fuse_across_ssa_value_def_at_depth1() {
  %0 = alloc() : memref<10x10xf32>
  %1 = alloc() : memref<10x10xf32>
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 0 into loop nest 1 at depth 1}}
      %v0 = affine.load %0[%i0, %i1] : memref<10x10xf32>
      affine.store %v0, %1[%i0, %i1] : memref<10x10xf32>
    }

    // RAW dependence from store in loop nest '%i1' to 'load %1' prevents
    // fusion loop nest '%i1' into loops after load.
    %v1 = affine.load %1[%i0, %c0] : memref<10x10xf32>
    "op0"(%v1) : (f32) -> ()

    // Loop nest '%i2' cannot be fused past SSA value def '%c2' which it uses.
    %c2 = constant 2 : index

    affine.for %i2 = 0 to 10 {
      // expected-remark@-1 {{block-level dependence preventing fusion of loop nest 1 into loop nest 0 at depth 1}}
      affine.store %cf7, %0[%i0, %c2] : memref<10x10xf32>
    }
  }
  return
}