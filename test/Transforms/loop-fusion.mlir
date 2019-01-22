// RUN: mlir-opt %s -loop-fusion -split-input-file -verify | FileCheck %s

// TODO(andydavis) Add more tests:
// *) Add nested fusion test cases when non-constant loop bound support is
//    added to iteration domain in dependence check.
// *) Add a test w/ floordiv/ceildiv/mod when supported in dependence check.
// *) Add tests which check fused computation slice indexing and loop bounds.
// TODO(andydavis) Test clean up: move memref allocs to func args.

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_raw_dep_for_locality() {
func @should_fuse_raw_dep_for_locality() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %1 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   store %cst, %0[%1] : memref<1xf32>
  // CHECK-NEXT:   %2 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   %3 = load %0[%2] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-DAG: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_reduction_to_pointwise() {
func @should_fuse_reduction_to_pointwise() {
  %a = alloc() : memref<10x10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %v0 = load %b[%i0] : memref<10xf32>
      %v1 = load %a[%i0, %i1] : memref<10x10xf32>
      %v3 = addf %v0, %v1 : f32
      store %v3, %b[%i0] : memref<10xf32>
    }
  }
  for %i2 = 0 to 10 {
    %v4 = load %b[%i2] : memref<10xf32>
    store %v4, %c[%i2] : memref<10xf32>
  }

  // Should fuse in entire inner loop on %i1 from source loop nest, as %i1
  // is not used in the access function of the store/load on %b.
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    for %i1 = 0 to 10 {
  // CHECK-NEXT:      %3 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:      %4 = load %2[%3] : memref<1xf32>
  // CHECK-NEXT:      %5 = load %0[%i0, %i1] : memref<10x10xf32>
  // CHECK-NEXT:      %6 = addf %4, %5 : f32
  // CHECK-NEXT:      %7 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:      store %6, %2[%7] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    %8 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:    %9 = load %2[%8] : memref<1xf32>
  // CHECK-NEXT:    store %9, %1[%i0] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-DAG: [[MAP_SHIFT_MINUS_ONE_R1:#map[0-9]+]] = (d0) -> (d0 - 1)
// CHECK-DAG: [[MAP_SHIFT_BY_ONE:#map[0-9]+]] = (d0, d1) -> (d0 + 1, d1 + 1)
// CHECK-DAG: [[MAP_SHIFT_MINUS_IV_R2:#map[0-9]+]] = (d0, d1, d2, d3) -> (-d0 + d2, -d1 + d3)

// CHECK-LABEL: func @should_fuse_loop_nests_with_shifts() {
func @should_fuse_loop_nests_with_shifts() {
  %a = alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 9 {
    for %i1 = 0 to 9 {
      %a0 = affine_apply (d0, d1) -> (d0 + 1, d1 + 1) (%i0, %i1)
      store %cf7, %a[%a0#0, %a0#1] : memref<10x10xf32>
    }
  }
  for %i2 = 1 to 10 {
    for %i3 = 1 to 10 {
      %v0 = load %a[%i2, %i3] : memref<10x10xf32>
    }
  }

  // Source slice affine apply sequence:
  // *) First two affine apply's map from the dst to src iteration space.
  // *) Third affine apply is access function around src store.
  // *) Fourth affine apply shifts the stores access function by '-1', because
  //    of the offset induced by reducing the memref shape from 10x10 to 9x9.
  // *) Fifth affine apply shifts the loads access function by '-1', because
  //    of the offset induced by reducing the memref shape from 10x10 to 9x9.
  // NOTE: Should create a private memref with reduced shape 9x9xf32.
  // CHECK:      %0 = alloc() : memref<1x1xf32>
  // CHECK-NEXT: for %i0 = 1 to 10 {
  // CHECK-NEXT:   for %i1 = 1 to 10 {
  // CHECK-NEXT:     %1 = affine_apply [[MAP_SHIFT_MINUS_ONE_R1]](%i0)
  // CHECK-NEXT:     %2 = affine_apply [[MAP_SHIFT_MINUS_ONE_R1]](%i1)
  // CHECK-NEXT:     %3 = affine_apply [[MAP_SHIFT_BY_ONE]](%1, %2)
  // CHECK-NEXT:     %4 = affine_apply [[MAP_SHIFT_MINUS_IV_R2]](%i0, %i1, %3#0, %3#1)
  // CHECK-NEXT:     store %cst, %0[%4#0, %4#1] : memref<1x1xf32>
  // CHECK-NEXT:     %5 = affine_apply [[MAP_SHIFT_MINUS_IV_R2]](%i0, %i1, %i0, %i1)
  // CHECK-NEXT:     %6 = load %0[%5#0, %5#1] : memref<1x1xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-DAG: [[MAP0:#map[0-9]+]] = (d0, d1, d2, d3) -> (-d0 + d2, -d1 + d3)

// CHECK-LABEL: func @should_fuse_loop_nest() {
func @should_fuse_loop_nest() {
  %a = alloc() : memref<10x10xf32>
  %b = alloc() : memref<10x10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      store %cf7, %a[%i0, %i1] : memref<10x10xf32>
    }
  }
  for %i2 = 0 to 10 {
    for %i3 = 0 to 10 {
      %v0 = load %a[%i3, %i2] : memref<10x10xf32>
      store %v0, %b[%i2, %i3] : memref<10x10xf32>
    }
  }
  for %i4 = 0 to 10 {
    for %i5 = 0 to 10 {
      %v1 = load %b[%i4, %i5] : memref<10x10xf32>
    }
  }
  // Expecting private memref for '%b' first, then private memref for '%a'.
  // CHECK:      [[NEWB:%[0-9]+]] = alloc() : memref<1x1xf32>
  // CHECK-NEXT: [[NEWA:%[0-9]+]] = alloc() : memref<1x1xf32>
  // CHECK-NEXT: for %i0 = 0 to 10 {
  // CHECK-NEXT:   for %i1 = 0 to 10 {
  // CHECK-NEXT:     %2 = affine_apply [[MAP0]](%i1, %i0, %i1, %i0)
  // CHECK-NEXT:     store %cst, [[NEWA]][%2#0, %2#1] : memref<1x1xf32>
  // CHECK-NEXT:     %3 = affine_apply [[MAP0]](%i1, %i0, %i1, %i0)
  // CHECK-NEXT:     %4 = load [[NEWA]][%3#0, %3#1] : memref<1x1xf32>
  // CHECK-NEXT:     %5 = affine_apply [[MAP0]](%i0, %i1, %i0, %i1)
  // CHECK-NEXT:     store %4, [[NEWB]][%5#0, %5#1] : memref<1x1xf32>
  // CHECK-NEXT:     %6 = affine_apply [[MAP0]](%i0, %i1, %i0, %i1)
  // CHECK-NEXT:     %7 = load [[NEWB]][%6#0, %6#1] : memref<1x1xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-DAG: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_across_intermediate_loop_with_no_deps() {
func @should_fuse_across_intermediate_loop_with_no_deps() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %v0, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %c[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %b[%i2] : memref<10xf32>
  }

  // Should fuse first loop (past second loop with no dependences) into third.
  // Note that fusion creates a private memref '%2' for the fused loop nest.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      %2 = alloc() : memref<1xf32>
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %3 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT:   %4 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:   store %3, %2[%4] : memref<1xf32>
  // CHECK-NEXT:   %5 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:   %6 = load %2[%5] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_all_loops() {
func @should_fuse_all_loops() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  // Set up flow dependences from first and second loops to third.
  for %i0 = 0 to 10 {
    store %cf7, %a[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %b[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v0 = load %a[%i2] : memref<10xf32>
    %v1 = load %b[%i2] : memref<10xf32>
  }

  // Should fuse first and second loops into third.
  // Expecting private memref for '%b' first, then private memref for '%a'.
  // CHECK:      [[NEWB:%[0-9]+]] = alloc() : memref<1xf32>
  // CHECK-NEXT: [[NEWA:%[0-9]+]] = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i0 = 0 to 10 {
  // CHECK-NEXT:   %2 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   store %cst, [[NEWA]][%2] : memref<1xf32>
  // CHECK-NEXT:   %3 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   store %cst, [[NEWB]][%3] : memref<1xf32>
  // CHECK-NEXT:   %4 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   %5 = load [[NEWA]][%4] : memref<1xf32>
  // CHECK-NEXT:   %6 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   %7 = load [[NEWB]][%6] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_first_and_second_loops() {
func @should_fuse_first_and_second_loops() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %a[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %a[%i1] : memref<10xf32>
    store %cf7, %b[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %c[%i2] : memref<10xf32>
  }

  // Should fuse first loop into the second (last loop should not be fused).
  // Should create private memref '%2' for fused loop.
  // CHECK:      %2 = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i0 = 0 to 10 {
  // CHECK-NEXT:   %3 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   store %cst, %2[%3] : memref<1xf32>
  // CHECK-NEXT:   %4 = affine_apply [[MAP0]](%i0, %i0)
  // CHECK-NEXT:   %5 = load %2[%4] : memref<1xf32>
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %6 = load %1[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_would_create_cycle() {
func @should_not_fuse_would_create_cycle() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %c = alloc() : memref<10xf32>

  %cf7 = constant 7.0 : f32

  // Set up the following dependences:
  // 1) loop0 -> loop1 on memref '%a'
  // 2) loop0 -> loop2 on memref '%b'
  // 3) loop1 -> loop2 on memref '%c'
  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %cf7, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %a[%i1] : memref<10xf32>
    %v1 = load %c[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v2 = load %b[%i2] : memref<10xf32>
    store %cf7, %c[%i2] : memref<10xf32>
  }
  // Should not fuse: fusing loop first loop into last would create a cycle.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %3 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   store %cst, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT:   %4 = load %2[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %5 = load %1[%i2] : memref<10xf32>
  // CHECK-NEXT:   store %cst, %2[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----
// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_across_waw_dep_with_private_memref() {
func @should_fuse_across_waw_dep_with_private_memref() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %m[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %m[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the WAW dependence between %i0 and %i1
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      %1 = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i2 = 0 to 10 {
  // CHECK-NEXT:    %2 = affine_apply #map0(%i2, %i2)
  // CHECK-NEXT:    store %cst, %1[%2] : memref<1xf32>
  // CHECK-NEXT:    %3 = affine_apply #map0(%i2, %i2)
  // CHECK-NEXT:    %4 = load %1[%3] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_war_dep_would_be_violated() {
func @should_not_fuse_war_dep_would_be_violated() {
  %a = alloc() : memref<10xf32>
  %b = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    %v0 = load %a[%i0] : memref<10xf32>
    store %v0, %b[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    store %cf7, %a[%i1] : memref<10xf32>
  }
  for %i2 = 0 to 10 {
    %v1 = load %b[%i2] : memref<10xf32>
  }
  // Fusing loop %i0 to %i2 would violate the WAR dependence between %i0 and %i1
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %2 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT:   store %2, %1[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i2 = 0 to 10 {
  // CHECK-NEXT:   %3 = load %1[%i2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_with_private_memref_if_top_level_access() {
func @should_fuse_with_private_memref_if_top_level_access() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }

  %c0 = constant 4 : index
  %v1 = load %m[%c0] : memref<10xf32>
  // Top-level load to '%m' should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      %1 = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i1 = 0 to 10 {
  // CHECK-NEXT:   %2 = affine_apply #map0(%i1, %i1)
  // CHECK-NEXT:   store %cst, %1[%2] : memref<1xf32>
  // CHECK-NEXT:   %3 = affine_apply #map0(%i1, %i1)
  // CHECK-NEXT:   %4 = load %1[%3] : memref<1xf32>
  // CHECK-NEXT: }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_no_top_level_access() {
func @should_fuse_no_top_level_access() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   %1 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:   store %cst, %0[%1] : memref<1xf32>
  // CHECK-NEXT:   %2 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:   %3 = load %0[%2] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: func @should_not_fuse_if_inst_at_top_level() {
func @should_not_fuse_if_inst_at_top_level() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  %c0 = constant 4 : index
  if #set0(%c0) {
  }
  // Top-level IfInst should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: func @should_not_fuse_if_inst_in_loop_nest() {
func @should_not_fuse_if_inst_in_loop_nest() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %c4 = constant 4 : index

  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    if #set0(%c4) {
    }
    %v0 = load %m[%i1] : memref<10xf32>
  }

  // IfInst in ForInst should prevent fusion.
  // CHECK:      for %i0 = 0 to 10 {
  // CHECK-NEXT:   store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK:      for %i1 = 0 to 10 {
  // CHECK-NEXT:   if #set0(%c4) {
  // CHECK-NEXT:   }  
  // CHECK-NEXT:   %1 = load %0[%i1] : memref<10xf32>
  // CHECK-NEXT: }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2) -> (d0, d1, d2)
// CHECK: [[MAP1:#map[0-9]+]] = (d0, d1, d2, d3, d4, d5) -> (-d0 + d3, -d1 + d4, -d2 + d5)
// CHECK: [[MAP_PERMUTE:#map[0-9]+]] = (d0, d1, d2) -> (d1, d2, d0)

#map0 = (d0, d1, d2) -> (d0, d1, d2)

// CHECK-LABEL: func @remap_ivs() {
func @remap_ivs() {
  %m = alloc() : memref<10x20x30xf32>

  %cf7 = constant 7.0 : f32
  for %i0 = 0 to 10 {
    for %i1 = 0 to 20 {
      for %i2 = 0 to 30 {
        %a0 = affine_apply (d0, d1, d2) -> (d0, d1, d2) (%i0, %i1, %i2)
        store %cf7, %m[%a0#0, %a0#1, %a0#2] : memref<10x20x30xf32>
      }
    }
  }
  for %i3 = 0 to 30 {
    for %i4 = 0 to 10 {
      for %i5 = 0 to 20 {
        %a1 = affine_apply (d0, d1, d2) -> (d1, d2, d0) (%i3, %i4, %i5)
        %v0 = load %m[%a1#0, %a1#1, %a1#2] : memref<10x20x30xf32>
      }
    }
  }
// CHECK:       for %i0 = 0 to 30 {
// CHECK-NEXT:    for %i1 = 0 to 10 {
// CHECK-NEXT:      for %i2 = 0 to 20 {
// CHECK-NEXT:        %1 = affine_apply [[MAP0]](%i1, %i2, %i0)
// CHECK-NEXT:        %2 = affine_apply [[MAP1]](%i1, %i2, %i0, %1#0, %1#1, %1#2)
// CHECK-NEXT:        store %cst, %0[%2#0, %2#1, %2#2] : memref<1x1x1xf32>
// CHECK-NEXT:        %3 = affine_apply [[MAP_PERMUTE]](%i0, %i1, %i2)
// CHECK-NEXT:        %4 = affine_apply [[MAP1]](%i1, %i2, %i0, %3#0, %3#1, %3#2)
// CHECK-NEXT:        %5 = load %0[%4#0, %4#1, %4#2] : memref<1x1x1xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

  return
}

// -----

// CHECK-DAG: #map0 = (d0, d1) -> (d0 * 4 + d1)
// CHECK-DAG: #map1 = (d0) -> (d0 floordiv 4, d0 mod 4)

// Reshape from a 64 x f32 to 16 x 4 x f32.
// CHECK-LABEL: func @fuse_reshape_64_16_4
func @fuse_reshape_64_16_4(%in : memref<64xf32>) {
  %out = alloc() : memref<16x4xf32>

  for %i0 = 0 to 64 {
    %v = load %in[%i0] : memref<64xf32>
    %idx = affine_apply (d0) -> (d0 floordiv 4, d0 mod 4) (%i0)
    store %v, %out[%idx#0, %idx#1] : memref<16x4xf32>
  }

  for %i1 = 0 to 16 {
    for %i2 = 0 to 4 {
      %w = load %out[%i1, %i2] : memref<16x4xf32>
      "foo"(%w) : (f32) -> ()
    }
  }
  return
  // CHECK:      for %i0 =
  // CHECK-NEXT:   for %i1 =
  // CHECK-NOT:    for
  // CHECK:        }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----
// CHECK-DAG: #map0 = (d0) -> (d0 floordiv 4)
// CHECK-DAG: #map1 = (d0) -> (d0 mod 4)
// CHECK-DAG: [[MAP2:#map[0-9]+]] = (d0, d1) -> (d0 * 4 + d1)
// CHECK-DAG: [[MAP3:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// Reshape a 16x4xf32 to 64xf32.
// CHECK-LABEL: func @fuse_reshape_16_4_64
func @fuse_reshape_16_4_64() {
  %in = alloc() : memref<16x4xf32>
  %out = alloc() : memref<64xf32>

  for %i0 = 0 to 16 {
    for %i1 = 0 to 4 {
      %v = load %in[%i0, %i1] : memref<16x4xf32>
      %idx = affine_apply (d0, d1) -> (4*d0 + d1) (%i0, %i1)
      store %v, %out[%idx] : memref<64xf32>
    }
  }

  for %i2 = 0 to 64 {
    %w = load %out[%i2] : memref<64xf32>
    "foo"(%w) : (f32) -> ()
  }
// CHECK:       for %i0 = 0 to 64 {
// CHECK-NEXT:    %2 = affine_apply #map0(%i0)
// CHECK-NEXT:    %3 = affine_apply #map1(%i0)
// CHECK-NEXT:    %4 = load %0[%2, %3] : memref<16x4xf32>
// CHECK-NEXT:    %5 = affine_apply [[MAP2]](%2, %3)
// CHECK-NEXT:    %6 = affine_apply [[MAP3]](%i0, %5)
// CHECK-NEXT:    store %4, %1[%6] : memref<1xf32>
// CHECK-NEXT:    %7 = affine_apply [[MAP3]](%i0, %i0)
// CHECK-NEXT:    %8 = load %1[%7] : memref<1xf32>
// CHECK-NEXT:    "foo"(%8) : (f32) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}


// -----

// TODO(b/123072438) Re-enable test MemRefRegion bug is fixed.
// All three loop nests below (6-d one, 2-d one, 2-d one is fused into a single
// 2-d loop nest).
// CHECK-LABEL: func @R6_to_R2_reshape
func @R6_to_R2_reshape_square() -> memref<64x9xi32> {
  %in = alloc() : memref<2x2x3x3x16x1xi32>
  %out = alloc() : memref<64x9xi32>
  %live_out = alloc() : memref<64x9xi32>

  // Initialize input.
  for %i0 = 0 to 2 {
    for %i1 = 0 to 2 {
      for %i2 = 0 to 3 {
        for %i3 = 0 to 3 {
          for %i4 = 0 to 16 {
            for %i5 = 0 to 1 {
              %val = "foo"(%i0, %i1, %i2, %i3, %i4, %i5) : (index, index, index, index, index, index) -> i32
              store %val, %in[%i0, %i1, %i2, %i3, %i4, %i5] : memref<2x2x3x3x16x1xi32>
            }
          }
        }
      }
    }
  }

  for %ii = 0 to 64 {
    for %jj = 0 to 9 {
      // Convert output coordinates to linear index.
      %a0 = affine_apply (d0, d1) -> (d0 * 9 + d1) (%ii, %jj)
      %a1 = affine_apply (d0) -> (
          d0 floordiv (2 * 3 * 3 * 16 * 1),
          (d0 mod 288) floordiv (3 * 3 * 16 * 1),
          ((d0 mod 288) mod 144) floordiv (3 * 16 * 1),
          (((d0 mod 288) mod 144) mod 48) floordiv (16 * 1),
          ((((d0 mod 288) mod 144) mod 48) mod 16),
          ((((d0 mod 144) mod 144) mod 48) mod 16) mod 1
        ) (%a0)
      %v = load %in[%a1#0, %a1#1, %a1#2, %a1#3, %a1#4, %a1#5]
        : memref<2x2x3x3x16x1xi32>
      store %v, %out[%ii, %jj] : memref<64x9xi32>
    }
  }

  for %i = 0 to 64 {
    for %j = 0 to 9 {
      %a = load %out[%i, %j] : memref<64x9xi32>
      %b = muli %a, %a : i32
      store %b, %live_out[%i, %j] : memref<64x9xi32>
    }
  }
  return %live_out : memref<64x9xi32>
}
// Everything above is fused to a single 2-d loop nest, and the 6-d tensor %in
// is eliminated if -memref-dataflow-opt is also supplied.
//
// CHECK:       %0 = alloc() : memref<64x9xi32>
// CHECK-NEXT:  %1 = alloc() : memref<1x1xi32>
// CHECK-NEXT:  %2 = alloc() :  memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:  for %i0 = 0 to 64 {
// CHECK-NEXT:    for %i1 = 0 to 9 {
// CHECK-NEXT:      %3 = affine_apply #map0(%i0, %i1)
// CHECK-NEXT:      %4 = affine_apply #map1(%i0, %i1)
// CHECK-NEXT:      %5 = affine_apply #map2(%i0, %i1)
// CHECK-NEXT:      %6 = affine_apply #map3(%i0, %i1)
// CHECK-NEXT:      %7 = affine_apply #map4(%i0, %i1)
// CHECK-NEXT:      %8 = "foo"(%3, %4, %5, %6, %7, %c0) : (index, index, index, index, index, index) -> i32
// CHECK-NEXT:      %9 = affine_apply #map5(%i0, %i1, %3, %4, %5, %6, %7, %c0)
// CHECK-NEXT:      store %8, %2[%9#0, %9#1, %9#2, %9#3, %9#4, %9#5] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      %10 = affine_apply #map6(%i0, %i1)
// CHECK-NEXT:      %11 = affine_apply #map7(%10)
// CHECK-NEXT:      %12 = affine_apply #map5(%i0, %i1, %11#0, %11#1, %11#2, %11#3, %11#4, %11#5)
// CHECK-NEXT:      %13 = load %2[%12#0, %12#1, %12#2, %12#3, %12#4, %12#5] : memref<1x2x3x3x16x1xi32>
// CHECK-NEXT:      %14 = affine_apply #map8(%i0, %i1, %i0, %i1)
// CHECK-NEXT:      store %13, %1[%14#0, %14#1] : memref<1x1xi32>
// CHECK-NEXT:      %15 = affine_apply #map8(%i0, %i1, %i0, %i1)
// CHECK-NEXT:      %16 = load %1[%15#0, %15#1] : memref<1x1xi32>
// CHECK-NEXT:      %17 = muli %16, %16 : i32
// CHECK-NEXT:      store %17, %0[%i0, %i1] : memref<64x9xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : memref<64x9xi32>

// -----

// CHECK-LABEL: func @fuse_symbolic_bounds
func @fuse_symbolic_bounds(%M : index, %N : index) {
  %N_plus_5 = affine_apply (d0) -> (d0 + 5)(%N)
  %m = alloc(%M, %N_plus_5) : memref<? x ? x f32>

  %c0 = constant 0.0 : f32
  %s = constant 5 : index

  for %i0 = 0 to %M {
    for %i1 = 0 to (d0) -> (d0 + 5) (%N) {
      store %c0, %m[%i0, %i1] : memref<? x ? x f32>
    }
  }

  for %i2 = 0 to %M {
    for %i3 = 0 to %N {
      %idx = affine_apply (d0, d1)[s0] -> (d0, d1 + s0) (%i2, %i3)[%s]
      %v = load %m[%idx#0, %idx#1] : memref<? x ? x f32>
    }
  }

  return
}

// -----
// CHECK-DAG: #map0 = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_reduction_at_depth1
func @should_fuse_reduction_at_depth1() {
  %a = alloc() : memref<10x100xf32>
  %b = alloc() : memref<10xf32>

  for %i0 = 0 to 10 {
    for %i1 = 0 to 100 {
      %v0 = load %b[%i0] : memref<10xf32>
      %v1 = load %a[%i0, %i1] : memref<10x100xf32>
      %v2 = "maxf"(%v0, %v1) : (f32, f32) -> f32
      store %v2, %b[%i0] : memref<10xf32>
    }
  }
  for %i2 = 0 to 10 {
    for %i3 = 0 to 100 {
      %v3 = load %b[%i2] : memref<10xf32>
      %v4 = load %a[%i2, %i3] : memref<10x100xf32>
      %v5 = subf %v4, %v3 : f32
      store %v5, %b[%i2] : memref<10xf32>
    }
  }
  // This test should fuse the src reduction loop at depth 1 in the destination
  // loop nest, which improves locality and enables subsequence passes to
  // decrease the reduction memref size and possibly place it in a faster
  // memory space.
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    for %i1 = 0 to 100 {
  // CHECK-NEXT:      %2 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:      %3 = load %1[%2] : memref<1xf32>
  // CHECK-NEXT:      %4 = load %0[%i0, %i1] : memref<10x100xf32>
  // CHECK-NEXT:      %5 = "maxf"(%3, %4) : (f32, f32) -> f32
  // CHECK-NEXT:      %6 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:      store %5, %1[%6] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i2 = 0 to 100 {
  // CHECK-NEXT:      %7 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:      %8 = load %1[%7] : memref<1xf32>
  // CHECK-NEXT:      %9 = load %0[%i0, %i2] : memref<10x100xf32>
  // CHECK-NEXT:      %10 = subf %9, %8 : f32
  // CHECK-NEXT:      %11 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:      store %10, %1[%11] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----
// CHECK: #map0 = (d0, d1, d2) -> (-d0 + d1, d2)

// CHECK-LABEL: func @should_fuse_at_src_depth1_and_dst_depth1
func @should_fuse_at_src_depth1_and_dst_depth1() {
  %a = alloc() : memref<100x16xf32>
  %b = alloc() : memref<100x16xf32>

  for %i0 = 0 to 100 {
    for %i1 = 0 to 16 {
      %v0 = load %a[%i0, %i1] : memref<100x16xf32>
      "op0"(%v0) : (f32) -> ()
    }
    for %i2 = 0 to 16 {
      %v1 = "op1"() : () -> (f32)
      store %v1, %b[%i0, %i2] : memref<100x16xf32>
    }
  }

  for %i3 = 0 to 100 {
    for %i4 = 0 to 16 {
      %v2 = load %b[%i3, %i4] : memref<100x16xf32>
      "op2"(%v2) : (f32) -> ()
    }
  }
  // We can slice iterations of the '%i0' and '%i1' loops in the the source
  // loop nest, but slicing at depth 2 and inserting the slice in the
  // destination loop nest at depth2 causes extra computation. Instead,
  // the fusion algorithm should detect that the source loop should be sliced
  // at depth 1 and the slice should be inserted at depth 1.
  // CHECK:       for %i0 = 0 to 100 {
  // CHECK-NEXT:    for %i1 = 0 to 16 {
  // CHECK-NEXT:      %2 = load %0[%i0, %i1] : memref<100x16xf32>
  // CHECK-NEXT:      "op0"(%2) : (f32) -> ()
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i2 = 0 to 16 {
  // CHECK-NEXT:      %3 = "op1"() : () -> f32
  // CHECK-NEXT:      %4 = affine_apply #map0(%i0, %i0, %i2)
  // CHECK-NEXT:      store %3, %1[%4#0, %4#1] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i3 = 0 to 16 {
  // CHECK-NEXT:      %5 = affine_apply #map0(%i0, %i0, %i3)
  // CHECK-NEXT:      %6 = load %1[%5#0, %5#1] : memref<1x16xf32>
  // CHECK-NEXT:      "op2"(%6) : (f32) -> ()
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----
// CHECK: #map0 = (d0, d1) -> (d0 * 10 + d1)
// CHECK: #map1 = (d0, d1, d2) -> (d0 * -10 - d1 + d2)

// CHECK-LABEL: func @should_fuse_src_depth1_at_dst_depth2
func @should_fuse_src_depth1_at_dst_depth2() {
  %a = alloc() : memref<100xf32>
  %c0 = constant 0.0 : f32

  for %i0 = 0 to 100 {
    store %c0, %a[%i0] : memref<100xf32>
  }

  for %i1 = 0 to 10 {
    for %i2 = 0 to 10 {
      %a0 = affine_apply (d0, d1) -> (d0 * 10 + d1) (%i1, %i2)
      %v0 = load %a[%a0] : memref<100xf32>
    }
  }
  // The source loop nest slice loop bound is a function of both destination
  // loop IVs, so we should slice at depth 1 and insert the slice at depth 2.
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    for %i1 = 0 to 10 {
  // CHECK-NEXT:      %1 = affine_apply #map0(%i0, %i1)
  // CHECK-NEXT:      %2 = affine_apply #map1(%i0, %i1, %1)
  // CHECK-NEXT:      store %cst, %0[%2] : memref<1xf32>
  // CHECK-NEXT:      %3 = affine_apply #map0(%i0, %i1)
  // CHECK-NEXT:      %4 = affine_apply #map1(%i0, %i1, %3)
  // CHECK-NEXT:      %5 = load %0[%4] : memref<1xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----
// CHECK: #map0 = ()[s0] -> (s0)

// CHECK-LABEL: func @fusion_at_depth0_not_currently_supported
func @fusion_at_depth0_not_currently_supported() {
  %0 = alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  for %i0 = 0 to 10 {
    store %cst, %0[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %1 = load %0[%c0] : memref<10xf32>
  }
  // NOTE: Should shrink memref size to 1 element access by load in dst loop
  // nest, and make the store in the slice store to the same element.
  // CHECK:       %0 = alloc() : memref<1xf32>
  // CHECK-NEXT:  for    %i0 = 0 to 10 {
  // CHECK-NEXT:    %1 = affine_apply #map0()[%c0]
  // CHECK-NEXT:    store %cst, %0[%1] : memref<1xf32>
  // CHECK-NEXT:    %2 = load %0[%c0] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-DAG: #map0 = (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (-d0 + d4, -d1 + d5, -d2 + d6, -d3 + d7, d8, d9)

// CHECK-LABEL: func @should_fuse_deep_loop_nests
func @should_fuse_deep_loop_nests() {
  %0 = alloc() : memref<2x2x3x3x16x10xf32, 2>
  %1 = alloc() : memref<2x2x3x3x16x10xf32, 2>
  %2 = alloc() : memref<3x3x3x3x16x10xf32, 2>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c1_0 = constant 1 : index
  %cst = constant 0.000000e+00 : f32
  for %i0 = 0 to 2 {
    for %i1 = 0 to 2 {
      for %i2 = 0 to 3 {
        for %i3 = 0 to 3 {
          for %i4 = 0 to 16 {
            for %i5 = 0 to 10 {
              %3 = load %0[%i0, %i1, %i2, %i3, %i4, %i5]
                : memref<2x2x3x3x16x10xf32, 2>
            }
          }
          for %i6 = 0 to 16 {
            for %i7 = 0 to 10 {
              store %cst, %1[%i0, %i1, %i2, %i3, %i6, %i7]
                : memref<2x2x3x3x16x10xf32, 2>
            }
          }
        }
      }
    }
  }
  for %i8 = 0 to 3 {
    for %i9 = 0 to 3 {
      for %i10 = 0 to 2 {
        for %i11 = 0 to 2 {
          for %i12 = 0 to 3 {
            for %i13 = 0 to 3 {
              for %i14 = 0 to 2 {
                for %i15 = 0 to 2 {
                  for %i16 = 0 to 16 {
                    for %i17 = 0 to 10 {
                      %5 = load %0[%i14, %i15, %i12, %i13, %i16, %i17]
                        : memref<2x2x3x3x16x10xf32, 2>
                    }
                  }
                  for %i18 = 0 to 16 {
                    for %i19 = 0 to 10 {
                      %6 = load %1[%i10, %i11, %i8, %i9, %i18, %i19]
                        : memref<2x2x3x3x16x10xf32, 2>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
// The first four loops of the source loop nest can be sliced with iteration
// bounds which are a function of the first four loops of destination loop nest,
// where the destination loops nests have been interchanged.
// CHECK:       for %i0 = 0 to 3 {
// CHECK-NEXT:    for %i1 = 0 to 3 {
// CHECK-NEXT:      for %i2 = 0 to 2 {
// CHECK-NEXT:        for %i3 = 0 to 2 {
// CHECK-NEXT:          for %i4 = 0 to 16 {
// CHECK-NEXT:            for %i5 = 0 to 10 {
// CHECK-NEXT:              %3 = load %0[%i2, %i3, %i0, %i1, %i4, %i5] : memref<2x2x3x3x16x10xf32, 2>
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          for %i6 = 0 to 16 {
// CHECK-NEXT:            for %i7 = 0 to 10 {
// CHECK-NEXT:              %4 = affine_apply #map0(%i2, %i3, %i0, %i1, %i2, %i3, %i0, %i1, %i6, %i7)
// CHECK-NEXT:              store %cst, %2[%4#0, %4#1, %4#2, %4#3, %4#4, %4#5] : memref<1x1x1x1x16x10xf32, 2>
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:          for %i8 = 0 to 3 {
// CHECK-NEXT:            for %i9 = 0 to 3 {
// CHECK-NEXT:              for %i10 = 0 to 2 {
// CHECK-NEXT:                for %i11 = 0 to 2 {
// CHECK-NEXT:                  for %i12 = 0 to 16 {
// CHECK-NEXT:                    for %i13 = 0 to 10 {
// CHECK-NEXT:                      %5 = load %0[%i10, %i11, %i8, %i9, %i12, %i13] : memref<2x2x3x3x16x10xf32, 2>
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                  for %i14 = 0 to 16 {
// CHECK-NEXT:                    for %i15 = 0 to 10 {
// CHECK-NEXT:                      %6 = affine_apply #map0(%i2, %i3, %i0, %i1, %i2, %i3, %i0, %i1, %i14, %i15)
// CHECK-NEXT:                      %7 = load %2[%6#0, %6#1, %6#2, %6#3, %6#4, %6#5] : memref<1x1x1x1x16x10xf32, 2>
// CHECK-NEXT:                    }
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}

// -----
// CHECK: #map0 = (d0, d1, d2) -> (-d0 + d1, d2)

// CHECK-LABEL: func @should_fuse_at_depth1_and_reduce_slice_trip_count
func @should_fuse_at_depth1_and_reduce_slice_trip_count() {
  %a = alloc() : memref<4x256xf32>
  %b = alloc() : memref<4x256xf32>

  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

  for %i0 = 0 to 4 {
    for %i1 = 0 to 256 {
      %v0 = load %b[%i0, %i1] : memref<4x256xf32>
    }
    for %i2 = 0 to 256 {
      store %cf0, %a[%i0, %i2] : memref<4x256xf32>
    }
  }

  for %d0 = 0 to 4 {
    for %d1 = 0 to 16 {
      %v1 = load %a[%d0, %d1] : memref<4x256xf32>
    }
  }
  // The cost of fusing at depth 2 is greater than the cost of fusing at depth 1
  // for two reasons:
  // 1) Inserting the unsliceable src loop %i1 to a higher depth removes
  //    redundant computation and reduces costs.
  // 2) Inserting the sliceable src loop %i2 at depth 1, we can still reduce
  //    its trip count to 16 (from 256) reducing costs.
  // NOTE: the size of the private memref created for the fused loop nest
  // is reduced from the original shape from 4x256 to 4x16 because of the
  // data accessed by the load.
  // CHECK:       %1 = alloc() : memref<1x16xf32>
  // CHECK-NEXT:  for %i0 = 0 to 4 {
  // CHECK-NEXT:    for %i1 = 0 to 256 {
  // CHECK-NEXT:      %2 = load %0[%i0, %i1] : memref<4x256xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i2 = 0 to 16 {
  // CHECK-NEXT:      %3 = affine_apply #map0(%i0, %i0, %i2)
  // CHECK-NEXT:      store %cst, %1[%3#0, %3#1] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i3 = 0 to 16 {
  // CHECK-NEXT:      %4 = affine_apply #map0(%i0, %i0, %i3)
  // CHECK-NEXT:      %5 = load %1[%4#0, %4#1] : memref<1x16xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_depth1_with_trip_count_20
func @should_fuse_at_depth1_with_trip_count_20() {
  %a = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

  for %i0 = 0 to 100 {
    store %cf0, %a[%i0]: memref<100xf32>
  }

  for %i1 = 0 to 5 {
    for %i2 = 0 to 10 {
      %v0 = load %a[%i2]: memref<100xf32>
    }
    for %i3 = 0 to 10 {
      for %i4 = 0 to 20 {
        %v1 = load %a[%i4]: memref<100xf32>
      }
    }
  }
  // NOTE: The size of the private memref created for fusion is shrunk to 20xf32
  // CHECK:       %0 = alloc() : memref<20xf32>
  // CHECK-NEXT:  for %i0 = 0 to 5 {
  // CHECK-NEXT:    for %i1 = 0 to 20 {
  // CHECK-NEXT:      store %cst, %0[%i1] : memref<20xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i2 = 0 to 10 {
  // CHECK-NEXT:      %1 = load %0[%i2] : memref<20xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i3 = 0 to 10 {
  // CHECK-NEXT:      for %i4 = 0 to 20 {
  // CHECK-NEXT:        %2 = load %0[%i4] : memref<20xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @should_fuse_at_depth1_with_trip_count_19
func @should_fuse_at_depth1_with_trip_count_19() {
  %a = alloc() : memref<100xf32>
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32

  for %i0 = 0 to 100 {
    store %cf0, %a[%i0]: memref<100xf32>
  }

  for %i1 = 0 to 5 {
    for %i2 = 0 to 19 {
      %v0 = load %a[%i2]: memref<100xf32>
    }
    for %i3 = 0 to 10 {
      for %i4 = 0 to 10 {
        %v1 = load %a[%i4]: memref<100xf32>
      }
    }
  }
  // NOTE: The size of the private memref created for fusion is shrunk to 19xf32
  // CHECK:       %0 = alloc() : memref<19xf32>
  // CHECK-NEXT:  for %i0 = 0 to 5 {
  // CHECK-NEXT:    for %i1 = 0 to 19 {
  // CHECK-NEXT:      store %cst, %0[%i1] : memref<19xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i2 = 0 to 19 {
  // CHECK-NEXT:      %1 = load %0[%i2] : memref<19xf32>
  // CHECK-NEXT:    }
  // CHECK-NEXT:    for %i3 = 0 to 10 {
  // CHECK-NEXT:      for %i4 = 0 to 10 {
  // CHECK-NEXT:        %2 = load %0[%i4] : memref<19xf32>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}


// -----
// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @should_fuse_with_private_memrefs_with_diff_shapes() {
func @should_fuse_with_private_memrefs_with_diff_shapes() {
  %m = alloc() : memref<100xf32>
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 100 {
    store %cf7, %m[%i0] : memref<100xf32>
  }
  for %i1 = 0 to 17 {
    %v0 = load %m[%i1] : memref<100xf32>
  }
  for %i2 = 0 to 82 {
    %v1 = load %m[%i2] : memref<100xf32>
  }
  // Should create two new private memrefs customized to the shapes accessed
  // by loops %i1 and %i2.
  // CHECK:      %0 = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i0 = 0 to 17 {
  // CHECK-NEXT:   %1 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:   store %cst, %0[%1] : memref<1xf32>
  // CHECK-NEXT:   %2 = affine_apply #map0(%i0, %i0)
  // CHECK-NEXT:   %3 = load %0[%2] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %4 = alloc() : memref<1xf32>
  // CHECK-NEXT: for %i1 = 0 to 82 {
  // CHECK-NEXT:   %5 = affine_apply #map0(%i1, %i1)
  // CHECK-NEXT:   store %cst, %4[%5] : memref<1xf32>
  // CHECK-NEXT:   %6 = affine_apply #map0(%i1, %i1)
  // CHECK-NEXT:   %7 = load %4[%6] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @fusion_should_not_remove_memref_arg(%arg0: memref<10xf32>) {
func @fusion_should_not_remove_memref_arg(%arg0: memref<10xf32>) {
  %cf7 = constant 7.0 : f32

  for %i0 = 0 to 10 {
    store %cf7, %arg0[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %arg0[%i1] : memref<10xf32>
  }
  // This tests that the loop nest '%i0' should not be removed after fusion
  // because it writes to memref argument '%arg0'.
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    store %cst, %arg0[%i0] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %0 = alloc() : memref<1xf32>
  // CHECK-NEXT:  for %i1 = 0 to 10 {
  // CHECK-NEXT:    %1 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:    store %cst, %0[%1] : memref<1xf32>
  // CHECK-NEXT:    %2 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:    %3 = load %0[%2] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (-d0 + d1)

// CHECK-LABEL: func @fusion_should_not_remove_escaping_memref()
func @fusion_should_not_remove_escaping_memref() -> memref<10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
  }
  for %i1 = 0 to 10 {
    %v0 = load %m[%i1] : memref<10xf32>
  }
  // This tests that the loop nest '%i0' should not be removed after fusion
  // because it writes to memref '%m' which is returned by the function. 
  // CHECK:       for %i0 = 0 to 10 {
  // CHECK-NEXT:    store %cst, %0[%i0] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  %1 = alloc() : memref<1xf32>
  // CHECK-NEXT:  for %i1 = 0 to 10 {
  // CHECK-NEXT:    %2 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:    store %cst, %1[%2] : memref<1xf32>
  // CHECK-NEXT:    %3 = affine_apply [[MAP0]](%i1, %i1)
  // CHECK-NEXT:    %4 = load %1[%3] : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %0 : memref<10xf32>
  return %m : memref<10xf32>
}

// -----

// This should fuse with the %in becoming a 1x1x1.
func @R3_to_R2_reshape() {
  %in = alloc() : memref<2x3x16xi32>

  %c0 = constant 0 : index

  for %i0 = 0 to 2 {
    for %i1 = 0 to 3 {
      for %i2 = 0 to 16 {
        %val = "foo"(%i0, %i1, %i2) : (index, index, index) -> i32
        store %val, %in[%i0, %i1, %i2] : memref<2x3x16xi32>
      }
    }
  }

  for %ii = 0 to 32 {
    for %jj = 0 to 3 {
      %a0 = affine_apply (d0, d1) -> (d0 * 3 + d1) (%ii, %jj)
      %a1 = affine_apply (d0) -> (d0 floordiv (3 * 16)) (%a0)
      %v = load %in[%a1#0, %jj, %c0]
        : memref<2x3x16xi32>
    }
  }
  return
}
// CHECK:      #map0 = (d0, d1) -> ((d0 * 3 + d1) floordiv 48)
// CHECK-NEXT: #map1 = ()[s0] -> (s0)
// CHECK-NEXT: #map2 = (d0, d1, d2, d3, d4) -> (d2 - (d0 * 25 + d1 * 24) floordiv 24, -d1 + d3, d4)
// CHECK-NEXT: #map3 = (d0, d1) -> (d0 * 3 + d1)
// CHECK-NEXT: #map4 = (d0) -> (d0 floordiv 48)
// CHECK-LABEL: func @R3_to_R2_reshape()
// CHECK:        %0 = alloc() : memref<1x1x1xi32>
// CHECK-NEXT:   for %i0 = 0 to 32 {
// CHECK-NEXT:     for %i1 = 0 to 3 {
// CHECK-NEXT:       %1 = affine_apply #map0(%i0, %i1)
// CHECK-NEXT:       %2 = affine_apply #map1()[%c0]
// CHECK-NEXT:       %3 = "foo"(%1, %i1, %2) : (index, index, index) -> i32
// CHECK-NEXT:       %4 = affine_apply #map2(%i0, %i1, %1, %i1, %2)
// CHECK-NEXT:       store %3, %0[%4#0, %4#1, %4#2] : memref<1x1x1xi32>
// CHECK-NEXT:       %5 = affine_apply #map3(%i0, %i1)
// CHECK-NEXT:       %6 = affine_apply #map4(%5)
// CHECK-NEXT:       %7 = affine_apply #map2(%i0, %i1, %6, %i1, %c0)
// CHECK-NEXT:       %8 = load %0[%7#0, %7#1, %7#2] : memref<1x1x1xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
