// RUN: mlir-opt %s -loop-fusion | FileCheck %s

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0 * 2 + 2)
// CHECK: [[MAP1:#map[0-9]+]] = (d0) -> (d0 * 3 + 1)
// CHECK: [[MAP2:#map[0-9]+]] = (d0) -> (d0 * 2)
// CHECK: [[MAP3:#map[0-9]+]] = (d0) -> (d0 * 2 + 1)
// CHECK: [[MAP4:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 * 2 - d1 - s0 * 7 + 3, d0 * 9 + d1 * 3 + s1 * 13 - 10)
// CHECK: [[MAP5:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 * 2 - 1, d1 * 3 + s0 * 2 + s1 * 3)
// CHECK: [[MAP6:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 * 2 - 1, d1 * 3 + s0 + s1 * 3)

// The dependence check for this test builds the following set of constraints,
// where the equality contraint equates the two accesses to the memref (from
// different loops), and the inequality constraints represent the upper and
// lower bounds for each loop. After elimination, this linear system can be
// shown to be non-empty (i.e. x0 = x1 = 1 is a solution). As such, the
// dependence check between accesses in the two loops will return true, and
// the loops (according to the current test loop fusion algorithm) should not be
// fused.
//
//   x0   x1   x2
//   2   -3    1   =  0
//   1    0    0   >= 0
//  -1    0    100 >= 0
//   0    1    0   >= 0
//   0   -1    100 >= 0
//
// CHECK-LABEL: mlfunc @loop_fusion_1d_should_not_fuse_loops() {
mlfunc @loop_fusion_1d_should_not_fuse_loops() {
  %m = alloc() : memref<100xf32, (d0) -> (d0)>
  // Check that the first loop remains unfused.
  // CHECK:      for %i0 = 0 to 100 {
  // CHECK-NEXT:   [[I0:%[0-9]+]] = affine_apply [[MAP0]](%i0)
  // CHECK:        store {{.*}}, %{{[0-9]+}}{{\[}}[[I0]]{{\]}}
  // CHECK-NEXT: }
  for %i0 = 0 to 100 {
    %a0 = affine_apply (d0) -> (d0 * 2 + 2) (%i0)
    %c1 = constant 1.0 : f32
    store %c1, %m[%a0] : memref<100xf32, (d0) -> (d0)>
  }
  // Check that the second loop remains unfused.
  // CHECK:      for %i1 = 0 to 100 {
  // CHECK-NEXT:   [[I1:%[0-9]+]] = affine_apply [[MAP1]](%i1)
  // CHECK-NEXT:   load %{{[0-9]+}}{{\[}}[[I1]]{{\]}}
  // CHECK-NEXT: } 
  for %i1 = 0 to 100 {
    %a1 = affine_apply (d0) -> (d0 * 3 + 1) (%i1)
    %v0 = load %m[%a1] : memref<100xf32, (d0) -> (d0)>
  }
  return
}

// The dependence check for this test builds the following set of constraints:
//
//   x0   x1   x2
//   2   -2   -1   =  0
//   1    0    0   >= 0
//  -1    0    100 >= 0
//   0    1    0   >= 0
//   0   -1    100 >= 0
//
// After elimination, this linear system can be shown to have no solutions, and
// so no dependence exists and the loops should be fused in this test (according
// to the current trivial test loop fusion policy).
//
//
// CHECK-LABEL: mlfunc @loop_fusion_1d_should_fuse_loops() {
mlfunc @loop_fusion_1d_should_fuse_loops() {
  %m = alloc() : memref<100xf32, (d0) -> (d0)>
  // Should fuse statements from the second loop into the first loop.
  // CHECK:      for %i0 = 0 to 100 {
  // CHECK-NEXT:   [[I0:%[0-9]+]] = affine_apply [[MAP2]](%i0)
  // CHECK:        store {{.*}}, %{{[0-9]+}}{{\[}}[[I0]]{{\]}}
  // CHECK-NEXT:   [[I1:%[0-9]+]] = affine_apply [[MAP3]](%i0)
  // CHECK-NEXT:   load %{{[0-9]+}}{{\[}}[[I1]]{{\]}}
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  for %i0 = 0 to 100 {
    %a0 = affine_apply (d0) -> (d0 * 2) (%i0)
    %c1 = constant 1.0 : f32
    store %c1, %m[%a0] : memref<100xf32, (d0) -> (d0)>
  }
  
  for %i1 = 0 to 100 {
    %a1 = affine_apply (d0) -> (d0 * 2 + 1) (%i1)

    %v0 = load %m[%a1] : memref<100xf32, (d0) -> (d0)>
  }
  return
}

// The dependence check for this test builds the following set of
// equality constraints (one for each memref dimension). Note: inequality
// constraints for loop bounds not shown.
//
//   i0  i1  i2  i3  s0  s1  s2  c
//   2  -1  -2   0  -7   0   0   4  = 0
//   9   3   0  -3   0   11 -3  -10 = 0
//
// CHECK-LABEL: mlfunc @loop_fusion_2d_should_not_fuse_loops() {
mlfunc @loop_fusion_2d_should_not_fuse_loops() {
  %m = alloc() : memref<10x10xf32>

  %s0 = constant 7 : index
  %s1 = constant 11 : index
  %s2 = constant 13 : index
  // Check that the first loop remains unfused.
  // CHECK:      for %i0 = 0 to 100 { 
  // CHECK-NEXT:   for %i1 = 0 to 50 {
  // CHECK-NEXT:     [[I0:%[0-9]+]] = affine_apply [[MAP4]](%i0, %i1)[%c7, %c11]
  // CHECK:          store {{.*}}, %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  for %i0 = 0 to 100 {
    for %i1 = 0 to 50 {
      %a0 = affine_apply
        (d0, d1)[s0, s1] ->
	  (d0 * 2 -d1 + -7 * s0 + 3 , d0 * 9 + d1 * 3 + 13 * s1 - 10)
	    (%i0, %i1)[%s0, %s1]
      %c1 = constant 1.0 : f32
      store %c1, %m[%a0#0, %a0#1] : memref<10x10xf32>
    }
  }
  // Check that the second loop remains unfused.
  // CHECK:      for %i2 = 0 to 100 {
  // CHECK-NEXT:   for %i3 = 0 to 50 {  
  // CHECK-NEXT:     [[I1:%[0-9]+]] = affine_apply [[MAP5]](%i2, %i3)[%c11, %c13]
  // CHECK-NEXT:   load %{{[0-9]+}}{{\[}}[[I1]]#0, [[I1]]#1{{\]}}
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  for %i2 = 0 to 100 {
    for %i3 = 0 to 50 {
      %a1 = affine_apply
        (d0, d1)[s0, s1] ->
	  (d0 * 2 - 1, d1 * 3 + s0 * 2 + s1 * 3) (%i2, %i3)[%s1, %s2]
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }

  return
}

// The dependence check for this test builds the following set of
// equality constraints (one for each memref dimension). Note: inequality
// constraints for loop bounds not shown.
//
//   i0  i1  i2  i3  s0  s1  s2  c
//   2  -1  -2   0  -7   0   0   4  = 0
//   9   3   0  -3   0   12 -3  -10 = 0
//
// The second equality will fail the GCD test and so the system has no solution,
// so the loops should be fused under the current test policy.
//
// CHECK-LABEL: mlfunc @loop_fusion_2d_should_fuse_loops() {
mlfunc @loop_fusion_2d_should_fuse_loops() {
  %m = alloc() : memref<10x10xf32>

  %s0 = constant 7 : index
  %s1 = constant 11 : index
  %s2 = constant 13 : index
  // Should fuse statements from the second loop into the first loop.
  // CHECK:      for %i0 = 0 to 100 {
  // CHECK-NEXT:   for %i1 = 0 to 50 {
  // CHECK-NEXT:     [[I0:%[0-9]+]] = affine_apply [[MAP4]](%i0, %i1)[%c7, %c11]
  // CHECK:          store {{.*}}, %{{[0-9]+}}{{\[}}[[I0]]#0, [[I0]]#1{{\]}}
  // CHECK-NEXT:     [[I1:%[0-9]+]] = affine_apply [[MAP6]](%i0, %i1)[%c11, %c13]
  // CHECK-NEXT:     load %{{[0-9]+}}{{\[}}[[I1]]#0, [[I1]]#1{{\]}}
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  for %i0 = 0 to 100 {
    for %i1 = 0 to 50 {
      %a0 = affine_apply
        (d0, d1)[s0, s1] ->
	  (d0 * 2 -d1 + -7 * s0 + 3 , d0 * 9 + d1 * 3 + 13 * s1 - 10)
	    (%i0, %i1)[%s0, %s1]
      %c1 = constant 1.0 : f32
      store %c1, %m[%a0#0, %a0#1] : memref<10x10xf32>
    }
  }

  for %i2 = 0 to 100 {
    for %i3 = 0 to 50 {
      %a1 = affine_apply
        (d0, d1)[s0, s1] ->
	  (d0 * 2 - 1, d1 * 3 + s0 + s1 * 3) (%i2, %i3)[%s1, %s2]
      %v0 = load %m[%a1#0, %a1#1] : memref<10x10xf32>
    }
  }

  return
}