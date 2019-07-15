// RUN: mlir-opt -loop-coalescing %s | FileCheck %s

// CHECK-LABEL: @one_3d_nest
func @one_3d_nest() {
  // Capture original bounds.  Note that for zero-based step-one loops, the
  // upper bound is also the number of iterations.
  // CHECK: %[[orig_lb:.*]] = constant 0
  // CHECK: %[[orig_step:.*]] = constant 1
  // CHECK: %[[orig_ub_k:.*]] = constant 3
  // CHECK: %[[orig_ub_i:.*]] = constant 42
  // CHECK: %[[orig_ub_j:.*]] = constant 56
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c42 = constant 42 : index
  %c56 = constant 56 : index
  // The range of the new loop.
  // CHECK:     %[[partial_range:.*]] = muli %[[orig_ub_i]], %[[orig_ub_j]]
  // CHECK-NEXT:%[[range:.*]] = muli %[[partial_range]], %[[orig_ub_k]]

  // Updated loop bounds.
  // CHECK: loop.for %[[i:.*]] = %[[orig_lb]] to %[[range]] step %[[orig_step]]
  loop.for %i = %c0 to %c42 step %c1 {
    // Inner loops must have been removed.
    // CHECK-NOT: loop.for

    // Reconstruct original IVs from the linearized one.
    // CHECK: %[[orig_k:.*]] = remis %[[i]], %[[orig_ub_k]]
    // CHECK: %[[div:.*]] = divis %[[i]], %[[orig_ub_k]]
    // CHECK: %[[orig_j:.*]] = remis %[[div]], %[[orig_ub_j]]
    // CHECK: %[[orig_i:.*]] = divis %[[div]], %[[orig_ub_j]]
    loop.for %j = %c0 to %c56 step %c1 {
      loop.for %k = %c0 to %c3 step %c1 {
        // CHECK: "use"(%[[orig_i]], %[[orig_j]], %[[orig_k]])
        "use"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}

func @unnormalized_loops() {
  // CHECK: %[[orig_step_i:.*]] = constant 2
  // CHECK: %[[orig_step_j:.*]] = constant 3
  // CHECK: %[[orig_lb_i:.*]] = constant 5
  // CHECK: %[[orig_lb_j:.*]] = constant 7
  // CHECK: %[[orig_ub_i:.*]] = constant 10
  // CHECK: %[[orig_ub_j:.*]] = constant 17
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c5 = constant 5 : index
  %c7 = constant 7 : index
  %c10 = constant 10 : index
  %c17 = constant 17 : index

  // Number of iterations in the outer loop.
  // CHECK: %[[diff_i:.*]] = subi %[[orig_ub_i]], %[[orig_lb_i]]
  // CHECK: %[[c1:.*]] = constant 1
  // CHECK: %[[step_minus_c1:.*]] = subi %[[orig_step_i]], %[[c1]]
  // CHECK: %[[dividend:.*]] = addi %[[diff_i]], %[[step_minus_c1]]
  // CHECK: %[[numiter_i:.*]] = divis %[[dividend]], %[[orig_step_i]]

  // Normalized lower bound and step for the outer loop.
  // CHECK: %[[lb_i:.*]] = constant 0
  // CHECK: %[[step_i:.*]] = constant 1

  // Number of iterations in the inner loop, the pattern is the same as above,
  // only capture the final result.
  // CHECK: %[[numiter_j:.*]] = divis {{.*}}, %[[orig_step_j]]

  // New bounds of the outer loop.
  // CHECK: %[[range:.*]] = muli %[[numiter_i]], %[[numiter_j]]
  // CHECK: loop.for %[[i:.*]] = %[[lb_i]] to %[[range]] step %[[step_i]]
  loop.for %i = %c5 to %c10 step %c2 {
    // The inner loop has been removed.
    // CHECK-NOT: loop.for
    loop.for %j = %c7 to %c17 step %c3 {
      // The IVs are rewritten.
      // CHECK: %[[normalized_j:.*]] = remis %[[i]], %[[numiter_j]]
      // CHECK: %[[normalized_i:.*]] = divis %[[i]], %[[numiter_j]]
      // CHECK: %[[scaled_j:.*]] = muli %[[normalized_j]], %[[orig_step_j]]
      // CHECK: %[[orig_j:.*]] = addi %[[scaled_j]], %[[orig_lb_j]]
      // CHECK: %[[scaled_i:.*]] = muli %[[normalized_i]], %[[orig_step_i]]
      // CHECK: %[[orig_i:.*]] = addi %[[scaled_i]], %[[orig_lb_i]]
      // CHECK: "use"(%[[orig_i]], %[[orig_j]])
      "use"(%i, %j) : (index, index) -> ()
    }
  }
  return
}

// Check with parametric loop bounds and steps, capture the bounds here.
// CHECK-LABEL: @parametric
// CHECK-SAME: %[[orig_lb1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[orig_ub1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[orig_step1:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[orig_lb2:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[orig_ub2:[A-Za-z0-9]+]]:
// CHECK-SAME: %[[orig_step2:[A-Za-z0-9]+]]:
func @parametric(%lb1 : index, %ub1 : index, %step1 : index,
                 %lb2 : index, %ub2 : index, %step2 : index) {
  // Compute the number of iterations for each of the loops and the total
  // number of iterations.
  // CHECK: %[[range1:.*]] = subi %[[orig_ub1]], %[[orig_lb1]]
  // CHECK: %[[orig_step1_minus_1:.*]] = subi %[[orig_step1]], %c1
  // CHECK: %[[dividend1:.*]] = addi %[[range1]], %[[orig_step1_minus_1]]
  // CHECK: %[[numiter1:.*]] = divis %[[dividend1]], %[[orig_step1]]
  // CHECK: %[[range2:.*]] = subi %[[orig_ub2]], %[[orig_lb2]]
  // CHECK: %[[orig_step2_minus_1:.*]] = subi %arg5, %c1
  // CHECK: %[[dividend2:.*]] = addi %[[range2]], %[[orig_step2_minus_1]]
  // CHECK: %[[numiter2:.*]] = divis %[[dividend2]], %[[orig_step2]]
  // CHECK: %[[range:.*]] = muli %[[numiter1]], %[[numiter2]] : index

  // Check that the outer loop is updated.
  // CHECK: loop.for %[[i:.*]] = %c0{{.*}} to %[[range]] step %c1
  loop.for %i = %lb1 to %ub1 step %step1 {
    // Check that the inner loop is removed.
    // CHECK-NOT: loop.for
    loop.for %j = %lb2 to %ub2 step %step2 {
      // Remapping of the induction variables.
      // CHECK: %[[normalized_j:.*]] = remis %[[i]], %[[numiter2]] : index
      // CHECK: %[[normalized_i:.*]] = divis %[[i]], %[[numiter2]] : index
      // CHECK: %[[scaled_j:.*]] = muli %[[normalized_j]], %[[orig_step2]]
      // CHECK: %[[orig_j:.*]] = addi %[[scaled_j]], %[[orig_lb2]]
      // CHECK: %[[scaled_i:.*]] = muli %[[normalized_i]], %[[orig_step1]]
      // CHECK: %[[orig_i:.*]] = addi %[[scaled_i]], %[[orig_lb1]]

      // CHECK: "foo"(%[[orig_i]], %[[orig_j]])
      "foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}

// CHECK-LABEL: @two_bands
func @two_bands() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  // CHECK: %[[outer_range:.*]] = muli
  // CHECK: loop.for %{{.*}} = %{{.*}} to %[[outer_range]]
  loop.for %i = %c0 to %c10 step %c1 {
    // Check that the "j" loop was removed and that the inner loops were
    // coalesced as well.  The preparation step for coalescing will inject the
    // subtraction operation unlike the IV remapping.
    // CHECK-NOT: loop.for
    // CHECK: subi
    loop.for %j = %c0 to %c10 step %c1 {
      // The inner pair of loops is coalesced separately.
      // CHECK: loop.for
      loop.for %k = %i to %j step %c1 {
        // CHECK_NOT: loop.for
        loop.for %l = %i to %j step %c1 {
          "foo"() : () -> ()
        }
      }
    }
  }
  return
}
