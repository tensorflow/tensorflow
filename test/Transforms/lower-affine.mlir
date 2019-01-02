// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop() {
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br ^bb1(%c1 : index)
// CHECK-NEXT: ^bb1(%0: index):	// 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   %c1_0 = constant 1 : index
// CHECK-NEXT:   %2 = addi %0, %c1_0 : index
// CHECK-NEXT:   br ^bb1(%2 : index)
// CHECK-NEXT: ^bb3:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @simple_loop() {
  for %i = 1 to 42 {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @pre(index) -> ()
func @body2(index, index) -> ()
func @post(index) -> ()

// CHECK-LABEL: func @imperfectly_nested_loops() {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br ^bb1(%c0 : index)
// CHECK-NEXT: ^bb1(%0: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @pre(%0) : (index) -> ()
// CHECK-NEXT:   %c7 = constant 7 : index
// CHECK-NEXT:   %c56 = constant 56 : index
// CHECK-NEXT:   br ^bb3(%c7 : index)
// CHECK-NEXT: ^bb3(%2: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %3 = cmpi "slt", %2, %c56 : index
// CHECK-NEXT:   cond_br %3, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%0, %2) : (index, index) -> ()
// CHECK-NEXT:   %c2 = constant 2 : index
// CHECK-NEXT:   %4 = addi %2, %c2 : index
// CHECK-NEXT:   br ^bb3(%4 : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   call @post(%0) : (index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %5 = addi %0, %c1 : index
// CHECK-NEXT:   br ^bb1(%5 : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @imperfectly_nested_loops() {
  for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @mid(index) -> ()
func @body3(index, index) -> ()

// CHECK-LABEL: func @more_imperfectly_nested_loops() {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br ^bb1(%c0 : index)
// CHECK-NEXT: ^bb1(%0: index):	// 2 preds: ^bb0, ^bb8
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, ^bb2, ^bb9
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @pre(%0) : (index) -> ()
// CHECK-NEXT:   %c7 = constant 7 : index
// CHECK-NEXT:   %c56 = constant 56 : index
// CHECK-NEXT:   br ^bb3(%c7 : index)
// CHECK-NEXT: ^bb3(%2: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %3 = cmpi "slt", %2, %c56 : index
// CHECK-NEXT:   cond_br %3, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%0, %2) : (index, index) -> ()
// CHECK-NEXT:   %c2 = constant 2 : index
// CHECK-NEXT:   %4 = addi %2, %c2 : index
// CHECK-NEXT:   br ^bb3(%4 : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   call @mid(%0) : (index) -> ()
// CHECK-NEXT:   %c18 = constant 18 : index
// CHECK-NEXT:   %c37 = constant 37 : index
// CHECK-NEXT:   br ^bb6(%c18 : index)
// CHECK-NEXT: ^bb6(%5: index):	// 2 preds: ^bb5, ^bb7
// CHECK-NEXT:   %6 = cmpi "slt", %5, %c37 : index
// CHECK-NEXT:   cond_br %6, ^bb7, ^bb8
// CHECK-NEXT: ^bb7:	// pred: ^bb6
// CHECK-NEXT:   call @body3(%0, %5) : (index, index) -> ()
// CHECK-NEXT:   %c3 = constant 3 : index
// CHECK-NEXT:   %7 = addi %5, %c3 : index
// CHECK-NEXT:   br ^bb6(%7 : index)
// CHECK-NEXT: ^bb8:	// pred: ^bb6
// CHECK-NEXT:   call @post(%0) : (index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %8 = addi %0, %c1 : index
// CHECK-NEXT:   br ^bb1(%8 : index)
// CHECK-NEXT: ^bb9:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
  for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @mid(%i) : (index) -> ()
    for %k = 18 to 37 step 3 {
      call @body3(%i, %k) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @affine_apply_loops_shorthand(%arg0: index) {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   br ^bb1(%c0 : index)
// CHECK-NEXT: ^bb1(%0: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %1 = cmpi "slt", %0, %arg0 : index
// CHECK-NEXT:   cond_br %1, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br ^bb3(%0 : index)
// CHECK-NEXT: ^bb3(%2: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %3 = cmpi "slt", %2, %c42 : index
// CHECK-NEXT:   cond_br %3, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%0, %2) : (index, index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %4 = addi %2, %c1 : index
// CHECK-NEXT:   br ^bb3(%4 : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   %c1_0 = constant 1 : index
// CHECK-NEXT:   %5 = addi %0, %c1_0 : index
// CHECK-NEXT:   br ^bb1(%5 : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @affine_apply_loops_shorthand(%N : index) {
  for %i = 0 to %N {
    for %j = %i to 42 {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @get_idx() -> (index)

#set1 = (d0) : (20 - d0 >= 0)
#set2 = (d0) : (d0 - 10 >= 0)

// CHECK-LABEL: func @if_only() {
// CHECK-NEXT:   %0 = call @get_idx() : () -> index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %1 = muli %0, %c-1 : index
// CHECK-NEXT:   %c20 = constant 20 : index
// CHECK-NEXT:   %2 = addi %1, %c20 : index
// CHECK-NEXT:   %3 = cmpi "sge", %2, %c0 : index
// CHECK-NEXT:   cond_br %3, [[thenBB:\^bb[0-9]+]], [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_only() {
  %i = call @get_idx() : () -> (index)
  if #set1(%i) {
    call @body(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_else() {
// CHECK-NEXT:   %0 = call @get_idx() : () -> index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %1 = muli %0, %c-1 : index
// CHECK-NEXT:   %c20 = constant 20 : index
// CHECK-NEXT:   %2 = addi %1, %c20 : index
// CHECK-NEXT:   %3 = cmpi "sge", %2, %c0 : index
// CHECK-NEXT:   cond_br %3, [[thenBB:\^bb[0-9]+]], [[elseBB:\^bb[0-9]+]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   br [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[elseBB]]:
// CHECK-NEXT:   call @mid(%0) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_else() {
  %i = call @get_idx() : () -> (index)
  if #set1(%i) {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @nested_ifs() {
// CHECK-NEXT:   %0 = call @get_idx() : () -> index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %1 = muli %0, %c-1 : index
// CHECK-NEXT:   %c20 = constant 20 : index
// CHECK-NEXT:   %2 = addi %1, %c20 : index
// CHECK-NEXT:   %3 = cmpi "sge", %2, %c0 : index
// CHECK-NEXT:   cond_br %3, ^bb1, ^bb4
// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   %c0_0 = constant 0 : index
// CHECK-NEXT:   %c-10 = constant -10 : index
// CHECK-NEXT:   %4 = addi %0, %c-10 : index
// CHECK-NEXT:   %5 = cmpi "sge", %4, %c0_0 : index
// CHECK-NEXT:   cond_br %5, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   br ^bb3
// CHECK-NEXT: ^bb3:	// 2 preds: ^bb1, ^bb2
// CHECK-NEXT:   br ^bb7
// CHECK-NEXT: ^bb4:	// pred: ^bb0
// CHECK-NEXT:   %c0_1 = constant 0 : index
// CHECK-NEXT:   %c-10_2 = constant -10 : index
// CHECK-NEXT:   %6 = addi %0, %c-10_2 : index
// CHECK-NEXT:   %7 = cmpi "sge", %6, %c0_1 : index
// CHECK-NEXT:   cond_br %7, ^bb5, ^bb6
// CHECK-NEXT: ^bb5:	// pred: ^bb4
// CHECK-NEXT:   call @mid(%0) : (index) -> ()
// CHECK-NEXT:   br ^bb6
// CHECK-NEXT: ^bb6:	// 2 preds: ^bb4, ^bb5
// CHECK-NEXT:   br ^bb7
// CHECK-NEXT: ^bb7:	// 2 preds: ^bb3, ^bb6
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @nested_ifs() {
  %i = call @get_idx() : () -> (index)
  if #set1(%i) {
    if #set2(%i) {
      call @body(%i) : (index) -> ()
    }
  } else {
    if #set2(%i) {
      call @mid(%i) : (index) -> ()
    }
  }
  return
}

#setN = (d0)[N,M,K,L] : (N - d0 + 1 >= 0, N - 1 >= 0, M - 1 >= 0, K - 1 >= 0, L - 42 == 0)

// CHECK-LABEL: func @multi_cond(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
// CHECK-NEXT:   %0 = call @get_idx() : () -> index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %1 = muli %0, %c-1 : index
// CHECK-NEXT:   %2 = addi %1, %arg0 : index
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %3 = addi %2, %c1 : index
// CHECK-NEXT:   %4 = cmpi "sge", %3, %c0 : index
// CHECK-NEXT:   cond_br %4, [[cond2BB:\^bb[0-9]+]], [[elseBB:\^bb[0-9]+]]
// CHECK-NEXT: [[cond2BB]]:
// CHECK-NEXT:   %c-1_0 = constant -1 : index
// CHECK-NEXT:   %5 = addi %arg0, %c-1_0 : index
// CHECK-NEXT:   %6 = cmpi "sge", %5, %c0 : index
// CHECK-NEXT:   cond_br %6, [[cond3BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond3BB]]:
// CHECK-NEXT:   %c-1_1 = constant -1 : index
// CHECK-NEXT:   %7 = addi %arg1, %c-1_1 : index
// CHECK-NEXT:   %8 = cmpi "sge", %7, %c0 : index
// CHECK-NEXT:   cond_br %8, [[cond4BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond4BB]]:
// CHECK-NEXT:   %c-1_2 = constant -1 : index
// CHECK-NEXT:   %9 = addi %arg2, %c-1_2 : index
// CHECK-NEXT:   %10 = cmpi "sge", %9, %c0 : index
// CHECK-NEXT:   cond_br %10, [[cond5BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond5BB]]:
// CHECK-NEXT:   %c-42 = constant -42 : index
// CHECK-NEXT:   %11 = addi %arg3, %c-42 : index
// CHECK-NEXT:   %12 = cmpi "eq", %11, %c0 : index
// CHECK-NEXT:   cond_br %12, [[thenBB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   br [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[elseBB]]:
// CHECK-NEXT:   call @mid(%0) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @multi_cond(%N : index, %M : index, %K : index, %L : index) {
  %i = call @get_idx() : () -> (index)
  if #setN(%i)[%N,%M,%K,%L] {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_for() {
func @if_for() {
// CHECK-NEXT:   %0 = call @get_idx() : () -> index
  %i = call @get_idx() : () -> (index)
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %1 = muli %0, %c-1 : index
// CHECK-NEXT:   %c20 = constant 20 : index
// CHECK-NEXT:   %2 = addi %1, %c20 : index
// CHECK-NEXT:   %3 = cmpi "sge", %2, %c0 : index
// CHECK-NEXT:   cond_br %3, [[midLoopInitBB:\^bb[0-9]+]], [[outerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[midLoopInitBB]]:
// CHECK-NEXT:   %c0_0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br [[midLoopCondBB:\^bb[0-9]+]](%c0_0 : index)
// CHECK-NEXT: [[midLoopCondBB]](%4: index):
// CHECK-NEXT:   %5 = cmpi "slt", %4, %c42 : index
// CHECK-NEXT:   cond_br %5, [[midLoopBodyBB:\^bb[0-9]+]], [[outerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[midLoopBodyBB]]:
// CHECK-NEXT:   %c0_1 = constant 0 : index
// CHECK-NEXT:   %c-10 = constant -10 : index
// CHECK-NEXT:   %6 = addi %4, %c-10 : index
// CHECK-NEXT:   %7 = cmpi "sge", %6, %c0_1 : index
// CHECK-NEXT:   cond_br %7, [[innerThenBB:\^bb[0-9]+]], [[innerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerThenBB:\^bb[0-9]+]]:
// CHECK-NEXT:   call @body2(%0, %4) : (index, index) -> ()
// CHECK-NEXT:   br [[innerEndBB]]
// CHECK-NEXT: [[innerEndBB]]:
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %8 = addi %4, %c1 : index
// CHECK-NEXT:   br [[midLoopCondBB]](%8 : index)
// CHECK-NEXT: [[outerEndBB]]:
// CHECK-NEXT:   br [[outerLoopInit:\^bb[0-9]+]]
  if #set1(%i) {
    for %j = 0 to 42 {
      if #set2(%j) {
        call @body2(%i, %j) : (index, index) -> ()
      }
    }
  }
// CHECK-NEXT: [[outerLoopInit]]:
// CHECK-NEXT:   %c0_2 = constant 0 : index
// CHECK-NEXT:   %c42_3 = constant 42 : index
// CHECK-NEXT:   br [[outerLoopCond:\^bb[0-9]+]](%c0_2 : index)
// CHECK-NEXT: [[outerLoopCond]](%9: index):
// CHECK-NEXT:   %10 = cmpi "slt", %9, %c42_3 : index
// CHECK-NEXT:   cond_br %10, [[outerLoopBody:\^bb[0-9]+]], [[outerLoopEnd:\^bb[0-9]+]]
// CHECK-NEXT: [[outerLoopBody]]:
// CHECK-NEXT:   %c0_4 = constant 0 : index
// CHECK-NEXT:   %c-10_5 = constant -10 : index
// CHECK-NEXT:   %11 = addi %9, %c-10_5 : index
// CHECK-NEXT:   %12 = cmpi "sge", %11, %c0_4 : index
// CHECK-NEXT:   cond_br %12, [[innerLoopInitBB:\^bb[0-9]+]], [[midEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerLoopInitBB:\^bb[0-9]+]]:
// CHECK-NEXT:   %c0_6 = constant 0 : index
// CHECK-NEXT:   %c42_7 = constant 42 : index
// CHECK-NEXT:   br [[innerLoopCondBB:\^bb[0-9]+]](%c0_6 : index)
// CHECK-NEXT: [[innerLoopCondBB]](%13: index):
// CHECK-NEXT:   %14 = cmpi "slt", %13, %c42_7 : index
// CHECK-NEXT:   cond_br %14, [[innerLoopBodyBB:\^bb[0-9]+]], [[innerLoopEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerLoopBodyBB]]:
// CHECK-NEXT:   call @body3(%9, %13) : (index, index) -> ()
// CHECK-NEXT:   %c1_8 = constant 1 : index
// CHECK-NEXT:   %15 = addi %13, %c1_8 : index
// CHECK-NEXT:   br [[innerLoopCondBB]](%15 : index)
// CHECK-NEXT: [[innerLoopEndBB]]:
// CHECK-NEXT:   br [[midEndBB]]
// CHECK-NEXT: [[midEndBB]]:
// CHECK-NEXT:   %c1_9 = constant 1 : index
// CHECK-NEXT:   %16 = addi %9, %c1_9 : index
// CHECK-NEXT:   br [[outerLoopCond]](%16 : index)
  for %k = 0 to 42 {
    if #set2(%k) {
      for %l = 0 to 42 {
        call @body3(%k, %l) : (index, index) -> ()
      }
    }
  }
// CHECK-NEXT: [[outerLoopEnd]]:
// CHECK-NEXT:   return
  return
}

#lbMultiMap = (d0)[s0] -> (d0, s0 - d0)
#ubMultiMap = (d0)[s0] -> (s0, d0 + 10)

// CHECK-LABEL: func @loop_min_max(%arg0: index) {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br ^bb1(%c0 : index)
// CHECK-NEXT: ^bb1(%{{[0-9]+}}: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %{{[0-9]+}}, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %2 = muli %0, %c-1 : index
// CHECK-NEXT:   %3 = addi %2, %arg0 : index
// CHECK-NEXT:   %4 = cmpi "sgt", %0, %3 : index
// CHECK-NEXT:   %5 = select %4, %0, %3 : index
// CHECK-NEXT:   %c10 = constant 10 : index
// CHECK-NEXT:   %6 = addi %0, %c10 : index
// CHECK-NEXT:   %7 = cmpi "slt", %arg0, %6 : index
// CHECK-NEXT:   %8 = select %7, %arg0, %6 : index
// CHECK-NEXT:   br ^bb3(%5 : index)
// CHECK-NEXT: ^bb3(%9: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %10 = cmpi "slt", %9, %8 : index
// CHECK-NEXT:   cond_br %10, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%0, %9) : (index, index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %11 = addi %9, %c1 : index
// CHECK-NEXT:   br ^bb3(%11 : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   %c1_0 = constant 1 : index
// CHECK-NEXT:   %12 = addi %0, %c1_0 : index
// CHECK-NEXT:   br ^bb1(%12 : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @loop_min_max(%N : index) {
  for %i = 0 to 42 {
    for %j = max #lbMultiMap(%i)[%N] to min #ubMultiMap(%i)[%N] {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map_7_values = (i) -> (i, i, i, i, i, i, i)

// Check that the "min" (cmpi "slt" + select) reduction sequence is emitted
// correctly for a an affine map with 7 results.

// CHECK-LABEL: func @min_reduction_tree(%arg0: index) {
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %[[c01:.+]] = cmpi "slt", %arg0, %arg0 : index
// CHECK-NEXT:   %[[r01:.+]] = select %[[c01]], %arg0, %arg0 : index
// CHECK-NEXT:   %[[c012:.+]] = cmpi "slt", %[[r01]], %arg0 : index
// CHECK-NEXT:   %[[r012:.+]] = select %[[c012]], %[[r01]], %arg0 : index
// CHECK-NEXT:   %[[c0123:.+]] = cmpi "slt", %[[r012]], %arg0 : index
// CHECK-NEXT:   %[[r0123:.+]] = select %[[c0123]], %[[r012]], %arg0 : index
// CHECK-NEXT:   %[[c01234:.+]] = cmpi "slt", %[[r0123]], %arg0 : index
// CHECK-NEXT:   %[[r01234:.+]] = select %[[c01234]], %[[r0123]], %arg0 : index
// CHECK-NEXT:   %[[c012345:.+]] = cmpi "slt", %[[r01234]], %arg0 : index
// CHECK-NEXT:   %[[r012345:.+]] = select %[[c012345]], %[[r01234]], %arg0 : index
// CHECK-NEXT:   %[[c0123456:.+]] = cmpi "slt", %[[r012345]], %arg0 : index
// CHECK-NEXT:   %[[r0123456:.+]] = select %[[c0123456]], %[[r012345]], %arg0 : index
// CHECK-NEXT:   br ^bb1(%c0 : index)
// CHECK-NEXT: ^bb1(%{{[0-9]+}}: index):	// 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %{{[0-9]+}} = cmpi "slt", %{{[0-9]+}}, %[[r0123456]] : index
// CHECK-NEXT:   cond_br %{{[0-9]+}}, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%{{[0-9]+}}) : (index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %14 = addi %12, %c1 : index
// CHECK-NEXT:   br ^bb1(%{{[0-9]+}} : index)
// CHECK-NEXT: ^bb3:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @min_reduction_tree(%v : index) {
  for %i = 0 to min #map_7_values(%v)[] {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

#map0 = () -> (0)
#map1 = ()[s0] -> (s0)
#map2 = (d0) -> (d0)
#map3 = (d0)[s0] -> (d0 + s0 + 1)
#map4 = (d0,d1,d2,d3)[s0,s1,s2] -> (d0 + 2*d1 + 3*d2 + 4*d3 + 5*s0 + 6*s1 + 7*s2)
#map5 = (d0,d1,d2) -> (d0,d1,d2)
#map6 = (d0,d1,d2) -> (d0 + d1 + d2)

// CHECK-LABEL: func @affine_applies()
func @affine_applies() {
^bb0:
// CHECK: %c0 = constant 0 : index
  %zero = affine_apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %c101 = constant 101 : index
  %101 = constant 101 : index
  %symbZero = affine_apply #map1()[%zero]
// CHECK-NEXT: %c102 = constant 102 : index
  %102 = constant 102 : index
  %copy = affine_apply #map2(%zero)

// CHECK-NEXT: %0 = addi %c0, %c0 : index
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %1 = addi %0, %c1 : index
  %one = affine_apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %c103 = constant 103 : index
// CHECK-NEXT: %c104 = constant 104 : index
// CHECK-NEXT: %c105 = constant 105 : index
// CHECK-NEXT: %c106 = constant 106 : index
// CHECK-NEXT: %c107 = constant 107 : index
// CHECK-NEXT: %c108 = constant 108 : index
// CHECK-NEXT: %c109 = constant 109 : index
  %103 = constant 103 : index
  %104 = constant 104 : index
  %105 = constant 105 : index
  %106 = constant 106 : index
  %107 = constant 107 : index
  %108 = constant 108 : index
  %109 = constant 109 : index
// CHECK-NEXT: %c2 = constant 2 : index
// CHECK-NEXT: %2 = muli %c104, %c2 : index
// CHECK-NEXT: %3 = addi %c103, %2 : index
// CHECK-NEXT: %c3 = constant 3 : index
// CHECK-NEXT: %4 = muli %c105, %c3 : index
// CHECK-NEXT: %5 = addi %3, %4 : index
// CHECK-NEXT: %c4 = constant 4 : index
// CHECK-NEXT: %6 = muli %c106, %c4 : index
// CHECK-NEXT: %7 = addi %5, %6 : index
// CHECK-NEXT: %c5 = constant 5 : index
// CHECK-NEXT: %8 = muli %c107, %c5 : index
// CHECK-NEXT: %9 = addi %7, %8 : index
// CHECK-NEXT: %c6 = constant 6 : index
// CHECK-NEXT: %10 = muli %c108, %c6 : index
// CHECK-NEXT: %11 = addi %9, %10 : index
// CHECK-NEXT: %c7 = constant 7 : index
// CHECK-NEXT: %12 = muli %c109, %c7 : index
// CHECK-NEXT: %13 = addi %11, %12 : index
  %four = affine_apply #map4(%103,%104,%105,%106)[%107,%108,%109]
  return
}

// CHECK-LABEL: func @multiresult_affine_apply()
func @multiresult_affine_apply() {
// CHECK-NEXT: %c1 = constant 1 : index
// CHECK-NEXT: %0 = addi %c1, %c1 : index
// CHECK-NEXT: %1 = addi %0, %c1 : index
  %one = constant 1 : index
  %tuple = affine_apply #map5 (%one, %one, %one)
  %three = affine_apply #map6 (%tuple#0, %tuple#1, %tuple#2)
  return
}

// CHECK-LABEL: func @args_ret_affine_apply(%arg0: index, %arg1: index)
func @args_ret_affine_apply(index, index) -> (index, index) {
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %arg0, %arg1 : index, index
  %00 = affine_apply #map2 (%0)
  %11 = affine_apply #map1 ()[%1]
  return %00, %11 : index, index
}
