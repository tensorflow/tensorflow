// RUN: mlir-opt -lower-affine %s | FileCheck %s

// CHECK-LABEL: func @empty() {
func @empty() {
  return     // CHECK:  return
}            // CHECK: }

func @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: func @simple_loop() {
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{.*}}: index):	// 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb3:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @simple_loop() {
  affine.for %i = 1 to 42 {
    call @body(%i) : (index) -> ()
  }
  return
}

/////////////////////////////////////////////////////////////////////

func @pre(index) -> ()
func @body2(index, index) -> ()
func @post(index) -> ()

// CHECK-LABEL: func @imperfectly_nested_loops() {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{.*}}: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 7 : index
// CHECK-NEXT:   %{{.*}} = constant 56 : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb3(%{{.*}}: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 2 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
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
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{.*}}: index):	// 2 preds: ^bb0, ^bb8
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb9
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @pre(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 7 : index
// CHECK-NEXT:   %{{.*}} = constant 56 : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb3(%{{.*}}: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 2 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 18 : index
// CHECK-NEXT:   %{{.*}} = constant 37 : index
// CHECK-NEXT:   br ^bb6(%{{.*}} : index)
// CHECK-NEXT: ^bb6(%{{.*}}: index):	// 2 preds: ^bb5, ^bb7
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb7, ^bb8
// CHECK-NEXT: ^bb7:	// pred: ^bb6
// CHECK-NEXT:   call @body3(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 3 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb6(%{{.*}} : index)
// CHECK-NEXT: ^bb8:	// pred: ^bb6
// CHECK-NEXT:   call @post(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb9:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @more_imperfectly_nested_loops() {
  affine.for %i = 0 to 42 {
    call @pre(%i) : (index) -> ()
    affine.for %j = 7 to 56 step 2 {
      call @body2(%i, %j) : (index, index) -> ()
    }
    call @mid(%i) : (index) -> ()
    affine.for %k = 18 to 37 step 3 {
      call @body3(%i, %k) : (index, index) -> ()
    }
    call @post(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @affine_apply_loops_shorthand(%{{.*}}: index) {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{.*}}: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb3(%{{.*}}: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @affine_apply_loops_shorthand(%N : index) {
  affine.for %i = 0 to %N {
    affine.for %j = (d0)[]->(d0)(%i)[] to 42 {
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
// CHECK-NEXT:   %{{.*}} = call @get_idx() : () -> index
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = constant 20 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[thenBB:\^bb[0-9]+]], [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_only() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_else() {
// CHECK-NEXT:   %{{.*}} = call @get_idx() : () -> index
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = constant 20 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[thenBB:\^bb[0-9]+]], [[elseBB:\^bb[0-9]+]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[elseBB]]:
// CHECK-NEXT:   call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @if_else() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @nested_ifs() {
// CHECK-NEXT:   %{{.*}} = call @get_idx() : () -> index
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = constant 20 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb4
// CHECK-NEXT: ^bb1:	// pred: ^bb0
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-10 = constant -10 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-10 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br ^bb3
// CHECK-NEXT: ^bb3:	// 2 preds: ^bb1, ^bb2
// CHECK-NEXT:   br ^bb7
// CHECK-NEXT: ^bb4:	// pred: ^bb0
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-10_2 = constant -10 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-10_2 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb5, ^bb6
// CHECK-NEXT: ^bb5:	// pred: ^bb4
// CHECK-NEXT:   call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br ^bb6
// CHECK-NEXT: ^bb6:	// 2 preds: ^bb4, ^bb5
// CHECK-NEXT:   br ^bb7
// CHECK-NEXT: ^bb7:	// 2 preds: ^bb3, ^bb6
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @nested_ifs() {
  %i = call @get_idx() : () -> (index)
  affine.if #set1(%i) {
    affine.if #set2(%i) {
      call @body(%i) : (index) -> ()
    }
  } else {
    affine.if #set2(%i) {
      call @mid(%i) : (index) -> ()
    }
  }
  return
}

#setN = (d0)[N,M,K,L] : (N - d0 + 1 >= 0, N - 1 >= 0, M - 1 >= 0, K - 1 >= 0, L - 42 == 0)

// CHECK-LABEL: func @multi_cond(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
// CHECK-NEXT:   %{{.*}} = call @get_idx() : () -> index
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[cond2BB:\^bb[0-9]+]], [[elseBB:\^bb[0-9]+]]
// CHECK-NEXT: [[cond2BB]]:
// CHECK-NEXT:   %{{.*}}-1_0 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-1_0 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[cond3BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond3BB]]:
// CHECK-NEXT:   %{{.*}}-1_1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-1_1 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[cond4BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond4BB]]:
// CHECK-NEXT:   %{{.*}}-1_2 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-1_2 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[cond5BB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[cond5BB]]:
// CHECK-NEXT:   %{{.*}}-42 = constant -42 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-42 : index
// CHECK-NEXT:   %{{.*}} = cmpi "eq", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[thenBB:\^bb[0-9]+]], [[elseBB]]
// CHECK-NEXT: [[thenBB]]:
// CHECK-NEXT:   call @body(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br [[endBB:\^bb[0-9]+]]
// CHECK-NEXT: [[elseBB]]:
// CHECK-NEXT:   call @mid(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   br [[endBB]]
// CHECK-NEXT: [[endBB]]:
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @multi_cond(%N : index, %M : index, %K : index, %L : index) {
  %i = call @get_idx() : () -> (index)
  affine.if #setN(%i)[%N,%M,%K,%L] {
    call @body(%i) : (index) -> ()
  } else {
    call @mid(%i) : (index) -> ()
  }
  return
}

// CHECK-LABEL: func @if_for() {
func @if_for() {
// CHECK-NEXT:   %{{.*}} = call @get_idx() : () -> index
  %i = call @get_idx() : () -> (index)
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = constant 20 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[midLoopInitBB:\^bb[0-9]+]], [[outerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[midLoopInitBB]]:
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br [[midLoopCondBB:\^bb[0-9]+]](%{{.*}} : index)
// CHECK-NEXT: [[midLoopCondBB]](%{{.*}}: index):
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[midLoopBodyBB:\^bb[0-9]+]], [[outerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[midLoopBodyBB]]:
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-10 = constant -10 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-10 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[innerThenBB:\^bb[0-9]+]], [[innerEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerThenBB:\^bb[0-9]+]]:
// CHECK-NEXT:   call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   br [[innerEndBB]]
// CHECK-NEXT: [[innerEndBB]]:
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br [[midLoopCondBB]](%{{.*}} : index)
// CHECK-NEXT: [[outerEndBB]]:
// CHECK-NEXT:   br [[outerLoopInit:\^bb[0-9]+]]
  affine.if #set1(%i) {
    affine.for %j = 0 to 42 {
      affine.if #set2(%j) {
        call @body2(%i, %j) : (index, index) -> ()
      }
    }
  }
// CHECK-NEXT: [[outerLoopInit]]:
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br [[outerLoopCond:\^bb[0-9]+]](%{{.*}} : index)
// CHECK-NEXT: [[outerLoopCond]](%{{.*}}: index):
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[outerLoopBody:\^bb[0-9]+]], [[outerLoopEnd:\^bb[0-9]+]]
// CHECK-NEXT: [[outerLoopBody]]:
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}}-10_5 = constant -10 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}}-10_5 : index
// CHECK-NEXT:   %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[innerLoopInitBB:\^bb[0-9]+]], [[midEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerLoopInitBB:\^bb[0-9]+]]:
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br [[innerLoopCondBB:\^bb[0-9]+]](%{{.*}} : index)
// CHECK-NEXT: [[innerLoopCondBB]](%{{.*}}: index):
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, [[innerLoopBodyBB:\^bb[0-9]+]], [[innerLoopEndBB:\^bb[0-9]+]]
// CHECK-NEXT: [[innerLoopBodyBB]]:
// CHECK-NEXT:   call @body3(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br [[innerLoopCondBB]](%{{.*}} : index)
// CHECK-NEXT: [[innerLoopEndBB]]:
// CHECK-NEXT:   br [[midEndBB]]
// CHECK-NEXT: [[midEndBB]]:
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br [[outerLoopCond]](%{{.*}} : index)
  affine.for %k = 0 to 42 {
    affine.if #set2(%k) {
      affine.for %l = 0 to 42 {
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

// CHECK-LABEL: func @loop_min_max(%{{.*}}: index) {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %{{.*}} = constant 42 : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{[0-9]+}}: index):	// 2 preds: ^bb0, ^bb5
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{[0-9]+}}, ^bb2, ^bb6
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:   %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "sgt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = constant 10 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb3(%{{.*}}: index):	// 2 preds: ^bb2, ^bb4
// CHECK-NEXT:   %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   cond_br %{{.*}}, ^bb4, ^bb5
// CHECK-NEXT: ^bb4:	// pred: ^bb3
// CHECK-NEXT:   call @body2(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb3(%{{.*}} : index)
// CHECK-NEXT: ^bb5:	// pred: ^bb3
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb6:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @loop_min_max(%N : index) {
  affine.for %i = 0 to 42 {
    affine.for %j = max #lbMultiMap(%i)[%N] to min #ubMultiMap(%i)[%N] {
      call @body2(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map_7_values = (i) -> (i, i, i, i, i, i, i)

// Check that the "min" (cmpi "slt" + select) reduction sequence is emitted
// correctly for a an affine map with 7 results.

// CHECK-LABEL: func @min_reduction_tree(%{{.*}}: index) {
// CHECK-NEXT:   %{{.*}} = constant 0 : index
// CHECK-NEXT:   %[[c01:.+]] = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[r01:.+]] = select %[[c01]], %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   %[[c012:.+]] = cmpi "slt", %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[r012:.+]] = select %[[c012]], %[[r01]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123:.+]] = cmpi "slt", %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123:.+]] = select %[[c0123]], %[[r012]], %{{.*}} : index
// CHECK-NEXT:   %[[c01234:.+]] = cmpi "slt", %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[r01234:.+]] = select %[[c01234]], %[[r0123]], %{{.*}} : index
// CHECK-NEXT:   %[[c012345:.+]] = cmpi "slt", %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[r012345:.+]] = select %[[c012345]], %[[r01234]], %{{.*}} : index
// CHECK-NEXT:   %[[c0123456:.+]] = cmpi "slt", %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   %[[r0123456:.+]] = select %[[c0123456]], %[[r012345]], %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{.*}} : index)
// CHECK-NEXT: ^bb1(%{{[0-9]+}}: index):	// 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %{{[0-9]+}} = cmpi "slt", %{{[0-9]+}}, %[[r0123456]] : index
// CHECK-NEXT:   cond_br %{{[0-9]+}}, ^bb2, ^bb3
// CHECK-NEXT: ^bb2:	// pred: ^bb1
// CHECK-NEXT:   call @body(%{{[0-9]+}}) : (index) -> ()
// CHECK-NEXT:   %{{.*}} = constant 1 : index
// CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:   br ^bb1(%{{[0-9]+}} : index)
// CHECK-NEXT: ^bb3:	// pred: ^bb1
// CHECK-NEXT:   return
// CHECK-NEXT: }
func @min_reduction_tree(%v : index) {
  affine.for %i = 0 to min #map_7_values(%v)[] {
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
// CHECK: %{{.*}} = constant 0 : index
  %zero = affine.apply #map0()

// Identity maps are just discarded.
// CHECK-NEXT: %{{.*}} = constant 101 : index
  %101 = constant 101 : index
  %symbZero = affine.apply #map1()[%zero]
// CHECK-NEXT: %{{.*}} = constant 102 : index
  %102 = constant 102 : index
  %copy = affine.apply #map2(%zero)

// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 1 : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
  %one = affine.apply #map3(%symbZero)[%zero]

// CHECK-NEXT: %{{.*}} = constant 103 : index
// CHECK-NEXT: %{{.*}} = constant 104 : index
// CHECK-NEXT: %{{.*}} = constant 105 : index
// CHECK-NEXT: %{{.*}} = constant 106 : index
// CHECK-NEXT: %{{.*}} = constant 107 : index
// CHECK-NEXT: %{{.*}} = constant 108 : index
// CHECK-NEXT: %{{.*}} = constant 109 : index
  %103 = constant 103 : index
  %104 = constant 104 : index
  %105 = constant 105 : index
  %106 = constant 106 : index
  %107 = constant 107 : index
  %108 = constant 108 : index
  %109 = constant 109 : index
// CHECK-NEXT: %{{.*}} = constant 2 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 3 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 4 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 5 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 6 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 7 : index
// CHECK-NEXT: %{{.*}} = muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
  %four = affine.apply #map4(%103,%104,%105,%106)[%107,%108,%109]
  return
}

// CHECK-LABEL: func @args_ret_affine_apply(%{{.*}}: index, %{{.*}}: index)
func @args_ret_affine_apply(index, index) -> (index, index) {
^bb0(%0 : index, %1 : index):
// CHECK-NEXT: return %{{.*}}, %{{.*}} : index, index
  %00 = affine.apply #map2 (%0)
  %11 = affine.apply #map1 ()[%1]
  return %00, %11 : index, index
}

//===---------------------------------------------------------------------===//
// Test lowering of Euclidean (floor) division, ceil division and modulo
// operation used in affine expressions.  In addition to testing the
// operation-level output, check that the obtained results are correct by
// applying constant folding transformation after affine lowering.
//===---------------------------------------------------------------------===//

#mapmod = (i) -> (i mod 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_mod
func @affine_apply_mod(%arg0 : index) -> (index) {
// CHECK-NEXT: %{{.*}} = constant 42 : index
// CHECK-NEXT: %{{.*}} = remis %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = constant 0 : index
// CHECK-NEXT: %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
  %0 = affine.apply #mapmod (%arg0)
  return %0 : index
}

#mapfloordiv = (i) -> (i floordiv 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_floordiv
func @affine_apply_floordiv(%arg0 : index) -> (index) {
// CHECK-NEXT: %{{.*}} = constant 42 : index
// CHECK-NEXT: %{{.*}} = constant 0 : index
// CHECK-NEXT: %{{.*}}-1 = constant -1 : index
// CHECK-NEXT: %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = subi %{{.*}}-1, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = divis %{{.*}}, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = subi %{{.*}}-1, %{{.*}} : index
// CHECK-NEXT: %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
  %0 = affine.apply #mapfloordiv (%arg0)
  return %0 : index
}

#mapceildiv = (i) -> (i ceildiv 42)

// --------------------------------------------------------------------------//
// IMPORTANT NOTE: if you change this test, also change the @lowered_affine_mod
// test in the "constant-fold.mlir" test to reflect the expected output of
// affine.apply lowering.
// --------------------------------------------------------------------------//
// CHECK-LABEL: func @affine_apply_ceildiv
func @affine_apply_ceildiv(%arg0 : index) -> (index) {
// CHECK-NEXT:  %{{.*}} = constant 42 : index
// CHECK-NEXT:  %{{.*}} = constant 0 : index
// CHECK-NEXT:  %{{.*}} = constant 1 : index
// CHECK-NEXT:  %{{.*}} = cmpi "sle", %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = divis %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = subi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
  %0 = affine.apply #mapceildiv (%arg0)
  return %0 : index
}

// CHECK-LABEL: func @affine_load
func @affine_load(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.load %0[%i0 + symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = constant 7 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = load %{{.*}}[%{{.*}}] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_store
func @affine_store(%arg0 : index) {
  %0 = alloc() : memref<10xf32>
  %1 = constant 11.0 : f32 
  affine.for %i0 = 0 to 10 {
    affine.store %1, %0[%i0 - symbol(%arg0) + 7] : memref<10xf32>
  }
// CHECK:       %{{.*}}-1 = constant -1 : index
// CHECK-NEXT:  %{{.*}} = muli %{{.*}}, %{{.*}}-1 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = constant 7 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  return
}

// CHECK-LABEL: func @affine_dma_start
func @affine_dma_start(%arg0 : index) {
  %0 = alloc() : memref<100xf32>
  %1 = alloc() : memref<100xf32, 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_start %0[%i0 + 7], %1[%arg0 + 11], %2[%c0], %c64
        : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  }
// CHECK:       %{{.*}} = constant 7 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = constant 11 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}[%{{.*}}] : memref<100xf32>, memref<100xf32, 2>, memref<1xi32>
  return
}

// CHECK-LABEL: func @affine_dma_wait
func @affine_dma_wait(%arg0 : index) {
  %2 = alloc() : memref<1xi32>
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.dma_wait %2[%i0 + %arg0 + 17], %c64 : memref<1xi32>
  }
// CHECK:       %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  %{{.*}} = constant 17 : index
// CHECK-NEXT:  %{{.*}} = addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:  dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
  return
}
