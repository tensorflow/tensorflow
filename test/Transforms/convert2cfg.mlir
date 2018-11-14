// RUN: mlir-opt -convert-to-cfg %s | FileCheck %s

// CHECK-LABEL: cfgfunc @empty() {
mlfunc @empty() {
             // CHECK: bb0:
  return     // CHECK:  return
}            // CHECK: }

extfunc @body(index) -> ()

// Simple loops are properly converted.
// CHECK-LABEL: cfgfunc @simple_loop() {
// CHECK-NEXT: bb0:
// CHECK-NEXT:   br bb1
// CHECK-NEXT: bb1:	// pred: bb0
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br bb2(%c1 : index)
// CHECK-NEXT: bb2(%0: index):	// 2 preds: bb1, bb3
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, bb3, bb4
// CHECK-NEXT: bb3:	// pred: bb2
// CHECK-NEXT:   call @body(%0) : (index) -> ()
// CHECK-NEXT:   %c1_0 = constant 1 : index
// CHECK-NEXT:   %2 = addi %0, %c1_0 : index
// CHECK-NEXT:   br bb2(%2 : index)
// CHECK-NEXT: bb4:	// pred: bb2
// CHECK-NEXT:   return
// CHECK-NEXT: }
mlfunc @simple_loop() {
  for %i = 1 to 42 {
    call @body(%i) : (index) -> ()
  }
  return
}

// Direct calls get renamed if asked (IR data structures properly updated) and
// keep the same name otherwise.
cfgfunc @simple_caller() {
bb0:
// CHECK: call @simple_loop() : () -> ()
  call @simple_loop() : () -> ()
  return
}

// Constant loads get renamed if asked (IR data structure properly updated) and
// keep the same name otherwise.
cfgfunc @simple_indirect_caller() {
bb0:
// CHECK: %f = constant @simple_loop : () -> ()
  %f = constant @simple_loop : () -> ()
  call_indirect %f() : () -> ()
  return
}

cfgfunc @nested_attributes() {
bb0:
  %0 = constant 0 : index
// CHECK: call @body(%c0) {attr1: [@simple_loop : () -> (), @simple_loop : () -> ()]} : (index) -> ()
  call @body(%0) {attr1: [@simple_loop : () -> (), @simple_loop : () -> ()]} : (index) -> ()
// Note: the {{\[}} construct is necessary to prevent FileCheck from
// interpreting [[ as the start of its variable in the pattern below.
// CHECK: call @body(%c0) {attr2: {{\[}}{{\[}}{{\[}}@simple_loop : () -> ()]], [@simple_loop : () -> ()]]} : (index) -> ()
  call @body(%0) {attr2: [[[@simple_loop : () -> ()]], [@simple_loop : () -> ()]]} : (index) -> ()
  return
}

// CHECK-LABEL: cfgfunc @ml_caller() {
mlfunc @ml_caller() {
// Direct calls inside ML functions are renamed if asked (given that the
// function itself is also converted).
// CHECK: call @simple_loop() : () -> ()
  call @simple_loop() : () -> ()
// Direct calls to not yet declared ML functions are also renamed.
// CHECK: call @more_imperfectly_nested_loops() : () -> ()
  call @more_imperfectly_nested_loops() : () -> ()
  return
}

/////////////////////////////////////////////////////////////////////

extfunc @body_args(index) -> (index)
extfunc @other(index, i32) -> (i32)

// Arguments and return values of the functions are converted.
// CHECK-LABEL: cfgfunc @mlfunc_args(i32, i32) -> (i32, i32) {
// CHECK-NEXT: bb0(%arg0: i32, %arg1: i32):
// CHECK-NEXT:   %c0_i32 = constant 0 : i32
// CHECK-NEXT:   br bb1
// CHECK-NEXT: bb1:	// pred: bb0
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br bb2(%c0 : index)
// CHECK-NEXT: bb2(%0: index):	// 2 preds: bb1, bb3
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, bb3, bb4
// CHECK-NEXT: bb3:	// pred: bb2
// CHECK-NEXT:   %2 = call @body_args(%0) : (index) -> index
// CHECK-NEXT:   %3 = call @other(%2, %arg0) : (index, i32) -> i32
// CHECK-NEXT:   %4 = call @other(%2, %3) : (index, i32) -> i32
// CHECK-NEXT:   %5 = call @other(%2, %arg1) : (index, i32) -> i32
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %6 = addi %0, %c1 : index
// CHECK-NEXT:   br bb2(%6 : index)
// CHECK-NEXT: bb4:	// pred: bb2
// CHECK-NEXT:   %c0_0 = constant 0 : index
// CHECK-NEXT:   %7 = call @other(%c0_0, %c0_i32) : (index, i32) -> i32
// CHECK-NEXT:   return %c0_i32, %7 : i32, i32
// CHECK-NEXT: }
mlfunc @mlfunc_args(%a : i32, %b : i32) -> (i32, i32) {
  %r1 = constant 0 : i32
  for %i = 0 to 42 {
    %1 = call @body_args(%i) : (index) -> (index)
    %2 = call @other(%1, %a) : (index, i32) -> (i32)
    %3 = call @other(%1, %2) : (index, i32) -> (i32)
    %4 = call @other(%1, %b) : (index, i32) -> (i32)
  }
  %ri = constant 0 : index
  %r2 = call @other(%ri, %r1) : (index, i32) -> (i32)
  return %r1, %r2 : i32, i32
}

/////////////////////////////////////////////////////////////////////

extfunc @pre(index) -> ()
extfunc @body2(index, index) -> ()
extfunc @post(index) -> ()

// CHECK-LABEL: cfgfunc @imperfectly_nested_loops() {
// CHECK-NEXT: bb0:
// CHECK-NEXT:   br bb1
// CHECK-NEXT: bb1:	// pred: bb0
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br bb2(%c0 : index)
// CHECK-NEXT: bb2(%0: index):	// 2 preds: bb1, bb7
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, bb3, bb8
// CHECK-NEXT: bb3:	// pred: bb2
// CHECK-NEXT:   call @pre(%0) : (index) -> ()
// CHECK-NEXT:   br bb4
// CHECK-NEXT: bb4:	// pred: bb3
// CHECK-NEXT:   %c7 = constant 7 : index
// CHECK-NEXT:   %c56 = constant 56 : index
// CHECK-NEXT:   br bb5(%c7 : index)
// CHECK-NEXT: bb5(%2: index):	// 2 preds: bb4, bb6
// CHECK-NEXT:   %3 = cmpi "slt", %2, %c56 : index
// CHECK-NEXT:   cond_br %3, bb6, bb7
// CHECK-NEXT: bb6:	// pred: bb5
// CHECK-NEXT:   call @body2(%0, %2) : (index, index) -> ()
// CHECK-NEXT:   %c2 = constant 2 : index
// CHECK-NEXT:   %4 = addi %2, %c2 : index
// CHECK-NEXT:   br bb5(%4 : index)
// CHECK-NEXT: bb7:	// pred: bb5
// CHECK-NEXT:   call @post(%0) : (index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %5 = addi %0, %c1 : index
// CHECK-NEXT:   br bb2(%5 : index)
// CHECK-NEXT: bb8:	// pred: bb2
// CHECK-NEXT:   return
// CHECK-NEXT: }
mlfunc @imperfectly_nested_loops() {
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

extfunc @mid(index) -> ()
extfunc @body3(index, index) -> ()

// CHECK-LABEL: cfgfunc @more_imperfectly_nested_loops() {
// CHECK-NEXT: bb0:
// CHECK-NEXT:   br bb1
// CHECK-NEXT: bb1:	// pred: bb0
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c42 = constant 42 : index
// CHECK-NEXT:   br bb2(%c0 : index)
// CHECK-NEXT: bb2(%0: index):	// 2 preds: bb1, bb11
// CHECK-NEXT:   %1 = cmpi "slt", %0, %c42 : index
// CHECK-NEXT:   cond_br %1, bb3, bb12
// CHECK-NEXT: bb3:	// pred: bb2
// CHECK-NEXT:   call @pre(%0) : (index) -> ()
// CHECK-NEXT:   br bb4
// CHECK-NEXT: bb4:	// pred: bb3
// CHECK-NEXT:   %c7 = constant 7 : index
// CHECK-NEXT:   %c56 = constant 56 : index
// CHECK-NEXT:   br bb5(%c7 : index)
// CHECK-NEXT: bb5(%2: index):	// 2 preds: bb4, bb6
// CHECK-NEXT:   %3 = cmpi "slt", %2, %c56 : index
// CHECK-NEXT:   cond_br %3, bb6, bb7
// CHECK-NEXT: bb6:	// pred: bb5
// CHECK-NEXT:   call @body2(%0, %2) : (index, index) -> ()
// CHECK-NEXT:   %c2 = constant 2 : index
// CHECK-NEXT:   %4 = addi %2, %c2 : index
// CHECK-NEXT:   br bb5(%4 : index)
// CHECK-NEXT: bb7:	// pred: bb5
// CHECK-NEXT:   call @mid(%0) : (index) -> ()
// CHECK-NEXT:   br bb8
// CHECK-NEXT: bb8:	// pred: bb7
// CHECK-NEXT:   %c18 = constant 18 : index
// CHECK-NEXT:   %c37 = constant 37 : index
// CHECK-NEXT:   br bb9(%c18 : index)
// CHECK-NEXT: bb9(%5: index):	// 2 preds: bb8, bb10
// CHECK-NEXT:   %6 = cmpi "slt", %5, %c37 : index
// CHECK-NEXT:   cond_br %6, bb10, bb11
// CHECK-NEXT: bb10:	// pred: bb9
// CHECK-NEXT:   call @body3(%0, %5) : (index, index) -> ()
// CHECK-NEXT:   %c3 = constant 3 : index
// CHECK-NEXT:   %7 = addi %5, %c3 : index
// CHECK-NEXT:   br bb9(%7 : index)
// CHECK-NEXT: bb11:	// pred: bb9
// CHECK-NEXT:   call @post(%0) : (index) -> ()
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %8 = addi %0, %c1 : index
// CHECK-NEXT:   br bb2(%8 : index)
// CHECK-NEXT: bb12:	// pred: bb2
// CHECK-NEXT:   return
// CHECK-NEXT: }
mlfunc @more_imperfectly_nested_loops() {
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
