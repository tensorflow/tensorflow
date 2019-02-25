// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

// Maps used in dynamic_for below.
// CHECK-DAG: #[[idmap:.*]] = (d0) -> (d0)
// CHECK-DAG: #[[diffmap:.*]] = (d0, d1) -> (d0 - d1)
// CHECK-DAG: #[[addmap:.*]] = (d0, d1) -> (d0 + d1)
// CHECK-DAG: #[[prodconstmap:.*]] = (d0) -> (d0 * 3)
// CHECK-DAG: #[[addconstmap:.*]] = (d0) -> (d0 + 3)
// CHECK-DAG: #[[composedmap:.*]] = (d0, d1) -> (d0 * 3 + d1)
// CHECK-DAG: #[[id2dmap:.*]] = (d0, d1) -> (d0, d1)

// This function will be detected by the test pass that will insert
// EDSC-constructed blocks with arguments forming an infinite loop.
// CHECK-LABEL: @blocks
func @blocks() {
  return
//CHECK:        %c42_i32 = constant 42 : i32
//CHECK-NEXT:   %c1234_i32 = constant 1234 : i32
//CHECK-NEXT:   br ^bb1(%c42_i32, %c1234_i32 : i32, i32)
//CHECK-NEXT: ^bb1(%0: i32, %1: i32):	// 2 preds: ^bb0, ^bb2
//CHECK-NEXT:   br ^bb2(%0, %1 : i32, i32)
//CHECK-NEXT: ^bb2(%2: i32, %3: i32):	// pred: ^bb1
//CHECK-NEXT:   %4 = addi %2, %3 : i32
//CHECK-NEXT:   br ^bb1(%2, %4 : i32, i32)
//CHECK-NEXT: }
}

// This function will be detected by the test pass that will insert an
// EDSC-constructed empty `for` loop that corresponds to
//   for %arg0 to %arg1 step 2
// before the `return` instruction.
// CHECK-LABEL: func @dynamic_for_func_args(%arg0: index, %arg1: index) {
// CHECK:  for %i0 = #[[idmap]](%arg0) to #[[idmap]](%arg1) step 3 {
// CHECK:  {{.*}} = affine.apply #[[prodconstmap]](%arg0)
// CHECK:  {{.*}} = affine.apply #[[composedmap]](%arg0, %arg1)
// CHECK:  {{.*}} = affine.apply #[[addconstmap]](%arg0)
func @dynamic_for_func_args(%arg0 : index, %arg1 : index) {
  return
}

// This function will be detected by the test pass that will insert an
// EDSC-constructed empty `for` loop that corresponds to
//   for (%arg0 - %arg1) to (%arg2 + %arg3) step 2
// before the `return` instruction.
//
// CHECK-LABEL: func @dynamic_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
// CHECK:  %[[step:.*]] = constant 2 : index
// CHECK:  %[[lb:.*]] = affine.apply #[[diffmap]](%arg0, %arg1)
// CHECK:  %[[ub:.*]] = affine.apply #[[addmap]](%arg2, %arg3)
// CHECK:  for %i0 = #[[idmap]](%[[lb]]) to #[[idmap]](%[[ub]]) step 2 {
func @dynamic_for(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  return
}

// These functions will be detected by the test pass that will insert an
// EDSC-constructed 1-D pointwise-add loop with assignments to scalars before
// the `return` instruction.
//
// CHECK-LABEL: func @assignments_1(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
// CHECK: for %[[iv:.*]] = 0 to 4 {
// CHECK:   %[[a:.*]] = load %arg0[%[[iv]]] : memref<4xf32>
// CHECK:   %[[b:.*]] = load %arg1[%[[iv]]] : memref<4xf32>
// CHECK:   %[[tmp:.*]] = mulf %[[a]], %[[b]] : f32
// CHECK:   store %[[tmp]], %arg2[%[[iv]]] : memref<4xf32>
func @assignments_1(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
  return
}

// CHECK-LABEL: func @assignments_2(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
// CHECK: for %[[iv:.*]] = {{.*}} to {{.*}} {
// CHECK:   %[[a:.*]] = load %arg0[%[[iv]]] : memref<?xf32>
// CHECK:   %[[b:.*]] = load %arg1[%[[iv]]] : memref<?xf32>
// CHECK:   %[[tmp:.*]] = mulf %[[a]], %[[b]] : f32
// CHECK:   store %[[tmp]], %arg2[%[[iv]]] : memref<?xf32>
func @assignments_2(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  return
}

// This function will be detected by the test pass that will insert an
// EDSC-constructed empty `for` loop with max/min bounds that correspond to
//   for max(%arg0, %arg1) to (%arg2, %arg3) step 1
// before the `return` instruction.
//
// CHECK-LABEL: func @max_min_for(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
// CHECK:  for %i0 = max #[[id2dmap]](%arg0, %arg1) to min #[[id2dmap]](%arg2, %arg3) {
func @max_min_for(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) {
  return
}

func @callee()
func @callee_args(index, index)
func @second_order_callee(() -> ()) -> (() -> (index))

// This function will be detected by the test pass that will insert an
// EDSC-constructed chain of indirect calls that corresponds to
//   @callee()
//   var x = @second_order_callee(@callee)
//   @callee_args(x, x)
// before the `return` instruction.
//
// CHECK-LABEL: @call_indirect
// CHECK: %f = constant @callee : () -> ()
// CHECK: %f_0 = constant @callee_args : (index, index) -> ()
// CHECK: %f_1 = constant @second_order_callee : (() -> ()) -> (() -> index)
// CHECK: call_indirect %f() : () -> ()
// CHECK: %0 = call_indirect %f_1(%f) : (() -> ()) -> (() -> index)
// CHECK: %1 = call_indirect %0() : () -> index
// CHECK: call_indirect %f_0(%1, %1) : (index, index) -> ()
func @call_indirect() {
  return
}
