// RUN: mlir-opt %s -inline | FileCheck %s
// RUN: mlir-opt %s -inline -mlir-print-debuginfo | FileCheck %s --check-prefix INLINE-LOC

// Inline a function that takes an argument.
func @func_with_arg(%c : i32) -> i32 {
  %b = addi %c, %c : i32
  return %b : i32
}

// CHECK-LABEL: func @inline_with_arg
func @inline_with_arg(%arg0 : i32) -> i32 {
  // CHECK-NEXT: addi
  // CHECK-NEXT: return

  %0 = call @func_with_arg(%arg0) : (i32) -> i32
  return %0 : i32
}

// Inline a function that has multiple return operations.
func @func_with_multi_return(%a : i1) -> (i32) {
  cond_br %a, ^bb1, ^bb2

^bb1:
  %const_0 = constant 0 : i32
  return %const_0 : i32

^bb2:
  %const_55 = constant 55 : i32
  return %const_55 : i32
}

// CHECK-LABEL: func @inline_with_multi_return() -> i32
func @inline_with_multi_return() -> i32 {
// CHECK-NEXT:    [[VAL_7:%.*]] = constant 0 : i1
// CHECK-NEXT:    cond_br [[VAL_7]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[VAL_8:%.*]] = constant 0 : i32
// CHECK-NEXT:    br ^bb3([[VAL_8]] : i32)
// CHECK:       ^bb2:
// CHECK-NEXT:    [[VAL_9:%.*]] = constant 55 : i32
// CHECK-NEXT:    br ^bb3([[VAL_9]] : i32)
// CHECK:       ^bb3([[VAL_10:%.*]]: i32):
// CHECK-NEXT:    return [[VAL_10]] : i32

  %false = constant 0 : i1
  %x = call @func_with_multi_return(%false) : (i1) -> i32
  return %x : i32
}

// Check that location information is updated for inlined instructions.
func @func_with_locations(%c : i32) -> i32 {
  %b = addi %c, %c : i32 loc("mysource.cc":10:8)
  return %b : i32 loc("mysource.cc":11:2)
}

// INLINE-LOC-LABEL: func @inline_with_locations
func @inline_with_locations(%arg0 : i32) -> i32 {
  // INLINE-LOC-NEXT: addi %{{.*}}, %{{.*}} : i32 loc(callsite("mysource.cc":10:8 at "mysource.cc":55:14))
  // INLINE-LOC-NEXT: return

  %0 = call @func_with_locations(%arg0) : (i32) -> i32 loc("mysource.cc":55:14)
  return %0 : i32
}


// Check that external functions are not inlined.
func @func_external()

// CHECK-LABEL: func @no_inline_external
func @no_inline_external() {
  // CHECK-NEXT: call @func_external()
  call @func_external() : () -> ()
  return
}
